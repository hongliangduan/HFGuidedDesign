import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizer import Tokenizer

# EPS = 1e-8

def sample_prior(a, b, _len=26):
    # returns uniform prior of shape (a, b)
    prior = torch.ones((a, b), dtype=torch.float32) / float(_len)
    return prior

class D3PMCELoss(nn.Module):
    def __init__(self, tokenizer: Tokenizer, reduction='mean'):
        super().__init__()
        self.tokenizer = tokenizer
        self.K = tokenizer.K
        self.ce = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, predictions, tgt, input_mask):
        """
        predictions: [B, L, vocab_size] (logits)
        tgt: [B, L] (long token ids)
        input_mask: [B, L] float or bool (1 for real tokens)
        """
        p = predictions[:, :, :self.K]  # logits for real tokens
        mask = input_mask.bool()  # [B,L]

        # pick unpadded positions
        p_unpadded = p[mask]          # shape [N, K]
        t_unpadded = tgt[mask]        # shape [N]

        # cross entropy expects logits and target ids
        ce_loss = self.ce(p_unpadded, t_unpadded)
        return ce_loss

class D3PMLVBLoss(nn.Module):
    def __init__(self, tokenizer: Tokenizer, tmax=500):
        super().__init__()
        self.tokenizer = tokenizer
        self.K = tokenizer.K
        self.tmax = tmax
        # no use of nn.KLDivLoss; compute KL explicitly: KL(q || p) = sum q * (log q - log p)

    def _normalize(self, probs):
        # probs: (..., K)
        # probs = probs.clamp(min=EPS)
        s = probs.sum(dim=-1, keepdim=True)
        # return probs / (s + EPS)
        return probs / (s)

    def forward(self, src_onehot, q, predictions, tgt, tgt_onehot, input_mask, timesteps, Q, Q_bar):
        """
        src_onehot: [B, L, K] one-hot of x_t
        q: [B, L, K] forward transition probabilities q(x_t | x_0) (i.e. rows sum to 1)
        predictions: [B, L, vocab_size] logits from model
        tgt: [B, L] token ids (x0)
        tgt_onehot: [B, L, K] one-hot of x0
        input_mask: [B, L] float
        timesteps: [B] long
        Q: [T+1, K, K] transition matrices? (or proper shape as in your code)
        Q_bar: [T+1, K, K] cumulative transition matrices
        """
        device = predictions.device
        B, L = tgt.shape
        p_logits = predictions[:, :, :self.K]
        # p = F.softmax(p_logits, dim=-1).clamp(min=EPS)  # [B,L,K]
        p = F.softmax(p_logits, dim=-1)
        total_losses = []

        mask_len = input_mask.sum(dim=1).long()  # [B]

        for i in range(B):
            D = int(mask_len[i].item())
            t = int(timesteps[i].item())
            if D == 0:
                # empty sequence fallback (shouldn't normally happen)
                total_losses.append(torch.tensor(0.0, device=device))
                continue

            x_t = src_onehot[i, :D].to(device)       # [D, K]
            x_0 = tgt_onehot[i, :D].to(device)       # [D, K]
            q_i = q[i, :D].to(device)                # [D, K]
            pred_i = p[i, :D].to(device)             # [D, K]

            if t == 1:
                # reconstruction CE: use logits directly
                ce = D3PMCELoss(self.tokenizer)
                r_loss = ce(predictions[i].unsqueeze(0), tgt[i].unsqueeze(0), input_mask[i].unsqueeze(0))
                total_losses.append(r_loss)
                continue

            if t == self.tmax:
               
                q_true = q_i
                q_true = self._normalize(q_true)
                prior = sample_prior(q_true.shape[0], q_true.shape[1], _len=self.K).to(device)  # [D, K]
                prior = self._normalize(prior)

                # kl = (q_true * (torch.log(q_true) - torch.log(prior + EPS))).sum(dim=1)  # [D]
                kl = (q_true * (torch.log(q_true) - torch.log(prior))).sum(dim=1)
                kl_loss = kl.mean()
                total_losses.append(kl_loss)
                continue


            Q_t = Q[t] if not torch.is_tensor(t) else Q[t]  
            Q_bar_t_minus1 = Q_bar[t-1]
            Q_bar_t = Q_bar[t]

            Q_t = Q_t.to(device).to(torch.float32)
            Q_bar_t_minus1 = Q_bar_t_minus1.to(device).to(torch.float32)
            Q_bar_t = Q_bar_t.to(device).to(torch.float32)

            A = torch.matmul(x_t, Q_t.transpose(0,1))   # [D, K]

            B = torch.matmul(x_0, Q_bar_t_minus1)      # [D, K]


            Q_expand = Q_bar_t_minus1.unsqueeze(0).expand(D, self.K, self.K)  # [D,K,K]
            B_pred = pred_i.unsqueeze(2) * Q_expand  # [D, K, K]

            q_t = A.unsqueeze(1) * B_pred  # [D, K, K]

            p_theta_marg = torch.bmm(q_t.transpose(1,2), pred_i.unsqueeze(2)).squeeze(-1)  # [D, K]
            # normalize
            p_theta_marg = self._normalize(p_theta_marg)


            num = A * B  


            denom_probs = torch.matmul(x_0, Q_bar_t)  # [D, K]
            denom = (denom_probs * x_t).sum(dim=1, keepdim=True)  
            
            # denom = denom.clamp(min=EPS)

            q_t_minus1 = num / denom  
            q_t_minus1 = self._normalize(q_t_minus1)

            # kl_per_pos = (q_t_minus1 * (torch.log(q_t_minus1) - torch.log(p_theta_marg + EPS))).sum(dim=1)  # [D]
            kl_per_pos = (q_t_minus1 * (torch.log(q_t_minus1) - torch.log(p_theta_marg))).sum(dim=1)
            kl_loss_i = kl_per_pos.mean()
            total_losses.append(kl_loss_i)


        losses = torch.stack(total_losses).to(device)
        lvb = losses.mean()
        return lvb

class CombinedLoss(nn.Module):
    def __init__(self, tokenizer: Tokenizer, tmax=500, lambda_weight=1.0):
        super().__init__()
        self.tokenizer = tokenizer
        self.tmax = tmax
        self.lambda_weight = lambda_weight
        self.ce_loss = D3PMCELoss(tokenizer=tokenizer)
        self.lvb_loss = D3PMLVBLoss(tokenizer=tokenizer, tmax=tmax)

    def forward(self, predictions, src_onehot, q, tgt, tgt_onehot, input_mask, timesteps, Q, Q_bar):
        ce_loss = self.ce_loss(predictions, tgt, input_mask)
        lvb_loss = self.lvb_loss(src_onehot, q, predictions, tgt, tgt_onehot, input_mask, timesteps, Q, Q_bar)
        total_loss = lvb_loss + self.lambda_weight * ce_loss
        return total_loss, ce_loss, lvb_loss