import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.positional_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(x.size(0), -1)
        return self.positional_embedding(position_ids)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        if key_padding_mask is not None and key_padding_mask.dtype != torch.bool:
            key_padding_mask = key_padding_mask.to(torch.bool)
        out, _ = self.attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return out

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff)
        self.up_proj = nn.Linear(d_model, d_ff)
        self.down_proj = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        norm = x.norm(p=2, dim=-1, keepdim=True) * (x.size(-1) ** -0.5)
        return x / (norm + self.eps) * self.weight

class EnhancedTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = SwiGLU(d_model, d_ff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        attn_output = self.attention(self.norm1(x), key_padding_mask=key_padding_mask)
        x = x + self.dropout(attn_output)
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x

class AdaptiveNoiseScheduleEmbedding(nn.Module):
    def __init__(self, d_model, max_timesteps=1000):
        super().__init__()
        position = torch.arange(max_timesteps).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_timesteps, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.scale_proj = nn.Linear(d_model, d_model)
        self.bias_proj = nn.Linear(d_model, d_model)

    def forward(self, timestep):
        time_emb = self.pe[timestep]
        scale = torch.sigmoid(self.scale_proj(time_emb))
        bias = self.bias_proj(time_emb)
        return scale * time_emb + bias

class TransformerDenoiser(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=8, d_ff=3072, max_len=50,
                 dropout=0.1, use_gradient_checkpointing=False, max_timesteps=500):
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.vocab_size = vocab_size
        self.pad_id = vocab_size  
        self.token_embedding = nn.Embedding(vocab_size + 1, d_model)  
        self.position_embedding = LearnablePositionalEncoding(max_len, d_model)
        self.time_embedding = AdaptiveNoiseScheduleEmbedding(d_model, max_timesteps)
        self.layers = nn.ModuleList([
            EnhancedTransformerBlock(d_model, nhead, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.output_norm = RMSNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight[:-1], std=0.02)  
                nn.init.zeros_(module.weight[-1])  

    def forward(self, x, timestep, key_padding_mask=None):
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(token_emb)
        time_emb = self.time_embedding(timestep).unsqueeze(1)
        hidden = token_emb + pos_emb + time_emb
        for layer in self.layers:
            hidden = layer(hidden, key_padding_mask)
        hidden = self.output_norm(hidden)
        logits = self.output_proj(hidden)
        return logits
