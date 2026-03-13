import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os


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

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        if key_padding_mask is not None and key_padding_mask.dtype != torch.bool:
            key_padding_mask = key_padding_mask.to(torch.bool)
        out, _ = self.attn(query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
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
        self.self_attention = MultiHeadAttention(d_model, nhead, dropout)
        self.cross_attention = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = SwiGLU(d_model, d_ff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm3 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, peptide, receptor, key_padding_mask_pep=None, key_padding_mask_rec=None):
        attn_output = self.self_attention(
            self.norm1(peptide), 
            self.norm1(peptide), 
            self.norm1(peptide),
            key_padding_mask=key_padding_mask_pep
        )
        peptide = peptide + self.dropout(attn_output)
        
        cross_output = self.cross_attention(
            query=self.norm2(peptide),
            key=self.norm2(receptor),
            value=self.norm2(receptor),
            key_padding_mask=key_padding_mask_rec
        )
        peptide = peptide + self.dropout(cross_output)
        
        peptide = peptide + self.dropout(self.feed_forward(self.norm3(peptide)))
        return peptide, receptor

class AdaptiveNoiseScheduleEmbedding(nn.Module):
    def __init__(self, d_model, max_timesteps=500):
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
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=8, d_ff=1024, max_len=50,
                 dropout=0.1, use_gradient_checkpointing=False, max_timesteps=500):
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.vocab_size = vocab_size
        self.pad_id = vocab_size  
        self.token_embedding = nn.Embedding(vocab_size + 1, d_model)  
        self.position_embedding = LearnablePositionalEncoding(max_len, d_model)
        self.time_embedding = AdaptiveNoiseScheduleEmbedding(d_model, max_timesteps)
        self.layers = nn.ModuleList([
            self.build_block(d_model, nhead, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.output_norm = RMSNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        self._init_weights()

    def build_block(self, d_model, nhead, d_ff, dropout):
        return nn.ModuleDict({
            'attention': MultiHeadAttention(d_model, nhead, dropout),
            'feed_forward': SwiGLU(d_model, d_ff),
            'norm1': RMSNorm(d_model),
            'norm2': RMSNorm(d_model),
            'dropout': nn.Dropout(dropout)
        })
    
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

            norm_x = layer.norm1(hidden)
            attn_output = layer.attention(norm_x, norm_x, norm_x, key_padding_mask=key_padding_mask)
            hidden = hidden + layer.dropout(attn_output)
            
            norm_x = layer.norm2(hidden)
            ff_output = layer.feed_forward(norm_x)
            hidden = hidden + layer.dropout(ff_output)
        
        hidden = self.output_norm(hidden)
        return self.output_proj(hidden)

class ComplexTransformerDenoiser(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=8, d_ff=1024, 
                 max_len_pep=50, max_len_rec=100, dropout=0.1, max_timesteps=500):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.pad_id = vocab_size
        
        self.token_embedding = nn.Embedding(vocab_size + 1, d_model)
        
        self.position_embedding_pep = LearnablePositionalEncoding(max_len_pep, d_model)
        self.position_embedding_rec = LearnablePositionalEncoding(max_len_rec, d_model)
        
        self.time_embedding = AdaptiveNoiseScheduleEmbedding(d_model, max_timesteps)
        
        self.layers = nn.ModuleList([
            EnhancedTransformerBlock(d_model, nhead, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.output_norm = RMSNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        
        self.init_new_params()

    def init_new_params(self):

        nn.init.normal_(self.position_embedding_rec.positional_embedding.weight, mean=0, std=0.02)
        
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.cross_attention.attn.in_proj_weight)
            nn.init.zeros_(layer.cross_attention.attn.in_proj_bias)
            nn.init.xavier_uniform_(layer.cross_attention.attn.out_proj.weight)
            if layer.cross_attention.attn.out_proj.bias is not None:
                nn.init.zeros_(layer.cross_attention.attn.out_proj.bias)
            
            nn.init.ones_(layer.norm2.weight)

    def forward(self, peptide, receptor, timestep, 
               key_padding_mask_pep=None, key_padding_mask_rec=None):
        """
        peptide: [B, L_pep]
        receptor: [B, L_rec]
        timestep: [B]
        """
        pep_emb = self.token_embedding(peptide)
        pep_pos = self.position_embedding_pep(pep_emb)
        time_emb = self.time_embedding(timestep).unsqueeze(1)
        hidden_pep = pep_emb + pep_pos + time_emb
        
        rec_emb = self.token_embedding(receptor)
        rec_pos = self.position_embedding_rec(rec_emb)
        hidden_rec = rec_emb + rec_pos
        
        for layer in self.layers:
            hidden_pep, hidden_rec = layer(
                hidden_pep, 
                hidden_rec,
                key_padding_mask_pep=key_padding_mask_pep,
                key_padding_mask_rec=key_padding_mask_rec
            )
        
        hidden_pep = self.output_norm(hidden_pep)
        return self.output_proj(hidden_pep)

def migrate_weights(single_model, complex_model):

    state_dict = {}
    single_dict = single_model.state_dict()
    
    for name in ['token_embedding.weight', 'output_proj.weight', 'output_norm.weight']:
        state_dict[name] = single_dict[name]
    
    state_dict['time_embedding.pe'] = single_dict['time_embedding.pe']
    state_dict['time_embedding.scale_proj.weight'] = single_dict['time_embedding.scale_proj.weight']
    state_dict['time_embedding.scale_proj.bias'] = single_dict['time_embedding.scale_proj.bias']
    state_dict['time_embedding.bias_proj.weight'] = single_dict['time_embedding.bias_proj.weight']
    state_dict['time_embedding.bias_proj.bias'] = single_dict['time_embedding.bias_proj.bias']
    
    state_dict['position_embedding_pep.positional_embedding.weight'] = \
        single_dict['position_embedding.positional_embedding.weight']
    
    num_layers = complex_model.num_layers
    for i in range(num_layers):
        prefix = f'layers.{i}.self_attention.attn.'
        single_prefix = f'layers.{i}.attention.attn.'
        state_dict[prefix + 'in_proj_weight'] = single_dict[single_prefix + 'in_proj_weight']
        state_dict[prefix + 'in_proj_bias'] = single_dict[single_prefix + 'in_proj_bias']
        state_dict[prefix + 'out_proj.weight'] = single_dict[single_prefix + 'out_proj.weight']
        state_dict[prefix + 'out_proj.bias'] = single_dict[single_prefix + 'out_proj.bias']
        
        ff_prefix = f'layers.{i}.feed_forward.'
        state_dict[ff_prefix + 'gate_proj.weight'] = single_dict[ff_prefix + 'gate_proj.weight']
        state_dict[ff_prefix + 'gate_proj.bias'] = single_dict[ff_prefix + 'gate_proj.bias']
        state_dict[ff_prefix + 'up_proj.weight'] = single_dict[ff_prefix + 'up_proj.weight']
        state_dict[ff_prefix + 'up_proj.bias'] = single_dict[ff_prefix + 'up_proj.bias']
        state_dict[ff_prefix + 'down_proj.weight'] = single_dict[ff_prefix + 'down_proj.weight']
        state_dict[ff_prefix + 'down_proj.bias'] = single_dict[ff_prefix + 'down_proj.bias']
        
        state_dict[f'layers.{i}.norm1.weight'] = single_dict[f'layers.{i}.norm1.weight']
        state_dict[f'layers.{i}.norm3.weight'] = single_dict[f'layers.{i}.norm2.weight']  
    
    complex_model.load_state_dict(state_dict, strict=False)
    return complex_model

if __name__ == "__main__":

    VOCAB_SIZE = 26  
    D_MODEL = 512
    NHEAD = 8
    NUM_LAYERS = 8
    D_FF = 1024
    MAX_LEN_PEP = 50
    MAX_LEN_REC = 100
    MAX_TIMESTEPS = 500
    

    single_model = TransformerDenoiser(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_len=MAX_LEN_PEP,
        max_timesteps=MAX_TIMESTEPS
    )
    
  
    checkpoint_path = "./checkpoints/best_model_rank0.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        single_model.load_state_dict(state_dict)
    else:
        print(f" Warning: Checkpoint file not found at {checkpoint_path}")

    
    complex_model = ComplexTransformerDenoiser(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_len_pep=MAX_LEN_PEP,
        max_len_rec=MAX_LEN_REC,
        max_timesteps=MAX_TIMESTEPS
    )
    
    complex_model = migrate_weights(single_model, complex_model)
    
    output_path = "./checkpoints/complex_denoiser.pt"
    torch.save({
        'model_state_dict': complex_model.state_dict(),
        'config': {
            'vocab_size': VOCAB_SIZE,
            'd_model': D_MODEL,
            'nhead': NHEAD,
            'num_layers': NUM_LAYERS,
            'd_ff': D_FF,
            'max_len_pep': MAX_LEN_PEP,
            'max_len_rec': MAX_LEN_REC,
            'max_timesteps': MAX_TIMESTEPS
        }
    }, output_path)
    
    complex_model.eval()
    with torch.no_grad():
        batch_size = 2
        peptide = torch.randint(0, VOCAB_SIZE, (batch_size, 30))
        receptor = torch.randint(0, VOCAB_SIZE, (batch_size, 80))
        timestep = torch.tensor([100, 200])
        
        output = complex_model(peptide, receptor, timestep)
