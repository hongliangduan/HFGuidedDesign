import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import csv
import math

from utils.loss import CombinedLoss
from utils.tokenizer import Tokenizer
from utils.transition_matrix_random import DiffusionScheduler
from complexes_denoiser import ComplexTransformerDenoiser 
from complexes_denoiser import migrate_weights
from peptides_denoiser import TransformerDenoiser  

# ========== Dataset ==========
class ComplexDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.receptor_seqs = df["Receptor Sequence"].tolist()
        self.peptide_seqs = df["Peptide Sequence"].tolist()
        
    def __len__(self):
        return len(self.receptor_seqs)
    
    def __getitem__(self, idx):
        return self.receptor_seqs[idx], self.peptide_seqs[idx]

# ========== Padding Helper ==========
def _pad(seqs, pad_id, max_len=None):
    seqs_list = [s.tolist() if isinstance(s, torch.Tensor) else s for s in seqs]
    if max_len is None:
        max_len = max(len(s) for s in seqs_list)
    return torch.tensor([s + [pad_id] * (max_len - len(s)) for s in seqs_list], dtype=torch.long)

# ========== Sample Transition ==========
def sample_transition_matrix(x_0, Q_bar_t):
    q_x = x_0 @ Q_bar_t
    x_t = torch.multinomial(q_x, num_samples=1).squeeze(-1)
    return x_t, q_x

# ========== Collater ==========
class ComplexCollater:
    def __init__(self, tokenizer, Q, Q_bar, num_timesteps=500, max_len_rec=500, max_len_pep=50):
        self.tokenizer = tokenizer
        self.Q = Q
        self.Q_bar = Q_bar
        self.K = tokenizer.K
        self.num_timesteps = num_timesteps
        self.pad_id = tokenizer.vocab_size
        self.max_len_rec = max_len_rec
        self.max_len_pep = max_len_pep

    def __call__(self, batch):
        receptor_seqs, peptide_seqs = zip(*batch)
        B = len(batch)


        rec_tokenized = [self.tokenizer.tokenize(seq)[:self.max_len_rec] for seq in receptor_seqs]
        rec_lengths = [len(t) for t in rec_tokenized]
        rec_max_len = min(max(rec_lengths), self.max_len_rec) 
        rec_tensor = _pad(rec_tokenized, self.pad_id, max_len=rec_max_len)
        rec_mask = (rec_tensor != self.pad_id).float()  


        pep_tokenized = [self.tokenizer.tokenize(seq)[:self.max_len_pep] for seq in peptide_seqs]
        pep_lengths = [len(t) for t in pep_tokenized]
        pep_max_len = min(max(pep_lengths), self.max_len_pep)


        tgt_pep = _pad(pep_tokenized, self.pad_id, max_len=pep_max_len)
        tgt_onehot = self.tokenizer.one_hot(tgt_pep, pad_id=self.pad_id)

        x_t_onehot = torch.zeros_like(tgt_onehot)
        q_x = torch.zeros_like(tgt_onehot)
        src_pep = torch.full((B, pep_max_len), self.pad_id, dtype=torch.long)
        timesteps = torch.zeros(B, dtype=torch.long)
        pep_mask = (tgt_pep != self.pad_id).float()  

        for i in range(B):
            L = pep_lengths[i]  
            

            t_probs = (np.arange(1, self.num_timesteps, dtype=np.float32)) ** (-0.8)
            t_probs /= t_probs.sum()
            t = np.random.choice(np.arange(1, self.num_timesteps), p=t_probs)
            timesteps[i] = t

            # t = np.random.randint(1, self.num_timesteps)
            # timesteps[i] = t


            x_0_i = tgt_onehot[i, :L]

            x_t_ids, q_i = sample_transition_matrix(x_0_i, self.Q_bar[t])
            x_t_onehot[i, :L] = F.one_hot(x_t_ids, num_classes=self.K).float()
            q_x[i, :L] = q_i
            src_pep[i, :L] = x_t_ids

        return {
            'receptor': rec_tensor,          # [B, L_rec]
            'receptor_mask': rec_mask,       # [B, L_rec]
            'peptide': src_pep,               # [B, L_pep]
            'timesteps': timesteps,           # [B]
            'target_pep': tgt_pep,            # [B, L_pep]
            'target_onehot': tgt_onehot,      # [B, L_pep, K]
            'q_x': q_x,                       # [B, L_pep, K]
            'x_t_onehot': x_t_onehot,         # [B, L_pep, K]
            'pep_mask': pep_mask              # [B, L_pep]
        }
    
# ========== Setup Dataloaders ==========
def setup_dataloaders(batch_size, data_path, num_timesteps, rank, world_size):
    dataset = ComplexDataset(data_path)
    
    train_data, temp_data = train_test_split(dataset, test_size=0.01, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    

    tokenizer = Tokenizer()
    diffusion_scheduler = DiffusionScheduler(K=tokenizer.K)
    Q_bar, Q = diffusion_scheduler.q_random_schedule(timesteps=num_timesteps)


    collater = ComplexCollater(
        tokenizer, Q, Q_bar, num_timesteps, 
        max_len_rec=500,  
        max_len_pep=50  
    )


    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_data, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_data, num_replicas=world_size, rank=rank, shuffle=False)


    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, collate_fn=collater, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, sampler=val_sampler, collate_fn=collater, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, sampler=test_sampler, collate_fn=collater, num_workers=4, pin_memory=True)

    return tokenizer, Q, Q_bar, train_loader, val_loader, test_loader

# ========== Training and Validation Functions ==========
def train_epoch(model, loader, criterion, optimizer, scheduler, device, Q, Q_bar, global_step, train_csv_writer, rank, epoch):
    
    model.train()
    epoch_loss = 0.0
    epoch_ce = 0.0
    epoch_lvb = 0.0

    pbar = tqdm(loader, desc=f"Training Epoch {epoch}", leave=False)
    
    for batch_idx, batch in enumerate(pbar):

        receptor = batch['receptor'].to(device)
        receptor_mask = batch['receptor_mask'].to(device)
        peptide = batch['peptide'].to(device)
        timesteps = batch['timesteps'].to(device)
        target_pep = batch['target_pep'].to(device)
        target_onehot = batch['target_onehot'].to(device)
        q_x = batch['q_x'].to(device)
        x_t_onehot = batch['x_t_onehot'].to(device)
        pep_mask = batch['pep_mask'].to(device)

        predictions = model(
            peptide=peptide,          
            receptor=receptor,        
            timestep=timesteps,      
            key_padding_mask_pep=(~pep_mask.bool()),  
            key_padding_mask_rec=(~receptor_mask.bool()) 
        )
        

        loss, ce_loss, lvb_loss = criterion(
            predictions, 
            x_t_onehot, 
            q_x, 
            target_pep, 
            target_onehot, 
            pep_mask, 
            timesteps, 
            Q, 
            Q_bar
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        epoch_ce += ce_loss.item()
        epoch_lvb += lvb_loss.item()

        if rank == 0:
            mean_t = timesteps.float().mean().item()
            train_csv_writer.writerow([
                global_step[0],
                epoch,
                loss.item(),
                ce_loss.item(),
                lvb_loss.item(),
                mean_t  
            ])

        global_step[0] += 1

        pbar.set_postfix({
            "step": global_step[0],
            "loss": f"{loss.item():.4f}",
            "ce": f"{ce_loss.item():.4f}",
            "lvb": f"{lvb_loss.item():.4f}",
        })

    avg_loss = epoch_loss / len(loader)
    avg_ce = epoch_ce / len(loader)
    avg_lvb = epoch_lvb / len(loader)
    return avg_loss, avg_ce, avg_lvb

def validate_epoch(model, loader, criterion, device, Q, Q_bar, epoch):
    model.eval()
    total_loss = 0.0
    total_ce = 0.0
    total_lvb = 0.0
    val_steps = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Validation Epoch {epoch}", leave=False)
        for batch in pbar:
            receptor = batch['receptor'].to(device)
            receptor_mask = batch['receptor_mask'].to(device)
            peptide = batch['peptide'].to(device)
            timesteps = batch['timesteps'].to(device)
            target_pep = batch['target_pep'].to(device)
            target_onehot = batch['target_onehot'].to(device)
            q_x = batch['q_x'].to(device)
            x_t_onehot = batch['x_t_onehot'].to(device)
            pep_mask = batch['pep_mask'].to(device)
            
            predictions = model(
                peptide=peptide,
                receptor=receptor,
                timestep=timesteps,
                key_padding_mask_pep=(~pep_mask.bool()),
                key_padding_mask_rec=(~receptor_mask.bool())
            )
            
            loss, ce_loss, lvb_loss = criterion(
                predictions, 
                x_t_onehot, 
                q_x, 
                target_pep, 
                target_onehot, 
                pep_mask, 
                timesteps, 
                Q, 
                Q_bar
            )
            
            total_loss += loss.item()
            total_ce += ce_loss.item()
            total_lvb += lvb_loss.item()
            val_steps += 1
            
            pbar.set_postfix({
                "val_loss": f"{loss.item():.4f}",
                "val_ce": f"{ce_loss.item():.4f}",
                "val_lvb": f"{lvb_loss.item():.4f}",
            })
    
    avg_loss = total_loss / val_steps
    avg_ce = total_ce / val_steps
    avg_lvb = total_lvb / val_steps
    return avg_loss, avg_ce, avg_lvb

def evaluate_test_set(model, test_loader, criterion, device, Q, Q_bar):
    model.eval()
    test_loss = 0.0
    test_ce = 0.0
    test_lvb = 0.0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing", leave=False)
        for batch in pbar:
            receptor = batch['receptor'].to(device)
            receptor_mask = batch['receptor_mask'].to(device)
            peptide = batch['peptide'].to(device)
            timesteps = batch['timesteps'].to(device)
            target_pep = batch['target_pep'].to(device)
            target_onehot = batch['target_onehot'].to(device)
            q_x = batch['q_x'].to(device)
            x_t_onehot = batch['x_t_onehot'].to(device)
            pep_mask = batch['pep_mask'].to(device)
            
            predictions = model(
                peptide=peptide,
                receptor=receptor,
                timestep=timesteps,
                key_padding_mask_pep=(~pep_mask.bool()),
                key_padding_mask_rec=(~receptor_mask.bool())
            )
            
            loss, ce_loss, lvb_loss = criterion(
                predictions, 
                x_t_onehot, 
                q_x, 
                target_pep, 
                target_onehot, 
                pep_mask, 
                timesteps, 
                Q, 
                Q_bar
            )
            
            test_loss += loss.item()
            test_ce += ce_loss.item()
            test_lvb += lvb_loss.item()
            
            pbar.set_postfix({
                "test_loss": f"{loss.item():.4f}",
                "test_ce": f"{ce_loss.item():.4f}",
                "test_lvb": f"{lvb_loss.item():.4f}"
            })
    
    avg_loss = test_loss / len(test_loader)
    avg_ce = test_ce / len(test_loader)
    avg_lvb = test_lvb / len(test_loader)
    
    print(f"\nTest Results:")
    print(f"  - Loss: {avg_loss:.4f}")
    print(f"  - CE: {avg_ce:.4f}")
    print(f"  - LVB: {avg_lvb:.4f}")
    return avg_loss, avg_ce, avg_lvb

# ========== Main Training Function ==========
def run_training(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    batch_size = 10
    num_timesteps = 500
    data_path = "../datasets/ComplexDatasets.csv"  


    tokenizer, Q, Q_bar, train_loader, val_loader, test_loader = setup_dataloaders(
        batch_size, data_path, num_timesteps, rank, world_size
    )
    Q, Q_bar = Q.to(rank), Q_bar.to(rank)


    model = ComplexTransformerDenoiser(
        vocab_size=tokenizer.K,
        d_model=512, 
        nhead=8, 
        num_layers=8, 
        d_ff=1024,   
        max_len_pep=50,   
        max_len_rec=500, 
        max_timesteps=num_timesteps
    ).to(rank)


    single_model = TransformerDenoiser(
        vocab_size=tokenizer.K,
        d_model=512,
        nhead=8,
        num_layers=8,
        d_ff=1024,
        max_len=50,
        max_timesteps=num_timesteps
    ).to(rank)
    pretrained_path = "../checkpoints/best_model_rank0_peptide.pt"
    if os.path.exists(pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        

        single_model.load_state_dict(state_dict)

    else:
        print(f" Warning: Pretrained weight file not found at {pretrained_path}")

    
    model = migrate_weights(single_model, model)


    for name, param in model.named_parameters():
        if 'position_embedding_pep' in name or 'self_attention' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True


    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("\n===== Model Parameters =====")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")

    model = DDP(model, device_ids=[rank])

    criterion = CombinedLoss(tokenizer=tokenizer, tmax=num_timesteps, lambda_weight=0.5)
    base_lr = 1e-6 
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.01)


    num_epochs = 50
    total_steps = len(train_loader) * num_epochs
    warmup_steps = 501
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float('inf')
    val_history = {'epochs': [], 'val_loss': [], 'val_ce': [], 'val_lvb': []}
    global_step = [0]

    if rank == 0:
        train_csv_file = open('./csv/training_steps_complex.csv', mode='w', newline='')
        train_csv_writer = csv.writer(train_csv_file)
        train_csv_writer.writerow(['Step', 'Epoch', 'Train_Loss', 'Train_CE', 'Train_LVB', 'Mean_t'])

        val_csv_file = open('validation_epochs.csv', mode='w', newline='')
        val_csv_writer = csv.writer(val_csv_file)
        val_csv_writer.writerow(['Epoch', 'Val_Loss', 'Val_CE', 'Val_LVB'])
    else:
        train_csv_writer = None
        val_csv_writer = None

    try:
        for epoch in range(1, num_epochs + 1):
            train_loader.sampler.set_epoch(epoch)


            train_loss, train_ce, train_lvb = train_epoch(
                model, train_loader, criterion, optimizer, scheduler,
                rank, Q, Q_bar, global_step, train_csv_writer, rank, epoch
            )


            val_loss, val_ce, val_lvb = validate_epoch(model, val_loader, criterion, rank, Q, Q_bar, epoch)

            if rank == 0:

                val_history['epochs'].append(epoch)
                val_history['val_loss'].append(val_loss)
                val_history['val_ce'].append(val_ce)
                val_history['val_lvb'].append(val_lvb)
                val_csv_writer.writerow([epoch, val_loss, val_ce, val_lvb])
                val_csv_file.flush()
                train_csv_file.flush()

                print(f"Epoch {epoch}/{num_epochs} (Step {global_step[0]})")
                print(f"  Train - Loss: {train_loss:.4f} | CE: {train_ce:.4f} | LVB: {train_lvb:.4f}")
                print(f"  Val   - Loss: {val_loss:.4f} | CE: {val_ce:.4f} | LVB: {val_lvb:.4f}")


                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'epoch': epoch,
                        'step': global_step[0],
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_loss': val_loss,
                        'tokenizer': tokenizer,
                        'Q': Q.cpu(),
                        'Q_bar': Q_bar.cpu(),
                        'timesteps': num_timesteps,
                        'model_config': {
                            'vocab_size': tokenizer.K,
                            'd_model': 512,
                            'nhead': 16,
                            'num_layers': 8,
                            'd_ff': 1024,
                            'max_len_pep': 50,
                            'max_len_rec': 500,
                            'dropout': 0.1,
                            'max_timesteps': num_timesteps
                        }
                    }, f"checkpoints/best_complex_model_rank{rank}.pt")  
                    print(f"Saved best complex model at epoch {epoch}")


        if rank == 0:
            test_loss, test_ce, test_lvb = evaluate_test_set(model, test_loader, criterion, rank, Q, Q_bar)
            

            final_checkpoint = {
                'epoch': num_epochs,
                'final_step': global_step[0],
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'tokenizer': tokenizer, 
                'Q': Q.cpu(),
                'Q_bar': Q_bar.cpu(),
                'timesteps': num_timesteps,
                'model_config': {
                    'vocab_size': tokenizer.K,
                    'd_model': 512,
                    'nhead': 8,
                    'num_layers': 8,
                    'd_ff': 1024,
                    'max_len_pep': 50,
                    'max_len_rec': 500,
                    'dropout': 0.1,
                    'max_timesteps': num_timesteps,
                },
                'val_history': val_history,
                'test_metrics': {
                    'test_loss': test_loss,
                    'test_ce': test_ce,
                    'test_lvb': test_lvb
                }
            }
            
            torch.save(final_checkpoint, "../checkpoints/final_complex_model.pt")
            print("Saved final complex model checkpoint to checkpoints/final_complex_model.pt")
            
    finally:
        if rank == 0:
            if train_csv_file:
                train_csv_file.close()
                print("Training CSV file closed.")
            if val_csv_file:
                val_csv_file.close()
                print("Validation CSV file closed.")

    dist.destroy_process_group()

# ========== Entry Point ==========
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs for distributed training.")
    mp.spawn(run_training, args=(world_size,), nprocs=world_size, join=True)