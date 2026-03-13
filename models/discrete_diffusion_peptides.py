import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
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
from peptides_denoiser import TransformerDenoiser

# ========== Dataset ==========
class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

# ========== Padding Helper ==========
def _pad(seqs, pad_id):
    seqs_list = [s.tolist() if isinstance(s, torch.Tensor) else s for s in seqs]
    max_len = max(len(s) for s in seqs_list)
    return torch.tensor([s + [pad_id] * (max_len - len(s)) for s in seqs_list], dtype=torch.long)

# ========== Sample Transition ==========
def sample_transition_matrix(x_0, Q_bar_t):
    q_x = x_0 @ Q_bar_t
    x_t = torch.multinomial(q_x, num_samples=1).squeeze(-1)
    return x_t, q_x

# ========== Collater ==========

class DynamicTimestepSampler:
    def __init__(self, num_timesteps, strategy="importance"):
        self.num_timesteps = num_timesteps
        self.strategy = strategy
        self.epoch = 0
        
    def set_epoch(self, epoch):
        self.epoch = epoch
        
    def sample(self):
        if self.strategy == "importance":
            t_probs = (np.arange(1, self.num_timesteps, dtype=np.float32)) ** (-0.8)
            t_probs /= t_probs.sum()
            return np.random.choice(np.arange(1, self.num_timesteps), p=t_probs)
            
        elif self.strategy == "curriculum":
            max_t = max(100, min(500 - (self.epoch - 1) * 50, self.num_timesteps))
            return np.random.randint(1, max_t)
            
        elif self.strategy == "adaptive":
            if self.epoch < 5:
                focus_low, focus_high = 250, 500
            elif self.epoch < 10:
                focus_low, focus_high = 100, 250
            else:
                focus_low, focus_high = 1, 100

            if np.random.random() < 0.8:  
                return np.random.randint(focus_low, focus_high)
            else:
                return np.random.randint(1, self.num_timesteps)
                
        else: 
            return np.random.randint(1, self.num_timesteps)


class D3PMCollater:
    def __init__(self, tokenizer, Q, Q_bar, num_timesteps=500):
        self.tokenizer = tokenizer
        self.Q = Q
        self.Q_bar = Q_bar
        self.K = tokenizer.K
        self.num_timesteps = num_timesteps
        self.pad_id = tokenizer.vocab_size
        self.timestep_sampler = DynamicTimestepSampler(num_timesteps, strategy="importance")
    
    def set_epoch(self, epoch):
        self.timestep_sampler.set_epoch(epoch)

    def __call__(self, batch):
        tokenized = [self.tokenizer.tokenize(seq) for seq in batch]
        lengths = [len(t) for t in tokenized]
        max_len = max(lengths)
        B = len(batch)

        tgt = _pad([torch.tensor(seq, dtype=torch.long) for seq in tokenized], self.pad_id)  
        tgt_one_hot = self.tokenizer.one_hot(tgt, pad_id=self.pad_id)  
        x_t_onehot = torch.zeros_like(tgt_one_hot)
        q_x = torch.zeros_like(tgt_one_hot)
        src = torch.full((B, max_len), self.pad_id, dtype=torch.long)  
        timesteps = torch.zeros(B, dtype=torch.long)

        for i in range(B):
            L = lengths[i]

            t = self.timestep_sampler.sample()
            timesteps[i] = t

            x_0_i = tgt_one_hot[i, :L]
            x_t_ids, q_i = sample_transition_matrix(x_0_i, self.Q_bar[t])
            x_t_onehot[i, :L] = F.one_hot(x_t_ids, num_classes=self.K).float()
            q_x[i, :L] = q_i
            src[i, :L] = x_t_ids

        input_mask = (tgt != self.pad_id).float() 

        return src, timesteps, tgt, tgt_one_hot, q_x, x_t_onehot, input_mask
    
# ========== Setup Dataloaders ==========
def setup_dataloaders(batch_size, data_path, num_timesteps, rank, world_size):
    
    df = pd.read_parquet(data_path)
    sequences = df["Sequence"].dropna().tolist()
    sequences = [seq[:50] for seq in sequences if len(seq) > 0]

    train_seqs, temp_seqs = train_test_split(sequences, test_size=0.01, random_state=42)
    val_seqs, test_seqs = train_test_split(temp_seqs, test_size=0.5, random_state=42)
    

    tokenizer = Tokenizer()
    diffusion_scheduler = DiffusionScheduler(K=tokenizer.K)
    Q_bar, Q = diffusion_scheduler.q_random_schedule(timesteps=num_timesteps)

    train_dataset = SequenceDataset(train_seqs)
    val_dataset = SequenceDataset(val_seqs)
    test_dataset = SequenceDataset(test_seqs)

    collater = D3PMCollater(tokenizer, Q, Q_bar, num_timesteps=num_timesteps)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=collater, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, collate_fn=collater, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, collate_fn=collater, num_workers=4, pin_memory=True)

    return tokenizer, Q, Q_bar, train_loader, val_loader, test_loader
    
def train_epoch(model, loader, criterion, optimizer, scheduler, device, Q, Q_bar, global_step, train_csv_writer, rank, epoch):
    model.train()
    epoch_loss = epoch_ce = epoch_lvb = 0.0

    pbar = tqdm(loader, desc=f"Training Epoch {epoch}", leave=False)
    for batch_idx, batch in enumerate(pbar):
        src, timesteps, tgt, tgt_onehot, q_x, src_onehot, input_mask = batch
        src, tgt = src.to(device), tgt.to(device)
        src_onehot, q_x, tgt_onehot = src_onehot.to(device), q_x.to(device), tgt_onehot.to(device)
        input_mask, timesteps = input_mask.to(device), timesteps.to(device)

        predictions = model(src, timesteps, key_padding_mask=(~input_mask.bool()))
        loss, ce_loss, lvb_loss = criterion(predictions, src_onehot, q_x, tgt, tgt_onehot, input_mask, timesteps, Q, Q_bar)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()


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
            src, timesteps, tgt, tgt_onehot, q_x, src_onehot, input_mask = batch
            src, tgt = src.to(device), tgt.to(device)
            src_onehot, q_x, tgt_onehot = src_onehot.to(device), q_x.to(device), tgt_onehot.to(device)
            input_mask, timesteps = input_mask.to(device), timesteps.to(device)
            
            predictions = model(src, timesteps, key_padding_mask=(~input_mask.bool()))
            loss, ce_loss, lvb_loss = criterion(
                predictions, src_onehot, q_x, tgt, tgt_onehot, input_mask, timesteps, Q, Q_bar
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
            src, timesteps, tgt, tgt_onehot, q_x, src_onehot, input_mask = batch
            src, tgt = src.to(device), tgt.to(device)
            src_onehot, q_x, tgt_onehot = src_onehot.to(device), q_x.to(device), tgt_onehot.to(device)
            input_mask, timesteps = input_mask.to(device), timesteps.to(device)
            
            predictions = model(src, timesteps, key_padding_mask=(~input_mask.bool()))
            
            loss, ce_loss, lvb_loss = criterion(
                predictions, src_onehot, q_x, tgt, tgt_onehot, input_mask, timesteps, Q, Q_bar
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

    batch_size = 128
    num_timesteps = 500
    data_path = "../datasets/uniref50_lenlt50_clean.parquet"

    tokenizer, Q, Q_bar, train_loader, val_loader, test_loader = setup_dataloaders(
        batch_size, data_path, num_timesteps, rank, world_size
    )
    Q, Q_bar = Q.to(rank), Q_bar.to(rank)

    model = TransformerDenoiser(
        vocab_size=tokenizer.K, d_model=512, nhead=16, num_layers=8,
        d_ff=1024, max_timesteps=num_timesteps, max_len=50
    ).to(rank)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("\n===== Model Parameters =====")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")

    model = DDP(model, device_ids=[rank])

    criterion = CombinedLoss(tokenizer=tokenizer, tmax=num_timesteps, lambda_weight=0.5)
    base_lr = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.01)

    
    num_epochs = 5
    total_steps = len(train_loader) * num_epochs   # 8 = num_epochs
    warmup_steps = 500
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float('inf')
    # num_epochs = 8
    val_history = {'epochs': [], 'val_loss': [], 'val_ce': [], 'val_lvb': []}
    global_step = [0]

    if rank == 0:
        train_csv_file = open('./csv/training_steps.csv', mode='w', newline='')
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
            train_loader.collate_fn.set_epoch(epoch)
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
                            'max_len': 50,
                            'dropout': 0.1,
                            'max_timesteps': num_timesteps
                        }
                    }, f"../checkpoints/best_model_rank{rank}_peptide.pt")

        if rank == 0:
            test_loss, test_ce, test_lvb = evaluate_test_set(model, test_loader, criterion, rank, Q, Q_bar)
            
            # Prepare final checkpoint
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
                    'nhead': 16,
                    'num_layers': 8,
                    'd_ff': 1024,
                    'max_len': 50,
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
            
            # Save final checkpoint
            torch.save(final_checkpoint, "../checkpoints/final_model_checkpoint.pt")
            print("Saved final model checkpoint to checkpoints/final_model_checkpoint.pt")
            
 

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
    mp.spawn(run_training, args=(world_size,), nprocs=world_size)