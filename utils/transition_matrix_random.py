import torch
import numpy as np

def _beta_schedule(num_timesteps, schedule='linear', start=1e-5, end=0.999, max=8):
    """Generate a beta schedule for the diffusion process."""
    if schedule == 'linear':
        betas = torch.linspace(start, end, num_timesteps)
    elif schedule == 'sohl-dickstein':
        betas = torch.linspace(0, num_timesteps-1, num_timesteps)
        betas = 1 / (num_timesteps - betas + 1)
    elif schedule == "cosine":
        betas = torch.linspace(np.pi / 2, 0, num_timesteps)
        betas = torch.cos(betas) * (end - start) + start
    elif schedule == "exp":
        betas = torch.linspace(0, max, num_timesteps)
        betas = torch.exp(betas) * (end - start) + start
    else:
        raise ValueError("Must select a valid schedule; ['linear', 'sohl-dickstein', 'cosine', 'exp']")
    return betas

def cumprod_matrix(matrices):
    """Compute the cumulative product of a list of matrices."""
    if not matrices:
        return []
    out = [matrices[0]]
    acc = matrices[0]
    for i in range(1, len(matrices)):
        acc = acc @ matrices[i]
        out.append(acc)
    return out

class DiffusionScheduler:
    def __init__(self, K):
        
        self.K = K

    def q_random_schedule(self, timesteps=500, schedule='sohl-dickstein'):
        """
        Generates a schedule of transition matrices with random (uniform) transitions.
        
        Args:
            timesteps (int): The number of diffusion timesteps.
            schedule (str): The beta schedule type ('linear', 'sohl-dickstein', 'cosine', 'exp').
            
        Returns:
            tuple: A tuple containing:
                - Q_prod (torch.Tensor): Cumulative product of transition matrices, shape [timesteps, K, K].
                - Q_t (torch.Tensor): Scheduled transition matrices, shape [timesteps, K, K].
        """
        print(f"Using Random schedule: {schedule}")
        
        betas = _beta_schedule(timesteps, schedule=schedule)
        
        Q_t = []  # List to hold scheduled matrices
        
        for i in range(len(betas)):
            # Off-diagonal elements: uniform probability scaled by beta
            q_non_diag = torch.ones((self.K, self.K)) / self.K * betas[i]
            # Diagonal elements: ensure row sums to 1
            norm_constant = (1 - q_non_diag.sum(dim=0, keepdim=True)) # Sum over rows for each column
            q_diag = torch.eye(self.K, dtype=torch.float32) * norm_constant
            R = q_diag + q_non_diag
            Q_t.append(R)
        
        Q_prod = cumprod_matrix(Q_t)
        Q_prod = torch.stack(Q_prod)  # Stack cumulative product matrices
        Q_t = torch.stack(Q_t)        # Stack scheduled matrices
        return Q_prod, Q_t

# Example Usage:
if __name__ == "__main__":
    K = 26 # Example alphabet size
    scheduler = DiffusionScheduler(K)
    
    Q_prod, Q_t = scheduler.q_random_schedule(timesteps=100, schedule='sohl-dickstein')
    
    print(f"Generated random schedule for {Q_t.shape[0]} timesteps.")
    print(f"Q_t shape: {Q_t.shape}")       # Expected: [100, K, K]
    print(f"Q_prod shape: {Q_prod.shape}") # Expected: [100, K, K]
    print(f"First transition matrix Q_t[0] (first few rows/cols):\n{Q_t[0, :5, :5]}")
    print(f"Last cumulative product matrix Q_prod[-1] (first few rows/cols):\n{Q_prod[-1, :5, :5]}")

    # Check row stochastic property for a few matrices
    print("\nChecking row sums for Q_t[0] (should be all ones):")
    print(Q_t[0].sum(dim=1))
    print("\nChecking row sums for Q_t[50] (should be all ones):")
    print(Q_t[50].sum(dim=1))