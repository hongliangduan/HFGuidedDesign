import torch
from models.complexes_denoiser import ComplexTransformerDenoiser
from tokenizer import Tokenizer
import torch.nn.functional as F


def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_key = k[7:]  
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict


def load_d3pm_checkpoint(checkpoint_path, device="cuda"):

    # checkpoint = torch.load(checkpoint_path, map_location=device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)


    tokenizer = Tokenizer()
    Q = checkpoint["Q"].to(device)
    Q_bar = checkpoint["Q_bar"].to(device)
    timesteps = checkpoint["timesteps"]

    config = checkpoint["model_config"]
    model = ComplexTransformerDenoiser(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        max_len_pep=config["max_len_pep"],
        max_len_rec=config["max_len_rec"],
        max_timesteps=config["max_timesteps"],
    ).to(device)

    state_dict = remove_module_prefix(checkpoint["model_state_dict"])
    model.load_state_dict(state_dict)
    model.eval()

    return model, tokenizer, Q, Q_bar, timesteps
