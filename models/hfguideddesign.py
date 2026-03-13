import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from utils.load_model_complex import load_d3pm_checkpoint

from utils.generate import generate_d3pm

from utils.tokenizer import Tokenizer
from utils.hf_utils import make_model_runners
from pathlib import Path




checkpoint_path = "../checkpoints/best_complex_model_rank0.pt"
model, tokenizer, Q, Q_bar, timesteps = load_d3pm_checkpoint(
    checkpoint_path, device="cuda"
)

model = model.to("cuda")
seq_len = 14
timestep = 500
target_protein = ""
chain_id = "A" 
model_runner = make_model_runners(Path(""), save_all=True)

# init_seqs = "GGGGGGGGGGGGGG"
# fixed_positions = {6: "G", 7: "G", 8: "G", 9: "G", 10: "G", 11: "G", 12: "G", 13: "G"}


peptide_type = "cycle"  # "linear" or "cycle" or "disulfide"
guidance_type = "structure"  # "structure" or "permeability"

generated_sequence = generate_d3pm(
    model,
    tokenizer,
    Q,
    Q_bar,
    timestep,
    seq_len,
    batch_size=5,
    device="cuda",
    target_protein=target_protein,
    chain_id=chain_id,
    guidance_type=guidance_type,
    # init_seqs=init_seqs,
    # fixed_positions=fixed_positions,
    model_runner=model_runner,
    peptide_type=peptide_type,
    guidance_scale=50.0,
)

print("Generated sequence:", generated_sequence)