import os
import json
import numpy as np

def calculate_ipae(json_file_path, fasta_file_path, normalization_factor=31):
    """
    Calculate the inter-chain interaction iPAE value.

    Args:
        json_file_path (str): Path to the AlphaFold2 JSON file.
        fasta_file_path (str): Path to the corresponding FASTA file.
        normalization_factor (float): Normalization factor (default: 31).

    Returns:
        float: Normalized iPAE value.
    """

    # Check if the JSON file exists
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")

    # Load JSON file and extract the PAE matrix
    with open(json_file_path) as f:
        confidences = json.load(f)
        pae_matrix = np.array(confidences['pae'])

    # Read FASTA file and determine the length of the target chain
    target_chain_length = None
    with open(fasta_file_path) as fasta_file:
        for line in fasta_file:
            line = line.strip()
            if line.endswith(":"):  # Target chain header line
                target_chain_length = len(line) - 1
                break

    if target_chain_length is None:
        raise ValueError("Target chain header not found in FASTA file")

    # Compute inter-chain PAE values
    pae_interaction1 = np.mean(pae_matrix[:target_chain_length, target_chain_length:])
    pae_interaction2 = np.mean(pae_matrix[target_chain_length:, :target_chain_length])
    ipae = round((pae_interaction1 + pae_interaction2) / 2, 2)

    # Normalize iPAE value
    ipae_normalized = ipae / normalization_factor

    return ipae_normalized
