import jax.numpy as jnp
import jax

from typing import Any, Dict

import numpy as np
from numpy.typing import NDArray

atom_types = [
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
]
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}


def softmax(arr, axis=-1) -> NDArray:

    arr = arr - np.max(arr, axis, keepdims=True)

    exp_arr = np.exp(arr)

    sum_exp = np.sum(exp_arr, axis, keepdims=True)

    softmax_result = exp_arr / sum_exp
    return softmax_result


def log_softmax(arr, axis=-1) -> NDArray:

    max_val = np.max(arr, axis, keepdims=True)
    arr = arr - max_val
    logsumexp = max_val + np.log(np.sum(np.exp(arr), axis, keepdims=True))
    log_softmax_result = arr - logsumexp

    return log_softmax_result


def elu(x, alpha=1.0):

    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))


def relu(x):

    return np.maximum(0, x)


def rg_loss(outputs, binder_len):
    xyz = outputs["structure_module"]
    ca = xyz["final_atom_positions"][:, atom_order["CA"]]
    ca = ca[-binder_len:]
    rg = np.sqrt(np.square(ca - ca.mean(0)).sum(-1).mean() + 1e-8)
    rg_th = 2.38 * ca.shape[0] ** 0.365
    return elu(rg - rg_th).tolist()


def termini_distance_loss(inputs, outputs, binder_len, threshold_distance=7.0):
    xyz = outputs["structure_module"]
    ca = xyz["final_atom_positions"][:, atom_order["CA"]]
    ca = ca[-binder_len:]  # Considering only the last _binder_len residues

    # Extract N-terminus (first CA atom) and C-terminus (last CA atom)
    n_terminus = ca[0]
    c_terminus = ca[-1]

    # Compute the distance between N and C termini
    termini_distance = np.linalg.norm(n_terminus - c_terminus)

    # Compute the deviation from the threshold distance using ELU activation
    deviation = elu(termini_distance - threshold_distance)

    # Ensure the loss is never lower than 0
    return relu(deviation)


def get_dgram_bins(outputs):
    dgram = outputs["distogram"]["logits"]
    if dgram.shape[-1] == 64:
        dgram_bins = np.append(0, np.linspace(2.3125, 21.6875, 63))
    if dgram.shape[-1] == 39:
        dgram_bins = np.linspace(3.25, 50.75, 39) + 1.25
    return dgram_bins


def get_contact_map(outputs, dist=8.0):

    dist_logits = outputs["distogram"]["logits"]
    dist_bins = get_dgram_bins(outputs)
    return (softmax(dist_logits) * (dist_bins < dist)).sum(-1)


def _get_con_loss(dgram, dgram_bins, cutoff=None, binary=True):
    """dgram to contacts"""
    if cutoff is None:
        cutoff = dgram_bins[-1]
    bins = dgram_bins < cutoff
    px = softmax((dgram))
    px_ = softmax(dgram - 1e7 * (1 - bins))
    # binary/cateogorical cross-entropy
    con_loss_cat_ent = -(px_ * log_softmax(dgram)).sum(-1)
    con_loss_bin_ent = -np.log((bins * px + 1e-5).sum(-1))
    return np.where(binary, con_loss_bin_ent, con_loss_cat_ent)


def binder_helicity_loss(inputs, outputs, target_len, binder_len):
    if "offset" in inputs:
        offset = inputs["offset"]
    else:
        idx = inputs["residue_index"].flatten()
        offset = idx[:, None] - idx[None, :]

    dgram = outputs["distogram"]["logits"]
    dgram_bins = get_dgram_bins(outputs)
    mask_2d = np.outer(
        np.append(np.zeros(target_len), np.ones(binder_len)),
        np.append(np.zeros(target_len), np.ones(binder_len)),
    )

    x = _get_con_loss(dgram, dgram_bins, cutoff=6.0, binary=True)
    if offset is None:
        if mask_2d is None:
            helix_loss = np.diagonal(x, 3).mean()
        else:
            helix_loss = np.diagonal(x * mask_2d, 3).sum() + (
                np.diagonal(mask_2d, 3).sum() + 1e-8
            )
    else:
        mask = offset == 3
        if mask_2d is not None:
            mask = np.where(mask_2d, mask, 0)
        helix_loss = np.where(mask, x, 0.0).sum() / (mask.sum() + 1e-8)

    return helix_loss


def get_con_loss(inputs, outputs, con_opt, mask_1d=None, mask_1b=None, mask_2d=None):

    # get top k
    def min_k(x, k=1, mask=None):
        y = np.sort(x if mask is None else np.where(mask, x, np.nan))
        k_mask = np.logical_and(np.arange(y.shape[-1]) < k, np.isnan(y) == False)
        return np.where(k_mask, y, 0).sum(-1) / (k_mask.sum(-1) + 1e-8)

    # decide on what offset to use
    if "offset" in inputs:
        offset = inputs["offset"]
    else:
        idx = inputs["residue_index"].flatten()
        offset = idx[:, None] - idx[None, :]

    dgram = outputs["distogram"]["logits"]
    dgram_bins = get_dgram_bins(outputs)

    p = _get_con_loss(
        dgram, dgram_bins, cutoff=con_opt["cutoff"], binary=con_opt["binary"]
    )
    if "seqsep" in con_opt:
        m = np.abs(offset) >= con_opt["seqsep"]
    else:
        m = np.ones_like(offset)

    # mask results
    if mask_1d is None:
        mask_1d = np.ones(m.shape[0])
    if mask_1b is None:
        mask_1b = np.ones(m.shape[0])

    if mask_2d is None:
        m = np.logical_and(m, mask_1b)
    else:
        m = np.logical_and(m, mask_2d)

    p = min_k(p, con_opt["num"], m)
    return min_k(p, con_opt["num_pos"], mask_1d)


def mask_loss(x: NDArray, mask: NDArray = None):
    if mask is None:
        return x.mean()
    else:
        x_masked = (x * mask).sum() / (1e-8 + mask.sum())
        return x_masked


def get_pae(outputs: Dict[str, Any]):
    prob = softmax(outputs["predicted_aligned_error"]["logits"], -1)
    breaks = outputs["predicted_aligned_error"]["breaks"]
    step = breaks[1] - breaks[0]
    bin_centers = breaks + step / 2
    bin_centers = np.append(bin_centers, bin_centers[-1] + step)
    return (prob * bin_centers).sum(-1)


def get_plddt(outputs):
    logits = outputs["predicted_lddt"]["logits"]
    num_bins = logits.shape[-1]
    bin_width = 1.0 / num_bins
    bin_centers = np.arange(start=0.5 * bin_width, stop=1.0, step=bin_width)
    probs = softmax(logits, axis=-1)
    return np.sum(probs * bin_centers[None, :], axis=-1)


def get_plddt_loss(outputs, mask_1d=None):
    p = 1 - get_plddt(outputs)
    return mask_loss(p, mask_1d)


def get_pae_loss(outputs, mask_1d=None, mask_1b=None, mask_2d=None):
    p = outputs["predicted_aligned_error"] / 31.0
    p = (p + p.T) / 2
    L = p.shape[0]
    if mask_1d is None:
        mask_1d = np.ones(L)
    if mask_1b is None:
        mask_1b = np.ones(L)
    if mask_2d is None:
        mask_2d = np.ones((L, L))
    mask_2d = mask_2d * mask_1d[:, None] * mask_1b[None, :]
    return mask_loss(p, mask_2d)


def get_contact_probs(logits: np.ndarray, breaks: np.ndarray) -> np.ndarray:
    threshold = 5.0 + 1e-3
    bin_tops = jnp.append(breaks, breaks[-1] + (breaks[-1] - breaks[-2]))
    is_contact_bin = 1.0 * (bin_tops <= threshold)
    probs = jax.nn.softmax(logits, axis=-1)
    contact_probs = jnp.einsum(
        "ijk,k->ij", probs, is_contact_bin, precision=jax.lax.Precision.HIGHEST
    )
    return np.array(contact_probs)


def get_ipae(predicted_aligned_error, mask_1d=None, mask_1b=None, mask_2d=None):
    p = predicted_aligned_error / 31.0
    p = (p + p.T) / 2
    L = p.shape[0]
    if mask_1d is None:
        mask_1d = np.ones(L)
    if mask_1b is None:
        mask_1b = np.ones(L)
    if mask_2d is None:
        mask_2d = np.ones((L, L))
    mask_2d = mask_2d * mask_1d[:, None] * mask_1b[None, :]
    x_masked = (p * mask_2d).sum() / (1e-8 + mask_2d.sum())
    return x_masked
