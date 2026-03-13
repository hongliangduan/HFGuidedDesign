import copy
import json
import os
import pickle
import shutil
import subprocess
import tarfile
from pathlib import Path
from time import time
from typing import Dict, Final, List, Optional, Tuple

import numpy as np
import requests
import yaml
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Residue import Residue
from loguru import logger
from numpy.typing import NDArray
from Bio.PDB.Structure import Structure
from design_loss import (
    binder_helicity_loss,
    get_con_loss,
    get_pae_loss,
    get_plddt_loss,
    rg_loss,
    termini_distance_loss,
    get_contact_probs,
    get_ipae,
)
from force_distance_constraint import DistanceConstraint
from hf_utils import get_binder_prediction

RESTYPES: Final[List[str]] = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]
EXTENDED_RESTYPES: Final[List[str]] = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
    "X",
]


ACID2RES_DICT = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}
RES2ACID_DICT = dict([val, key] for key, val in ACID2RES_DICT.items())

host_url = "https://api.colabfold.com"
headers = {}
use_pairing = False
submission_endpoint = "ticket/pair" if use_pairing else "ticket/msa"


def read_config_from_yaml(yaml_file):
    with open(yaml_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_cc_groups(cc_indexes, use_fixed_group=False, fixed_group=None):
    if use_fixed_group:
        fix_cc_indexes = []
        for i in fixed_group:
            fix_cc_indexes.append(cc_indexes[i])
        return [fix_cc_indexes]

    if len(cc_indexes) == 3:
        return [
            [cc_indexes[0], cc_indexes[1]],
            [cc_indexes[0], cc_indexes[2]],
            [cc_indexes[1], cc_indexes[2]],
        ]
    if len(cc_indexes) == 4:
        return [
            [cc_indexes[0], cc_indexes[1], cc_indexes[2], cc_indexes[3]],
            [cc_indexes[0], cc_indexes[2], cc_indexes[1], cc_indexes[3]],
            [cc_indexes[0], cc_indexes[3], cc_indexes[1], cc_indexes[2]],
        ]
    return [[cc_indexes[0], cc_indexes[1]]]


def CC_index(peptide, get_all: bool = False):
    indexes_of_c = []
    index = -1
    while True:
        index = peptide.find("C", index + 1)
        if index == -1:
            break
        indexes_of_c.append(index)
    C1 = indexes_of_c[0]
    C2 = indexes_of_c[-1]
    if get_all:
        return indexes_of_c
    return C2, C1


def parse_pdb_file(path: str) -> str:
    parser = PDBParser()
    structure = parser.get_structure("structure", path)
    res_list: List[Residue] = list(structure.get_chains())[0].child_list
    raw_res_dict = {}
    for i, res in enumerate(res_list):
        name = res.get_resname()
        index = res.get_id()[1]
        raw_res_dict[index] = RES2ACID_DICT[name]

    length = max(raw_res_dict.keys())
    seq = []
    for i in range(1, length + 1):
        if i in raw_res_dict.keys():
            seq.append(raw_res_dict[i])
        else:
            seq.append("X")
    return "".join(seq)


def CC_distance(peptide):
    C2, C1 = CC_index(peptide)
    return C2 - C1


def submit(seqs, mode, N=101):
    n, query = N, ""
    for seq in seqs:
        query += f">{n}\n{seq}\n"
        n += 1

    while True:
        error_count = 0
        try:
            res = requests.post(
                f"{host_url}/{submission_endpoint}",
                data={"q": query, "mode": mode},
                timeout=6.02,
                headers=headers,
            )
        except requests.exceptions.Timeout:
            continue
        except Exception as e:
            error_count += 1
            if error_count > 5:
                raise
            continue
        break

    try:
        out = res.json()
    except ValueError:
        out = {"status": "ERROR"}
    return out


def download(ID, path):
    error_count = 0
    while True:
        try:
            res = requests.get(
                f"{host_url}/result/download/{ID}", timeout=6.02, headers=headers
            )
        except requests.exceptions.Timeout:
            continue
        except Exception as e:
            error_count += 1
            if error_count > 5:
                raise
            continue
        break
    with open(path, "wb") as out:
        out.write(res.content)


def msa(path, seq):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
    x = seq
    tar_gz_file = f"{path}/out.tar.gz"
    mode = "pairgreedy"
    N = 101
    seqs_unique = []
    seqs = [x] if isinstance(x, str) else x
    [seqs_unique.append(x) for x in seqs if x not in seqs_unique]
    out = submit(seqs_unique, mode, N)
    ID, _ = out["id"], 0
    download(ID, tar_gz_file)


    with tarfile.open(tar_gz_file) as tar_gz:
        tar_gz.extractall(path)


    for file_name in os.listdir(path):
        if file_name.endswith(".sh"):
            os.remove(path + "/" + file_name)
        if file_name.endswith(".gz"):
            os.remove(path + "/" + file_name)
        if file_name.endswith(".m8"):
            os.remove(path + "/" + file_name)

    with open(f"{path}/uniref.a3m", "r") as file:
        lines = file.readlines()
    if lines:
        lines.pop()
    with open(f"{path}/uniref.a3m", "w") as file:
        file.writelines(lines)




def read_pdb_coordinates(file_path):
    protein_resno = []
    protein_atoms = []
    protein_atom_coords = []
    protein_res_name = []
    with open(file_path, "r") as pdb_file:
        for line in pdb_file:
            if line.startswith("ATOM"):
                resname = line[16:20].strip()
                resno = int(line[23:30])
                protein_resno.append(resno)
                atoms = line[12:16].strip()
                protein_atoms.append(atoms)
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                protein_atom_coords.append([x, y, z])
                protein_res_name.append(resname)

    return (
        np.array(protein_resno),
        np.array(protein_atoms),
        np.array(protein_atom_coords),
        np.array(protein_res_name),
    )


def find_peptide_index(arr):
    for i in range(len(arr) - 1, 0, -1):
        if arr[i] < arr[i - 1]:
            return i


def is_peptide_sequence_valid(
    peptide_sequence: str, cc_num: int = 2, strict: bool = False
) -> bool:
    """Check if a peptide sequence is valid about CC.

    Args:
        peptide_sequence (str): input str of peptide sequence
        strict (bool, optional): if limit CC number ==2 or >=2.

    Returns:
        bool: peptide sequence is valid or not
    """
    if not peptide_sequence:
        return False
    length = len(peptide_sequence)
    if strict:
        return (
            peptide_sequence.count("C") == cc_num
            and CC_distance(peptide_sequence) / length > 2 / 3
        )
    else:
        return (
            peptide_sequence.count("C") >= cc_num
            and CC_distance(peptide_sequence) / length > 2 / 3
        )


def copy_str_by_index(target: str, src: str, mask: NDArray) -> str:
    """Copy src into target by mask == 1.

    Args:
        target (str): target str
        src (str): src str
        mask (NDArray): mask array

    Returns:
        str: changed target str
    """
    target = list(target)
    for i, flag in enumerate(mask):
        if flag:
            target[i] = src[i]
    return "".join(target)


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


def matrix_softmax(input: NDArray) -> NDArray:
    exp = np.exp(input)
    return exp / (np.sum(exp, axis=1, keepdims=True) + 1e-6)


def is_peptide_with_mask(peptide_sequence: str) -> bool:
    """Check input peptide is incluede mask char 'X' or not.

    Args:
        peptide_sequence (str): Like 'ACXXXA' or 'ACADAAC'

    Returns:
        bool: is_peptide_with_mask 'X'
    """
    return peptide_sequence.count("X") > 0


def random_initialize_weights(
    peptide_length: int,
    peptide_mask_indexes: NDArray,
    cc_num: int = 2,
    initial_peptide_seq: Optional[str] = None,
) -> Tuple[NDArray, str]:
    """Initialize sequence probabilities"""

    if np.any(peptide_mask_indexes > 0):
        if initial_peptide_seq:
            # Fill 'C' in the unmasked index position of initial_peptide_seq to check initial_peptide_seq can get valid random sequence.
            free_sequence = "C" * peptide_length
            free_sequence = copy_str_by_index(
                free_sequence, initial_peptide_seq, peptide_mask_indexes
            )
            if not is_peptide_sequence_valid(free_sequence, cc_num=cc_num):
                raise ValueError(
                    "Input initial peptide sequence can not get valid sequence under input peptide_mask_indexes."
                )

    max_sample_times = 10000
    sample_times = 0
    restypes = np.array(RESTYPES)
    while True:

        weights = np.random.gumbel(0, 1, (peptide_length, 20))
        weights = matrix_softmax(weights)

        # Get the peptide sequence
        # Residue types
        random_peptide_sequence = "".join(restypes[np.argmax(weights, axis=1)])

        # only random the unlocked residues
        anti_mask = 1 - peptide_mask_indexes
        initial_peptide_seq = copy_str_by_index(
            initial_peptide_seq, random_peptide_sequence, anti_mask
        )

        # Prevent endless loop
        sample_times += 1
        if sample_times > max_sample_times:
            raise ValueError(
                "Randomly generation out of limit times: 10000, please check your input."
            )

        if is_peptide_sequence_valid(initial_peptide_seq, cc_num, True):
            return initial_peptide_seq


def mock_loss_input(target_len, binder_len, res_index, hotspot):
    inputs = {"opt": {}}
    inputs["opt"]["con"] = {
        "num": 2,
        "cutoff": 14.0,
        "binary": False,
        "seqsep": 9,
        "num_pos": float("inf"),
    }
    inputs["opt"]["i_con"] = {
        "num": 1,
        "cutoff": 21.6875,
        "binary": False,
        "num_pos": float("inf"),
    }
    inputs["opt"]["weights"] = {
        "pae": 0.1,
        "plddt": 0.1,
        "i_pae": 0.1,
        "con": 0.1,
        "i_con": 0.1,
    }
    if len(hotspot) > 0:
        inputs["opt"]["hotspot"] = hotspot

    inputs["seq_mask"] = np.ones((target_len + binder_len))

    inputs["residue_index"] = np.concatenate(
        [
            res_index,
            np.arange(target_len + 50, target_len + 50 + binder_len),
        ]
    )
    return inputs


def loss_binder(inputs, outputs, target_len, binder_len):
    """get losses"""
    opt = inputs["opt"]
    mask = inputs["seq_mask"]

    zeros = np.zeros_like(mask)
    tL, bL = target_len, binder_len

    binder_id = np.zeros_like(mask)
    binder_id[-bL:] = mask[-bL:]

    target_id = np.zeros_like(mask)
    if "hotspot" in opt:
        target_id[opt["hotspot"]] = mask[opt["hotspot"]]
        i_con_loss = get_con_loss(
            inputs, outputs, opt["i_con"], mask_1d=target_id, mask_1b=binder_id
        )
    else:
        target_id[:tL] = mask[:tL]
        i_con_loss = get_con_loss(
            inputs, outputs, opt["i_con"], mask_1d=binder_id, mask_1b=target_id
        )

    # unsupervised losses
    loss = {
        # "plddt": get_plddt_loss(outputs, mask_1d=binder_id),  # plddt over binder
        "pae": get_pae_loss(outputs, mask_1d=binder_id),  # pae over binder + interface
        "con": get_con_loss(
            inputs, outputs, opt["con"], mask_1d=binder_id, mask_1b=binder_id
        ),
        # interface
        "i_con": i_con_loss,
        "i_pae": get_pae_loss(outputs, mask_1d=binder_id, mask_1b=target_id),
        # "termini_distance_loss": termini_distance_loss(inputs, outputs, binder_len),
        "helix_loss": binder_helicity_loss(inputs, outputs, target_len, binder_len),
        "rg_loss": rg_loss(outputs, binder_len),
    }
    weight_loss = 1e-5
    weight_loss += loss["pae"] * 0.1
    weight_loss += loss["con"] * 1.0
    weight_loss += loss["i_con"] * 1.0
    weight_loss += loss["i_pae"] * 0.4
    weight_loss += loss["helix_loss"] * 0.3
    weight_loss += loss["rg_loss"] * 0.3
    return weight_loss


def is_cyclic_valid(
    cyclic_group: List[Tuple[int, int]],
    coord_dict: Dict[int, np.ndarray],
    name_dict: Dict[int, list[str]],
    bond_limit: Tuple[float, float],
) -> float:
    cyclic_norm = 0.0
    valid_num = 0
    for head, tail in cyclic_group:

        head_index = head + 1
        tail_index = tail + 1
        head_sg_coord = coord_dict[head_index][name_dict[head_index].index("SG")]
        tail_sg_coord = coord_dict[tail_index][name_dict[tail_index].index("SG")]
        act_dist = np.sqrt(np.square(head_sg_coord - tail_sg_coord).sum())
        if act_dist >= bond_limit[0] and act_dist <= bond_limit[1]:
            bond_limit = 0
            valid_num += 1
            continue
        cyclic_norm += min(
            np.abs(act_dist - bond_limit[0]), np.abs(act_dist - bond_limit[1])
        )

    if valid_num == len(cyclic_group):
        return 0

    return cyclic_norm / (len(cyclic_group) - valid_num)


def groups_predict_cycle(
    receptor_input,
    peptide_sequence,
    output_dir_base,
    receptor_if_residues,
    receptor_name,
    model_runners,
    num_iter,
    distance_constraints: List[DistanceConstraint],
    cc_groups: List[int] = None,
    use_relaxed_pdb: bool = True,
    gpu_id: int = 0,
):
    best_reward = 0
    best_cycic_norm = 0
    plddt = 0
    best_loss = 0
    best_index = 0
    stas_distance_constraints = []
    for i, cc_group in enumerate(cc_groups):
        loss, reward, cycic_norm, cur_plddt = predict_cycle(
            receptor_input,
            peptide_sequence,
            output_dir_base,
            receptor_if_residues,
            receptor_name,
            model_runners,
            num_iter,
            distance_constraints,
            cc_list=cc_group,
            group_index=i,
            use_relaxed_pdb=use_relaxed_pdb,
        )
        stas_distance_constraints.append(copy.deepcopy(distance_constraints))
        if cur_plddt > plddt:
            best_reward = reward
            best_cycic_norm = cycic_norm
            plddt = cur_plddt
            best_loss = loss
            best_index = i
    group_str = "_".join([str(i) for i in cc_groups[best_index]]) + f"_{best_index}"
    distance_constraints = stas_distance_constraints[best_index]
    return best_loss, best_reward, best_cycic_norm, plddt, "_" + group_str


def read_structure_coordinates(pdb_file: Path):
    parser = PDBParser(QUIET=True)
    structure: Structure = parser.get_structure("structure", pdb_file)[0]
    chain_coord_dict = {}
    chain_coord_name_dict = {}
    for i, chain in enumerate(structure):
        coord_dict = {}
        coord_name_dict = {}
        for residue in chain:
            atom_coords = []
            atom_names = []
            for atom in residue:
                atom_coords.append(atom.coord)
                atom_names.append(atom.get_name())
            coord_dict[residue.id[1]] = np.array(atom_coords)
            coord_name_dict[residue.id[1]] = atom_names

        chain_coord_dict[i] = coord_dict
        chain_coord_name_dict[i] = coord_name_dict

    return chain_coord_dict, chain_coord_name_dict


def get_pocket_dist(
    pocket: Dict[int, List[int]],
    chain_coord_dict: Dict[int, Dict[int, NDArray]],
    ligand_chain_id: int,
) -> float:
    """Calculate the distance between pocket and ligand."""

    pocket_coords = []
    for chain, pocket_indexes in pocket.items():
        for res_index in pocket_indexes:
            pocket_coords.append(chain_coord_dict[chain][res_index])
    # N * 3
    pocket_coords = np.concatenate(pocket_coords)
    # M * 3
    ligand_coords = np.concatenate(list(chain_coord_dict[ligand_chain_id].values()))

    coord1 = pocket_coords[:, np.newaxis, :]
    coord2 = ligand_coords[np.newaxis, :, :]
    delta = coord1 - coord2
    # N * M
    distances = np.linalg.norm(delta, axis=-1)
    cloest_dist = np.min(distances, axis=1)  # N

    return np.mean(cloest_dist)


def get_hotspot_precision(
    predict_hotspots: Dict[int, List[int]],
    gt_hotspots: Dict[int, List[int]],
) -> float:
    """Calculate the precision of hotspot pairs."""
    TP = 0
    FP = 0
    for chain, gt_hotspot in gt_hotspots.items():
        predict_hotspot = predict_hotspots.get(chain, [])
        TP += len(set(gt_hotspot) & set(predict_hotspot))
        FP += len(predict_hotspot) - TP
    if TP + FP == 0:
        return 0.0
    return TP / (TP + FP)


def get_predicted_hotspot_pair_dist(
    hotspot_pairs: List[Tuple[Tuple[int, int], Tuple[int, int]]],
    chain_coord_dict: Dict[int, Dict[int, NDArray]],
) -> float:
    """Calculate the distance between hotspot pairs."""
    total_distance = 0
    for (chain1, res1), (chain2, res2) in hotspot_pairs:
        coord1 = chain_coord_dict[chain1][res1][:, np.newaxis, :]
        coord2 = chain_coord_dict[chain2][res2][np.newaxis, :, :]
        delta = coord1 - coord2  # 形状 (2, 2, 3)
        distances = np.linalg.norm(delta, axis=-1)  # 形状 (2, 2)
        min_distance = np.min(distances)
        total_distance += min_distance

    if len(hotspot_pairs) > 0:
        return total_distance / len(hotspot_pairs)

    return 0.0


def make_index_ss(
    peptide_sequence: str,
    pep_head: int = 0,
    pep_tail: int = 0,
    cc_list: List[int] = None,
    is_nc_cyclic: bool = False,
) -> str:

    index_ss = []
    if cc_list is not None:
        for pair in zip(cc_list[::2], cc_list[1::2]):
            index_ss.append(pair)
        return index_ss

    else:
        if is_nc_cyclic:
            index_ss.append((0, len(peptide_sequence) - 1))
        else:
            index_ss.append((pep_head, pep_tail))
        return index_ss


def get_predict_hotspots(
    outputs,
    peptide_length,
    receptor_lengths,
    receptor_residue_index,
    ligand_chain_index,
):
    contact_probs = get_contact_probs(
        outputs["distogram"]["logits"], outputs["distogram"]["bin_edges"]
    )
    interface_mat = contact_probs[-peptide_length:]
    interface_mat[:, -peptide_length:] = 0.0
    interface_token = np.argmax(interface_mat, 1)
    token_chain_id = np.concatenate(
        [np.ones(chain_l) * i for i, chain_l in enumerate(receptor_lengths)]
    )
    interface_chain_index = token_chain_id[interface_token]
    interface_res_index = receptor_residue_index[interface_token]
    hotspot_pairs = []
    for i in range(len(interface_token)):
        hotspot_pair = (
            (interface_chain_index[i], interface_res_index[i]),
            (ligand_chain_index, i + 1),
        )
        hotspot_pairs.append(hotspot_pair)
    hotspot_dict = {}
    for (chain1, res1), (chain2, res2) in hotspot_pairs:
        hotspot_dict[chain1] = hotspot_dict.get(chain1, [])
        hotspot_dict[chain1].append(res1)
    return hotspot_pairs, hotspot_dict


def predict_cycle(
    receptor_input,
    peptide_sequence,
    output_dir_base,
    receptor_if_residues: dict[int, NDArray],
    receptor_name,
    model_runners,
    num_iter,
    distance_constraints: List[DistanceConstraint],
    pep_head: int = 0,
    pep_tail: int = 0,
    cc_list: List[int] = None,
    group_index: int = -1,
    use_relaxed_pdb: bool = True,
    is_nc_cyclic: bool = False,
):

    index_ss = make_index_ss(
        peptide_sequence, pep_head, pep_tail, cc_list, is_nc_cyclic
    )
    temp_result_dir = Path(output_dir_base).parent / "temp"
    if not temp_result_dir.exists():
        temp_result_dir.mkdir(parents=True)
    start_time = time()
    receptor_lengths, receptor_residue_index = get_binder_prediction(
        task_name=receptor_name,
        model_runners=model_runners,
        receptor_features=receptor_input,
        peptide_sequence=peptide_sequence,
        index_ss=index_ss,
        result_dir=temp_result_dir,
        use_relax=use_relaxed_pdb,
        save_all=True,
    )
    receptor_length = sum(receptor_lengths)
    logger.info(f"Prediction completed in {time() - start_time:.2f} seconds.")

    all_distance_available = True

    pdb_tag = "unrelaxed"
    if use_relaxed_pdb:
        pdb_tag = "relaxed"

    pdb_file = (
        temp_result_dir
        / f"{receptor_name}_{pdb_tag}_rank_001__multimer_v3_model_1_seed_000.pdb"
    )  # TLSP29_poc_unrelaxed_rank_001__multimer_v3_model_1_seed_000.pdb
    json_file = (
        temp_result_dir
        / f"{receptor_name}_scores_rank_001__multimer_v3_model_1_seed_000.json"
    )  # TLSP29_poc_scores_rank_001__multimer_v3_model_1_seed_000.json
    pkl_file = (
        temp_result_dir
        / f"{receptor_name}_all_rank_001__multimer_v3_model_1_seed_000.pickle"
    )  # TLSP29_poc_all_rank_001__multimer_v3_model_1_seed_000.pickle

    with open(json_file, "r") as fp:
        json_data = json.load(fp)

    with open(pkl_file, "rb") as fp:
        outputs = pickle.load(fp)

    plddt = sum(json_data["plddt"][-len(peptide_sequence) :]) / len(
        json_data["plddt"][-len(peptide_sequence) :]
    )
    pae = np.asarray(json_data["pae"])
    ipae = get_ipae(pae)
    ptm = json_data["ptm"]
    iptm = json_data["iptm"]

    coord_dict, name_dict = read_structure_coordinates(pdb_file)
    ligand_chain_index = len(receptor_lengths)

    # get cyclic_norm
    cycic_norm = 0.0
    if not is_nc_cyclic:
        cycic_norm = is_cyclic_valid(
            [(pep_head, pep_tail)],
            coord_dict[ligand_chain_index],
            name_dict[ligand_chain_index],
            bond_limit=(2.0, 2.3),
        )

    peptide_length = len(peptide_sequence)

    # get predict hotspot
    hotspot_pairs, hotspot_dict = get_predict_hotspots(
        outputs,
        peptide_length,
        receptor_lengths,
        receptor_residue_index,
        ligand_chain_index,
    )
    hotspot_distance = 0.0
    hotspot_precision = 0.0

    if len(receptor_if_residues) > 0:
        hotspot_distance = get_pocket_dist(
            receptor_if_residues, coord_dict, ligand_chain_index
        )
        hotspot_precision = get_hotspot_precision(hotspot_dict, receptor_if_residues)
    else:
        hotspot_distance = get_predicted_hotspot_pair_dist(hotspot_pairs, coord_dict)

    score = (
        (1 - ipae)
        + plddt * 0.02
        + 1 / (abs(hotspot_distance - 3.5) + 2.0)
        + ptm * 0.25
        + iptm * 0.25
        + hotspot_precision
    )
    score = score / 5.0

    relax_prefix = "unrelaxed_"
    if use_relaxed_pdb:
        relax_prefix = "relaxed_"

    if all_distance_available:
        if cc_list is not None:
            ss_indexs = "_".join([str(i) for i in cc_list])
            new_file = (
                Path(output_dir_base)
                / f"{relax_prefix}{num_iter}_{ss_indexs}_{group_index}.pdb"
            )
            shutil.copy(pdb_file, new_file)
        else:
            new_file = Path(output_dir_base) / f"{relax_prefix}{num_iter}.pdb"
            shutil.copy(pdb_file, new_file)

    pdb_file.unlink()
    json_file.unlink()
    pkl_file.unlink()
    return hotspot_distance, score, cycic_norm, plddt


def sequence_to_onehot(sequence, enable_extend: bool = False, max_length: int = 0):
    aatypes = RESTYPES
    if enable_extend:
        aatypes = EXTENDED_RESTYPES

    length = max(len(sequence), max_length)
    one_hot_arr = np.zeros((length, len(aatypes)), dtype=np.int32)
    for aa_index, aa_type in enumerate(sequence):
        aa_id = aatypes.index(aa_type)
        one_hot_arr[aa_index, aa_id] = 1
    return one_hot_arr


def onehot_to_sequence(onehot, enable_extend: bool = False):
    aatypes = RESTYPES
    if enable_extend:
        aatypes = EXTENDED_RESTYPES
    aatypes = np.array(aatypes)

    return "".join(aatypes[np.argmax(onehot, axis=1)])


def get_availables(
    state: NDArray,
    locked_mask: NDArray,
    enable_extend: bool = False,
    peptide_len: int = 0,
    allow_extra_C: bool = False,
) -> NDArray:
    """Get avaliables positions to mutate. unavailable positions are locked by locked_mask and current residues index in aatypes.

    Args:
        state (NDArray): input one-hot aatypes for residues.
        locked_mask (NDArray): locked_mask, 0 for available, 1 for locked.
        enable_extend (bool, optional): whether to enable extended peptide sequence. Defaults to False.

    Returns:
        NDArray: availables positions of state to mutate.
    """

    row, col = state.shape
    availables = np.arange(row * col)
    availables = availables.reshape(row, col)
    state_tamp = state.copy()
    if not allow_extra_C:
        # not allow to mutate into 'C'
        state_tamp[:, 4] = 1

    if enable_extend:
        # non-extended residues are not available to mutate into 'X'
        # extended residues are available to delete by mutating into 'X'
        delta_len = row - peptide_len
        split_index = np.flatnonzero(locked_mask == 0)[-1] + 1
        state_tamp[:, -1] = 1
        state_tamp[split_index - delta_len : split_index, -1] = 0

    # delete masked rows
    state_tamp[locked_mask == 1] = 1

    # select all availables residues
    availables: NDArray = availables[np.nonzero(state_tamp == 0)]

    return availables


def mutate_seq(
    peptide_sequence: str, ex_list: List[str], locked_mask: NDArray, cc_num: int = 2
) -> str:
    """Random mutate a peptide sequence except locked residues and peptide sequence not excuted.

    Args:
        peptide_sequence (peptide_sequence): input peptide sequence
        ex_list (List[str]): excuted peptides.
        locked_mask (NDArray): indexes of locked residues

    Returns:
        str: mutated peptide sequence.
    """
    restypes = np.array(RESTYPES)

    initial_peptide_seq = peptide_sequence
    seq_length = len(peptide_sequence)

    while True:
        weights = np.random.gumbel(0, 1, (seq_length, len(restypes)))
        weights = matrix_softmax(weights)
        # Get the peptide sequence
        # Residue types
        random_peptide_sequence = "".join(restypes[np.argmax(weights, axis=1)])
        anti_mask = 1 - locked_mask
        initial_peptide_seq = copy_str_by_index(
            initial_peptide_seq, random_peptide_sequence, anti_mask
        )
        # limit mutate seq is valid about CC
        if initial_peptide_seq not in ex_list and is_peptide_sequence_valid(
            initial_peptide_seq, cc_num, strict=True
        ):
            break
    return initial_peptide_seq


def mutate_extend_seq(
    peptide_sequence: str,
    ex_list: List[str],
    locked_mask: NDArray,
    init_len: int = 0,
    max_extend_length: int = 0,
    cc_num: int = 2,
) -> str:
    restypes = RESTYPES.copy()
    restypes.remove("C")
    extend_restypes = np.array(restypes + ["X"])
    restypes = np.array(restypes)

    delta_l = max_extend_length - init_len
    split_index = np.flatnonzero(locked_mask == 0)[-1]
    ex_start = split_index - delta_l + 1
    ex_end = split_index + 1

    unlocked_mask = 1 - locked_mask

    initial_peptide_seq = peptide_sequence
    while True:
        weights = np.random.gumbel(0, 1, (max_extend_length, len(restypes)))
        weights = matrix_softmax(weights)

        ex_weights = np.random.gumbel(0, 1, (delta_l, len(extend_restypes)))
        ex_weights = matrix_softmax(ex_weights)
        # Get the peptide sequence
        # Residue types

        random_list = restypes[np.argmax(weights, axis=1)]
        random_list[ex_start:ex_end] = extend_restypes[np.argmax(ex_weights, axis=1)]
        random_peptide_sequence = "".join(random_list)

        initial_peptide_seq = copy_str_by_index(
            initial_peptide_seq, random_peptide_sequence, unlocked_mask
        )
        # limit mutate seq is valid about CC
        if initial_peptide_seq not in ex_list and is_peptide_sequence_valid(
            initial_peptide_seq, cc_num, strict=True
        ):
            break
    return initial_peptide_seq


def get_locked_mask_from_seq(peptide_length: int, peptide_sequence: str) -> NDArray:
    """get locked mask from mutilated peptide sequence.

    Args:
        peptide_length (int): peptide length
        peptide_sequence (str): mutilated peptide sequence. e.g.  'XAXXXCXXCXX'

    Returns:
        NDArray: locked mask array. e.g. [0, 1, 1, 0, 0, 0, 1, 1, 1, 1] 1 mean locked , 0 mean not locked.
    """
    ligand_seq_locked_mask = np.ones(peptide_length, dtype=np.int64)
    if peptide_sequence:
        peptide_sequence_array = np.array(list(peptide_sequence))
        ligand_seq_locked_mask[peptide_sequence_array == "X"] = 0

    return ligand_seq_locked_mask


def get_locked_mask_from_flag(
    peptide_length: int, ligand_seq_locked_mask_index: List[int]
) -> NDArray:
    """Convert str mask list into np.array mask

    Args:
        peptide_length (int): input peptide length
        ligand_seq_locked_mask_index (List[int]): input peptide locked mask index str, like [0,1,2,3]

    Raises:
        ValueError: Any of input mask index out of peptide length.

    Returns:
        NDArray: locked mask array. 1 mean locked and 0 mean unlocked.
    """
    ligand_seq_locked_mask = np.zeros(peptide_length, dtype=np.int64)
    if len(ligand_seq_locked_mask_index) > 0:
        mask_indexes = np.array(ligand_seq_locked_mask_index, dtype=np.int32)

        # check input seq mask valid.
        if np.any(mask_indexes > peptide_length - 1):
            raise ValueError(
                f"Invalid residue index in args_ligand_seq_locked_mask: {ligand_seq_locked_mask_index}"
            )
        ligand_seq_locked_mask[mask_indexes] = 1

    return ligand_seq_locked_mask


def get_emphasize_locked_sequence_str(
    peptide_sequence: str, locked_mask: NDArray
) -> str:
    """get the colorful str in order to emphasize the locked residues.

    Args:
        peptide_sequence (str): peptide sequence
        locked_mask (NDArray): locked mask array

    Returns:
        str: emphasize locked sequence str. Residue C will be Red. Other Locked residues will be Blue.
    """
    peptide_sequence_char_list = list(peptide_sequence)
    if len(locked_mask) != len(peptide_sequence_char_list):
        delta_l = len(locked_mask) - len(peptide_sequence_char_list)
        split_index = np.flatnonzero(locked_mask == 0)[-1]
        locked_mask = list(locked_mask[: split_index - delta_l]) + list(
            locked_mask[split_index:]
        )
    for i, nce in enumerate(locked_mask):
        if nce == 1:
            if peptide_sequence_char_list[i] == "C":
                peptide_sequence_char_list[i] = (
                    f"<red>{peptide_sequence_char_list[i]}</red>"
                )
            else:
                peptide_sequence_char_list[i] = (
                    f"<blue>{peptide_sequence_char_list[i]}</blue>"
                )
    return "".join(peptide_sequence_char_list)
