import sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append(os.path.dirname(__file__))
import shutil
import json
from pathlib import Path
from Bio.PDB import PDBParser
from i_PAEcalculation import calculate_ipae
from pre import make_index_ss
import subprocess
from hf_utils import (
    get_binder_prediction,
    make_model_runners,
    make_receptor_input,
    load_features,
)


AA_THREE_TO_ONE = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}


def extract_sequence_from_pdb(pdb_file, chain_id="A"):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    chain = next((model[chain_id] for model in structure if chain_id in model), None)
    if chain is None:
        raise ValueError(f"Chain {chain_id} not found.")
    sequence = ""
    for residue in chain:
        if residue.get_id()[0] == " ":
            res_name = residue.get_resname()
            sequence += AA_THREE_TO_ONE.get(res_name, "")
    return sequence


def get_or_create_input_receptor(pdb_path: str, chain_id: str, save_dir: str):
    pdb_id = Path(pdb_path).stem[:4]
    feature_path = Path(save_dir) / f"{pdb_id}_receptor_features.pkl"
    if feature_path.exists():
        input_receptor = load_features(feature_path)
    else:
        feature_path = Path(save_dir) / f"{pdb_id}_receptor_features.pkl"
        template_dir = Path(f"/home/fuxin/lab/hhm/templates/{pdb_id}")
        template_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(pdb_path, template_dir)

        print(f"Constructing receptor characteristics: {feature_path}")
        receptor_seq = extract_sequence_from_pdb(pdb_path, chain_id)
        input_receptor = make_receptor_input(
            task_id=pdb_id,
            receptor_seq=[receptor_seq],
            cardinality=[1],
            fixed_templates_index=[0],
            fixed_template_paths=[template_dir],
            save_data=True,
            save_dir=Path(save_dir),
        )
    return input_receptor


def write_fasta(input_fasta, target_seq, peptide_seq):
    combined_sequence = f"{target_seq}:\n{peptide_seq}"
    fasta_content = f">complex\n{combined_sequence}\n"
    with open(input_fasta, "w") as f:
        f.write(fasta_content)


def parse_prediction(output_dir, target_seq, peptide_seq, input_fasta):
    rank_file = next(
        (f for f in os.listdir(output_dir) if "rank_001" in f and f.endswith(".json")),
        None,
    )
    if rank_file is None:
        return None, None, None

    pdb_file = next(
        (
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if "unrelaxed_rank_001" in f and f.endswith(".pdb")
        ),
        None,
    )
    if pdb_file is None:
        return None, None, None

    with open(os.path.join(output_dir, rank_file)) as fp:
        data = json.load(fp)

    plddt = data.get("plddt", [])
    iptm = data.get("iptm")
    peptide_plddt = plddt[len(target_seq) : len(target_seq) + len(peptide_seq)]
    avg_peptide_plddt = (
        sum(peptide_plddt) / len(peptide_plddt) if peptide_plddt else None
    )

    try:
        ipae = calculate_ipae(os.path.join(output_dir, rank_file), input_fasta)
    except Exception as e:
        print(f"iPAE Calculation Failure: {e}")
        ipae = None

    return iptm, avg_peptide_plddt, ipae


def cyclepeptide_protein(
    target_protein,
    peptide_sequences,
    model_runner,
    chain_id="A",
    keep_temp=False,
):
    if isinstance(peptide_sequences, str):
        peptide_sequences = [peptide_sequences]

    pdb_id = Path(target_protein).stem[:4]
    target_sequence = extract_sequence_from_pdb(target_protein, chain_id)

    input_receptor_dir = "./input_receptor_dir"
    Path(input_receptor_dir).mkdir(parents=True, exist_ok=True)
    input_receptor = get_or_create_input_receptor(
        target_protein, chain_id, input_receptor_dir
    )

    temp_dir = Path(f"./result/custom_temp_{pdb_id}")
    temp_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for idx, peptide_sequence in enumerate(peptide_sequences):
        save_dir = temp_dir / str(peptide_sequence)
        save_dir.mkdir(parents=True, exist_ok=True)

        index_ss = make_index_ss(peptide_sequence=peptide_sequence, is_nc_cyclic=True)
        get_binder_prediction(
            f"{pdb_id}_{peptide_sequence}",
            model_runner,
            input_receptor,
            peptide_sequence=peptide_sequence,
            index_ss=index_ss,
            result_dir=save_dir,
            use_relax=False,
        )

        input_fasta = save_dir / "temp.fasta"
        write_fasta(input_fasta, target_sequence, peptide_sequence)

        iptm, plddt, ipae = parse_prediction(
            save_dir, target_sequence, peptide_sequence, input_fasta
        )

        results.append((iptm, plddt, ipae))

        if not keep_temp:
            shutil.rmtree(save_dir, ignore_errors=True)

    return results


def linearpeptide_protein(
    target_protein,
    peptide_sequences,
    model_runner,
    chain_id="B",
    keep_temp=False,
):
    if isinstance(peptide_sequences, str):
        peptide_sequences = [peptide_sequences]

    pdb_id = Path(target_protein).stem[:4]
    target_sequence = extract_sequence_from_pdb(target_protein, chain_id)

    input_receptor_dir = "./input_receptor_dir"
    Path(input_receptor_dir).mkdir(parents=True, exist_ok=True)
    input_receptor = get_or_create_input_receptor(
        target_protein, chain_id, input_receptor_dir
    )

    temp_dir = Path(f"./result/custom_temp_{pdb_id}")
    temp_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for idx, peptide_sequence in enumerate(peptide_sequences):
        save_dir = temp_dir / str(peptide_sequence)
        save_dir.mkdir(parents=True, exist_ok=True)

        index_ss = make_index_ss(peptide_sequence=peptide_sequence, is_nc_cyclic=False)
        get_binder_prediction(
            f"{pdb_id}_{peptide_sequence}",
            model_runner,
            input_receptor,
            peptide_sequence=peptide_sequence,
            index_ss=index_ss,
            result_dir=save_dir,
            use_relax=False,
        )

        input_fasta = save_dir / "temp.fasta"
        write_fasta(input_fasta, target_sequence, peptide_sequence)

        iptm, plddt, ipae = parse_prediction(
            save_dir, target_sequence, peptide_sequence, input_fasta
        )

        results.append((iptm, plddt, ipae))

        if not keep_temp:
            shutil.rmtree(save_dir, ignore_errors=True)

    return results


def sspeptide_protein(
    target_protein,
    peptide_sequences,
    model_runner,
    chain_id="A",
    keep_temp=False,
):
    if isinstance(peptide_sequences, str):
        peptide_sequences = [peptide_sequences]

    pdb_id = Path(target_protein).stem[:4]
    target_sequence = extract_sequence_from_pdb(target_protein, chain_id)

    input_receptor_dir = "./input_receptor_dir"
    Path(input_receptor_dir).mkdir(parents=True, exist_ok=True)
    input_receptor = get_or_create_input_receptor(
        target_protein, chain_id, input_receptor_dir
    )

    temp_dir = Path(f"./custom_temp_{pdb_id}")
    temp_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for idx, peptide_sequence in enumerate(peptide_sequences):
        save_dir = temp_dir / str(peptide_sequence)
        save_dir.mkdir(parents=True, exist_ok=True)
        cc_list = [1, 10]
        index_ss = make_index_ss(
            peptide_sequence=peptide_sequence, is_nc_cyclic=False, cc_list=cc_list
        )
        get_binder_prediction(
            f"{pdb_id}_{peptide_sequence}",
            model_runner,
            input_receptor,
            peptide_sequence=peptide_sequence,
            index_ss=index_ss,
            result_dir=save_dir,
            use_relax=False,
        )

        input_fasta = save_dir / "temp.fasta"
        write_fasta(input_fasta, target_sequence, peptide_sequence)

        iptm, plddt, ipae = parse_prediction(
            save_dir, target_sequence, peptide_sequence, input_fasta
        )

        results.append((iptm, plddt, ipae))

        if not keep_temp:
            shutil.rmtree(save_dir, ignore_errors=True)

    return results


if __name__ == "__main__":
    # subprocess.run(["git", "switch", "v1"], cwd="/home/fuxin/lab/hhm/colab_highfold")
    model_runner = make_model_runners(
        Path("./colab_highfold/colabfold"),
        save_all=True,
        model_number=5,
    )
    target_protein = ""

    # peptide_sequences = ["QGEKELG", "DFGKLRS", "DVIKFEV", "ETPVNQG", "ETPVNPG"]
    peptide_sequences = ["GSEYEEDGWTVLEPD"]
    # peptide_sequences = ["GRATKSIPPIAFDD"]

    results = cyclepeptide_protein(
        target_protein=target_protein,
        peptide_sequences=peptide_sequences,
        model_runner=model_runner,
        chain_id="A",
        keep_temp=True,
    )

    for seq, (iptm, plddt, ipae) in zip(peptide_sequences, results):
        print(f"Results for peptide: {seq}")
        print(f"  ipTM:       {iptm}")
        print(f"  pLDDT:      {plddt}")
        print(f"  iPAE:       {ipae}")
