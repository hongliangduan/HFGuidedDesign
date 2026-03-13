import copy
import logging
from pathlib import Path

import colabfold.highfold_utils
from colabfold.highfold_utils import (
    append_binder_feature,
    do_predict,
    dump_features,
    get_model_runners,
    get_receptor_feature_from_seq,
    load_features,
)
from loguru import logger


def make_receptor_input(
    task_id: str,
    receptor_seq: list[str],
    cardinality: list[int],
    fixed_templates_index: list[int] = [],
    fixed_template_paths: list[str] = [],
    save_data: bool = True,
    save_dir: Path = None,
):
    """
    Create receptor input features from sequence.
    """
    if isinstance(receptor_seq, str):
        receptor_seq = [receptor_seq]
    elif isinstance(receptor_seq, list):
        receptor_seq = [seq for seq in receptor_seq if seq.strip()]

    receptor_data = get_receptor_feature_from_seq(
        receptor_seq,
        cardinality,
        True,
        fixed_templates_index,
        fixed_template_paths,
    )
    if save_data and save_dir is not None:
        receptor_data_path = save_dir / f"{task_id}_receptor_features.pkl"
        try:
            dump_features(receptor_data, receptor_data_path)
            print(f"Receptor data saved to {receptor_data_path}")
        except Exception as e:
            print(f"Error saving receptor data: {e}")

    return receptor_data


def load_receptor_input(receptor_data_path: Path):
    """
    Load receptor input features from a file.
    """
    if not receptor_data_path.exists():
        return None

    try:
        return load_features(receptor_data_path)
    except Exception as e:
        print(f"Error loading receptor data from {receptor_data_path}: {e}")
        return None


# def make_model_runners(model_dir: Path, save_all: bool = False):
#     """
#     Create model runners for the given models.
#     """
#     return get_model_runners(1, True, model_dir, save_all=save_all)

def make_model_runners(model_dir: Path, model_number: int = 1, save_all: bool = False):
    """
    Create model runners for the given models.
    """
    return get_model_runners(model_number, True, model_dir, save_all=save_all)


def get_binder_prediction(
    task_name,
    model_runners,
    receptor_features,
    peptide_sequence,
    index_ss,
    result_dir,
    use_relax=True,
    save_all=True,
):
    """
    Get binder prediction using the model runners.
    """
    logger.disable("alphafold")
    logger.disable("colabfold")
    module_logger = logging.getLogger("colabfold.highfold_utils")
    module_logger.setLevel(logging.ERROR)
    new_receptor = copy.deepcopy(receptor_features)
    input_feature = append_binder_feature(
        peptide_sequence, False, index_ss, new_receptor
    )

    # receptor_length = sum(input_feature["chain_length"][:-1])
    residue_index = input_feature["residue_index"] + 1
    residue_index = residue_index[: -input_feature["chain_length"][-1]]
    do_predict(
        task_id=task_name,
        result_dir=result_dir,
        feature_dict=input_feature,
        model_runners=model_runners,
        use_templates=True,
        use_relax=use_relax,
        save_all=save_all,
    )

    return (input_feature["chain_length"][:-1], residue_index)
