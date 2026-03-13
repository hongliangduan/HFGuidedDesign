from dataclasses import dataclass
from typing import List

from numpy.typing import NDArray


@dataclass
class DistanceConstraint:
    receptor_atom: str  # eg. 92_ASP_OD1

    ligand_atom: str  # eg. ARG_NE
    limit_dist: float  # eg. 3.0

    act_dist: float = float("inf")
    is_available: bool = False

    @staticmethod
    def assign_distance_constraint(
        distance_constraint: "DistanceConstraint",
        col: List[str],
        rows: List[str],
        dist_matrix: NDArray,
    ) -> None:

        receptor_atom_index = col.index(distance_constraint.receptor_atom)
        ligand_atom_dist = [float("inf")]
        for i, row in enumerate(rows):
            if distance_constraint.ligand_atom in row:
                ligand_atom_dist.append(dist_matrix[i][receptor_atom_index])
        distance_constraint.act_dist = min(ligand_atom_dist)
        distance_constraint.is_available = (
            distance_constraint.act_dist < distance_constraint.limit_dist
        )
