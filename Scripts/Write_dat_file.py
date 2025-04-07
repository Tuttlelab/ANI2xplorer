# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 15:08:43 2025

@author: bwb16179
"""

import numpy as np
import os
from ase import units

time = 0.5 * units.fs
energy = units.mol / units.kJ

dat_file_rc = """UNITS LENGTH=A TIME=fs ENERGY=kcal/mol
dAB: DISTANCE ATOMS={},{}
dAC: DISTANCE ATOMS={},{}
rc: CUSTOM ARG=dAB,dAC FUNC=x+y PERIODIC=NO
bb: RESTRAINT ARG=rc AT={} KAPPA={}
PRINT ARG=dAB,dAC,rc,bb.bias FILE=COLVAR_d_{}.dat STRIDE={}"""

dat_file_rc_R2 = """UNITS LENGTH=A TIME=ns ENERGY=kcal/mol
dAB: DISTANCE ATOMS={},{}
dAC: DISTANCE ATOMS={},{}
dAD: DISTANCE ATOMS={},{}
rc: CUSTOM ARG=dAB,dAC,dAD VAR=x,y,z FUNC=x+y+z PERIODIC=NO
bb: RESTRAINT ARG=rc AT={} KAPPA={}
PRINT ARG=dAB,dAC,dAD,rc,bb.bias FILE=COLVAR_d_{}.dat STRIDE={}"""


# =============================================================================
# =============================================================================
# # PLUMED INDEXING STARTS AT 1
# =============================================================================
# =============================================================================

# Reaction 1A
Reaction_1A_atoms = {
    "atom_A": 9,
    "atom_B": 17,
    "atom_C": 6,
    "atom_D": 15,
}


# Reaction 1B
Reaction_1B_atoms = {
    "atom_A": 1,
    "atom_B": 14,
    "atom_C": 4,
    "atom_D": 6,
}

# Reaction 1E
Reaction_2_atoms = {
    "atom_A": 1,
    "atom_B": 2,
    "atom_C": 3,
    "atom_D": 6,
    "atom_E": 7,
    "atom_F": 8,
}

# Reaction 1A

Reaction_1A = {
    "stride": 1,
    "R1_start": 2.88,
    "R1_end": 1.00,  # 0.96
    "R2_start": 4.45,
    "R2_end": 1.00,  # 0.96
    "windows": 50,
    "destination_folder": "../Working_Folder/Reaction_1A",
}
Reaction_1A_rc = np.linspace(
    Reaction_1A["R1_start"], Reaction_1A["R1_end"], num=Reaction_1A["windows"]
) + np.linspace(
    Reaction_1A["R2_start"], Reaction_1A["R2_end"], num=Reaction_1A["windows"]
)


# Reaction 1B

Reaction_1B = {
    "stride": 1,
    "R1_start": 3.54,
    "R1_end": 1.00,  # 0.96
    "R2_start": 2.59,
    "R2_end": 1.00,  # 0.96
    "windows": 50,
    "destination_folder": "../Working_Folder/Reaction_1B",
}
Reaction_1B_rc = np.linspace(
    Reaction_1B["R1_start"], Reaction_1B["R1_end"], num=Reaction_1B["windows"]
) + np.linspace(
    Reaction_1B["R2_start"], Reaction_1B["R2_end"], num=Reaction_1B["windows"]
)

# Reaction 2

Reaction_2 = {
    "stride": 1,
    "R1_start": 1.93,
    "R1_end": 0.90,  # 0.96
    "R2_start": 1.60,
    "R2_end": 0.90,  # 0.96
    "R3_start": 3.32,
    "R3_end": 1.80,  # 2.05
    "windows": 50,
    "destination_folder": "../Working_Folder/Reaction_2",
}
Reaction_2_rc = (
    np.linspace(
        Reaction_2["R1_start"], Reaction_2["R1_end"], num=Reaction_2["windows"]
    )
    + np.linspace(
        Reaction_2["R2_start"], Reaction_2["R2_end"], num=Reaction_2["windows"]
    )
    + np.linspace(
        Reaction_2["R3_start"], Reaction_2["R3_end"], num=Reaction_2["windows"]
    )
)


if "__main__" == __name__:

    selected_reaction = "1B"
    kappa = 250  # kcal/mol/A2
    Run = ""

    reactions = {
        "1A": (Reaction_1A, Reaction_1A_atoms, Reaction_1A_rc, dat_file_rc),
        "1B": (Reaction_1B, Reaction_1B_atoms, Reaction_1B_rc, dat_file_rc),
        "2": (Reaction_2, Reaction_2_atoms, Reaction_2_rc, dat_file_rc_R2),
    }

    if selected_reaction not in reactions:
        raise ValueError(f"Invalid reaction name: {selected_reaction}")

    reaction, reaction_atoms, rc_values, dat_file = reactions[
        selected_reaction
    ]
    import sys

    # sys.exit()

    stride = reaction["stride"]
    windows = reaction["windows"]
    destination_folder = reaction["destination_folder"]

    for i in range(windows):
        window = i + 1
        rc = rc_values[i]

        print(f"Window {window}: RC = {rc}")

        folder = os.path.join(destination_folder, Run)
        os.makedirs(folder, exist_ok=True)
        window_folder = os.path.join(folder, f"window_{window}")
        os.makedirs(window_folder, exist_ok=True)

        plumed_file = os.path.join(window_folder, f"plumed_{window}.dat")

        with open(plumed_file, "w") as f:
            if "2" in selected_reaction:
                f.write(
                    dat_file_rc_R2.format(
                        Reaction_2_atoms["atom_A"],
                        Reaction_2_atoms["atom_B"],
                        Reaction_2_atoms["atom_C"],
                        Reaction_2_atoms["atom_D"],
                        Reaction_2_atoms["atom_E"],
                        Reaction_2_atoms["atom_F"],
                        rc,
                        kappa,
                        window,
                        stride,
                    )
                )
            else:

                atoms = (
                    Reaction_1A_atoms
                    if "1A" in selected_reaction
                    else Reaction_1B_atoms
                )

                f.write(
                    dat_file_rc.format(
                        atoms["atom_A"],
                        atoms["atom_B"],
                        atoms["atom_C"],
                        atoms["atom_D"],
                        rc,
                        kappa,
                        window,
                        stride,
                    )
                )
