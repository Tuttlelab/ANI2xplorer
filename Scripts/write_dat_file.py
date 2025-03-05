# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 15:08:43 2025

@author: bwb16179
"""

import numpy as np
import os

dat_file_rc_R1A_B = """UNITS LENGTH=A TIME=ns ENERGY=kcal/mol
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
# # REMINDER - PLUMED INDEXING STARTS AT 1
# =============================================================================
# =============================================================================

# =============================================================================
# #Reaction 1A
# atom_A = 9
# atom_B = 17
# atom_C = 6
# atom_D = 15
#
# =============================================================================


# Reaction 1B
atom_A = 1
atom_B = 14
atom_C = 4
atom_D = 6


# =============================================================================
# #Reaction 2
# atom_A = 1
# atom_B = 2
# atom_C = 3
# atom_D = 6
# atom_E = 7
# atom_F = 8
# =============================================================================


# =============================================================================
#
# # Reaction 1A
#
# stride = 1
# kappa = 25 # kcal/mol/A^2
#
# R1_start = 2.88
# R1_end = 1.34
#
# R2_start = 4.4448
# R2_end = 1.79
#
# windows = 30
#
# destination_folder = "Working_Folder/ANI/Reaction_1C/"
# Run = "17"
#
# R1_dists = np.linspace(R1_start,R1_end,num=windows)
# R2_dists = np.linspace(R2_start,R2_end,num=windows)
# =============================================================================

# =============================================================================
# # Reaction 2A
#
# stride = 1
# kappa = 25 # kcal/mol/A^2
#
# R1_start = 3.54
# R1_end = 1.36
#
# R2_start = 2.59
# R2_end = 1.34
#
# windows = 30
#
# destination_folder = "" # set destination folder
# Run = "" # set which simulation run this is
#
# R1_dists = np.linspace(R1_start,R1_end,num=windows)
# R2_dists = np.linspace(R2_start,R2_end,num=windows)
# =============================================================================

# Reaction 2

stride = 1
kappa = 50  # kcal/mol/A^2

R1_start = 1.93
R1_end = 0.96

R2_start = 1.60
R2_end = 0.96

R3_start = 3.32
R3_end = 2.05

windows = 30

destination_folder = ""  # set destination folder
Run = ""  # set which simulation run this is

R1_dists = np.linspace(R1_start, R1_end, num=windows)
R2_dists = np.linspace(R2_start, R2_end, num=windows)
R3_dists = np.linspace(R3_start, R3_end, num=windows)

for i in range(windows):
    window = i + 1
    rc = R1_dists[i] + R2_dists[i]  # Reaction 1A + 1B
    rc = R1_dists[i] + R2_dists[i] + R3_dists[i]  # Reaction 2

    kappa = 50

    print(rc)
    folder = destination_folder + Run
    if not os.path.exists(folder):
        os.mkdir(folder)
    if not os.path.exists(folder + f"/window_{window}"):
        os.mkdir(folder + f"/window_{window}")
    with open(f"{folder}/window_{window}/plumed_{window}.dat", "w") as f:

        # Reactions 1A + 1B
        f.write(
            dat_file_rc_R1A_B.format(
                atom_A, atom_B, atom_C, atom_D, rc, kappa, window, stride
            )
        )

    # =============================================================================
    #         # Reaction 2
    #         f.write(dat_file_rc_R2.format(atom_A, atom_B, atom_C, atom_D, atom_E, atom_F,
    #                                    rc, kappa, window, stride))
    # =============================================================================

    f.close()
