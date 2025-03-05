#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 11:18:22 2025

@author: bwb16179
"""

from ase.io import read
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import plumed
import glob, sys
import tqdm

Reaction_1A = "../Data/Reaction_1A/full_sim_windows/1k/*/*COLVAR*"
Reaction_1B = "../Data/Reaction_1B/full_sim_windows/1k/*/*COLVAR*"
Reaction_2A = "../Data/Reaction_2/full_sim_windows/1k/*/*COLVAR*"

E1A = "../Data/Reaction_1A/energies/Energies_R1A_1k.npy"
E1B = "../Data/Reaction_1B/energies/Energies_R1B_1k.npy"
E2 = "../Data/Reaction_2/energies/Energies_R2_1k.npy"

reactions = [Reaction_1A, Reaction_1B, Reaction_2A]
energies = [E1A, E1B, E2]


def read_dat_file(file_pattern):
    trajectories = sorted(glob.glob(file_pattern))
    cv = []
    bias = []
    RC1 = []
    RC2 = []
    RC3 = []
    for dat_file in tqdm.tqdm(trajectories):
        df = plumed.read_as_pandas(dat_file)
        cv.append(df.rc.to_list())
        bias.append(df.bias.to_list())
        RC1.append(df.dAB.to_list())
        RC2.append(df.dAC.to_list())
        if "2A" in file_pattern:
            RC3.append(df.dAD.to_list())
    if "2A" in file_pattern:
        return cv, bias, RC1, RC2, RC3
    else:
        return cv, bias, RC1, RC2


for i, reaction in enumerate(reactions):

    if "2A" in reaction:
        cv, bias, RC1, RC2, RC3 = read_dat_file(reaction)
        RC3 = np.array(RC3).flatten()
    else:
        cv, bias, RC1, RC2 = read_dat_file(reaction)

    RC1 = np.array(RC1).flatten()
    RC2 = np.array(RC2).flatten()

    # Load energy npy file

    energies = np.load(energies[i])
    energies = energies[i] - energies[i].min()

    if i == 1 or i == 2:

        with open(f"test_colour_map_{i}_x_100.xyz", "w") as f:
            f.write(f"{len(RC1)}\n")
            f.write("x = RC1, y = RC2, z= E in kcal/mol\n")
            for i, x in enumerate(RC1):
                f.write(f"C {RC1[i]*100} {RC2[i]*100} {energies[i]}\n")

        # plot RC1 vs RC2 vs E
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(projection="3d")
        sc = ax1.scatter(
            RC1, RC2, energies, c=energies, cmap="RdBu_r", alpha=0.8
        )
        cbar = plt.colorbar(sc, pad=0.1)
        cbar.set_label("Energy (kcal/mol)")

        ax1.set_title("Reaction 1A")
        ax1.set_xlabel("Bond length 1")
        ax1.set_ylabel("Bond Length 2")
