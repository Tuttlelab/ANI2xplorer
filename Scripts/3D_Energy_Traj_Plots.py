#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 11:18:22 2025

@author: bwb16179
"""

from ase.io import read
import numpy as np
import matplotlib.pyplot as plt
import plumed
import glob, sys
import tqdm


def read_dat_file(file_pattern):
    trajectories = sorted(glob.glob(file_pattern))
    cv = []
    bias = []
    RC1 = []
    RC2 = []
    for dat_file in tqdm.tqdm(trajectories):
        df = plumed.read_as_pandas(dat_file)
        cv.append(df.rc.to_list())
        bias.append(df.bias.to_list())
        RC1.append(df.dAB.to_list())
        RC2.append(df.dAC.to_list())

    return cv, bias, RC1, RC2


def read_dat_file_2(file_pattern):
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
        RC3.append(df.dAD.to_list())

    return cv, bias, RC1, RC2, RC3


numpy_energy_arrays = [
    np.load("Reaction_1A.npy"),
    np.load("Reaction_1B.npy"),
    np.load("Reaction_2.npy"),
]

reactions = [
    "*COLVAR*",
    "*COLVAR*",
    "*COLVAR*",
]

for i, reaction in enumerate(reactions):

    energies = numpy_energy_arrays[i]
    energies = energies[i] - energies[i].min()

    name = reaction.split("/")[2]

    if "_1A_" in reaction:
        cv, bias, RC1, RC2 = read_dat_file(reaction)

        RC1 = np.array(RC1).flatten()
        RC2 = np.array(RC2).flatten()
        x_label = "d(15-6)"
        y_label = "d(9-17)"
        x_data = RC1
        y_data = RC2

    elif "_1B_" in reaction:
        cv, bias, RC1, RC2 = read_dat_file(reaction)

        RC1 = np.array(RC1).flatten()
        RC2 = np.array(RC2).flatten()
        x_label = "d(1-14)"
        y_label = "d(9-17)"
        x_data = RC1
        y_data = RC2

    elif "_2_" in reaction:
        cv, bias, RC1, RC2, RC3 = read_dat_file_2(reaction)

        RC1 = np.array(RC1).flatten()
        RC2 = np.array(RC2).flatten()
        RC3 = np.array(RC3).flatten()
        x_label = "d((1-2)+(4-6))"
        y_label = "d(9-17)"
        x_data = RC1 + RC2
        y_data = RC3

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(projection="3d")
    sc = ax1.scatter(
        x_data, y_data, energies, c=energies, cmap="RdBu_r", alpha=0.8
    )
    cbar = plt.colorbar(sc, pad=0.1)
    cbar.set_label("$\Delta E_{rel}$ (kcal/mol)")

    ax1.set_xlabel(x_label + " ($\AA$)")
    ax1.set_ylabel(y_label + " ($\AA$)")
    ax1.set_zticklabels([])
    ax1.invert_yaxis()
    ax1.invert_xaxis()
    plt.savefig(f"3D_{name}_energy_plot.png", dpi=1600)
