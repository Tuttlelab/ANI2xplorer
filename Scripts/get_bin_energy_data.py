#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 16:01:05 2025

@author: bwb16179
"""

import tqdm
import pandas
import numpy as np
from ase.io import read
import plumed
import torchani

ANI2x = torchani.models.ANI2x().to("cpu").ase()


def get_bin_energies(window):
    """
    Return the mean potential energy and minimum energy conformer position
    for a full window trajectory
    """

    mean_window_energy = window.mean()  # mean_window_nergy

    position_min_window_conf = window.argmin()  # position_min_window_conf

    return mean_window_energy, position_min_window_conf


def read_dat_file(file):
    """
    Read in COLVAR.dat file as a pandas DataFrame
    returns:
        cv (positions of reaction coordinate for each frame)
        bias (bias potential for each frame)
    """

    df = plumed.read_as_pandas(file)
    cv = df.rc.to_list()
    bias = df.bias.to_list()

    return cv, bias


def print_window_avg_conf(path_to_highest_window_traj, target_directory):
    """

    Parameters
    ----------
    path_to_highest_window_traj : path to trajectory of specific window
    target_directory: path to directory the XYZ file is to be written in

    Returns
    -------
    printed XYZ file of conformer with avg position of each atom

    """

    highest_window_traj = read(path_to_highest_window_traj, index=":")

    average_positions = np.mean(
        np.array([x.get_positions() for x in highest_window_traj]), axis=0
    )

    with open(f"{target_directory}/avg_positions_conf.xyz", "w", encoding="utf-8") as f:
        atoms = highest_window_traj[0].get_chemical_symbols()
        positions = average_positions
        f.write(f"{len(atoms)}\n\n")
        for i, atom in enumerate(atoms):
            x, y, z = positions[i]
            f.write(f"{atom} {str(x)} {str(y)} {str(z)}\n")
    f.close()


def get_highest_window():
    """
    split array of energies for full simulation (all windows) by the number of windows
    calculate the mean energy for window and positon of minimum energy conformer in window
    """

    energies = np.load("../Data/Reaction_1B/energies/Energies_R1B_10K.npy")
    energies = np.split(energies, 30)

    mean_window_energies = []
    positions_of_min_window_conf = []

    for window in tqdm.tqdm(energies):

        assert len(window) == 10001

        mean_window_energy, position_min_window_conf = get_bin_energies(window)

        positions_of_min_window_conf.append(position_min_window_conf)
        mean_window_energies.append(mean_window_energy)

    positions_of_min_window_conf = np.array(
        positions_of_min_window_conf
    )  # positions_of_min_window_conformers
    mean_window_energies = np.array(mean_window_energies)  # all windows mean Es

    print(
        "Bin with highest mean E is:  Window ",
        str(int(mean_window_energies.argmax() + 1)),
    )

    highest_window = int(
        mean_window_energies.argmax() + 1
    )  # numpy indexing starts from 0, but windows start from 1

    lowest_energy_conf = positions_of_min_window_conf[mean_window_energies.argmax()]

    print(
        f"The lowest energy conformer of Window {highest_window} is at postion {lowest_energy_conf}"
    )


if __name__ == "__main__":

    get_highest_window()
