#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 13:37:26 2025

@author: bwb16179
"""

import ase
from ase.io import read
import torchani
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def compute_energy(atoms):

    atoms.calc = ANI2x

    return atoms.get_potential_energy() * 23.06


def main(reaction, reaction_num, num_steps):

    with Pool(processes=10) as pool:
        Energies = list(
            tqdm(pool.imap(compute_energy, reaction), total=len(reaction))
        )

    Energies = np.array(Energies)
    np.save(
        f"../Data/Reaction_{reaction_num}/energies/Energies_{reaction_num}_{num_steps}.npy",
        Energies,
    )

    plt.plot(range(len(reaction)), Energies)
    plt.xlabel("Structure")
    plt.ylabel("Energy (kcal/mol)")
    plt.savefig("US_ANI_Energies_Reaction1B_1k.png", dpi=1600)


if __name__ == "__main__":

    ANI2x = torchani.models.ANI2x().to("cpu").ase()

    reaction_num = "R1A"
    num_steps = "1k"

    reaction = read(
        f"../Data/Reaction_{reaction_num}/trajectories/Reaction_{reaction_num}_{num_steps}.xyz",
        index=":",
    )

    main(reaction)
