#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 11:59:34 2024

@author: bwb16179
"""
import os
import glob
from ase.io import read, Trajectory
from ase.md.langevin import Langevin
from ase.calculators.plumed import Plumed
from ase import units
from torchani.models import ANI2x
import torch
import natsort


def run_sampling(
    window_path, steps, calculator, timestep=0.5 * units.fs, stride=1
):

    windows = natsort.natsorted(
        [w for w in glob.glob(f"{window_path}/*") if os.path.isdir(w)]
    )

    cwd = os.path.abspath(".")

    for i, window in enumerate(windows, start=1):
        os.chdir(cwd)
        os.chdir(window)

        if os.path.exists("trajectory.traj"):
            continue

        molecule = glob.glob("*.xyz")[0]
        setup = open(f"plumed_{i}.dat", "r").read().splitlines()
        atoms = read(molecule)

        os.environ["PLUMED_KERNEL"] = (
            "/users/bwb16179/bin/lib/libplumedKernel.so"
        )

        atoms.calc = Plumed(
            calc=calculator,
            input=setup,
            timestep=timestep,
            atoms=atoms,
            kT=1.0,
        )

        dyn = Langevin(atoms, timestep, temperature_K=298.15, friction=0.01)

        traj = Trajectory("trajectory.traj", "w", atoms)
        dyn.attach(traj.write, interval=1)

        print(f"Running dynamics in: {os.path.abspath('.')}")
        dyn.run(steps)

        last_frame = read("trajectory.traj", index=-1)
        next_window_path = f"../window_{i + 1}/molecule.xyz"
        last_frame.write(next_window_path)

        os.chdir(cwd)


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ANI2x = ANI2x().to(device).ase()

    steps = 10000 # Number of steps per window
    run = ""
    reaction = "Reaction_1B"
    # path = f"Working_Folder/ANI/{reaction}/{run}"
    path = f"/{reaction}/{run}"

    run_sampling(path, steps, calculator=ANI2x)
