#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 11:59:34 2024

@author: bwb16179
"""

from ase.io import read, Trajectory
from torchani.models import ANI2x
import torch

from ase.md.langevin import Langevin

from ase.calculators.plumed import Plumed
from ase import units

import matplotlib.pyplot as plt
import os, sys, json, pickle
import glob
import natsort


def run_sampling(window_path, steps, calculator, timestep=0.5 * units.fs):
    """
    window_path: Path to working folder, includes reaction and run
    steps: The number of steps per window of the simulation
    calculator: ASE calculator instance for use of MLP
    timestep: timestep of MD simulation in ase units (step*units.fs for fs)
    """

    windows = natsort.natsorted(
        [w for w in glob.glob(f"{window_path}/*") if os.path.isdir(w)]
    )

    cwd = os.path.abspath(".")

    # enumerate through windows of simulation performing
    # each sim in that windows directory

    for i, window in enumerate(windows, start=1):
        os.chdir(cwd)
        os.chdir(window)

        if os.path.exists("trajectory.traj"):  # for restarts, don't run again
            continue

        molecule = glob.glob("*.xyz")[0]
        setup = open(f"plumed_{i}.dat", "r").read().splitlines()
        atoms = read(molecule)

        # PLUMED must be pre-compiled for running window simulation
        # This system variable must be set on a per user basis

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

        # use Langevin thermostat for simulation

        dyn = Langevin(atoms, timestep, temperature_K=298.15, friction=0.01)

        # attach ASE trajectory object

        traj = Trajectory("trajectory.traj", "w", atoms)
        dyn.attach(traj.write, interval=1)

        print(f"Running dynamics in: {os.path.abspath('.')}")
        dyn.run(steps)

        # every following window continues where the last finished

        last_frame = read("trajectory.traj", index=-1)
        next_window_path = f"../window_{i + 1}/molecule.xyz"
        last_frame.write(next_window_path)

        os.chdir(cwd)


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ANI2x = ANI2x().to(device).ase()

    steps = 1000  # number of steps per window
    run = 6  # which run of the simulation this si
    reaction = "Reaction_1A"  # Reaction 1A, 1B or 2
    path = f"path/to/wrking/folder/{reaction}/{run}"

    run_sampling(path, steps, calculator=ANI2x)
