#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:48:54 2025

@author: bwb16179
"""

import tqdm
import numpy as np
from ase.io import read
from ase.build import minimize_rotation_and_translation as mrat
import matplotlib.pyplot as plt
import glob
import plumed
from ase import Atoms
from natsort import natsorted


def calc_rmsd(c1, c2):
    """
    Parameters
    ----------
    c1 : np.array
      coordinates of target mol

    c2 : np.array
      coordinates of molecule to be fit

    Returns
    -------
    rmsd : float
        root mean square distance of c2 from c1

    """
    rmsd = 0.0
    c_trans, U, ref_trans = fit_rms(c1, c2)
    new_c2 = np.dot(c2 - c_trans, U) + ref_trans
    rmsd = np.sqrt(np.average(np.sum((c1 - new_c2) ** 2, axis=1)))
    return rmsd


def fit_rms(ref_c, c):
    # move geometric center to the origin
    ref_trans = np.average(ref_c, axis=0)
    ref_c = ref_c - ref_trans
    c_trans = np.average(c, axis=0)
    c = c - c_trans

    # covariance matrix
    C = np.dot(c.T, ref_c)

    # Singular Value Decomposition
    (r1, s, r2) = np.linalg.svd(C)

    # compute sign (remove mirroring)
    if np.linalg.det(C) < 0:
        r2[2, :] *= -1.0
    U = np.dot(r1, r2)
    return (c_trans, U, ref_trans)


def get_all_rmsd(target_mol: Atoms, trajectory_mols: list):
    """

    Parameters
    ----------
    target_mol : ase.Atoms
        reference molecule.
    trajectory_mols : list
        list of ase.Atoms. molecules to be minimized and fit to target_mol

    Returns
    -------
    rmsds : list
        list of rmsds for each mol in trajectory_mols, list of floats.

    """

    target_mol_coords = target_mol.get_positions()

    rmsds = []

    for asemol in trajectory_mols:
        mrat(target_mol, asemol)
        rmsds.append(calc_rmsd(target_mol_coords, asemol.get_positions()))

    return rmsds


def read_dat_file(file):
    """


    Parameters
    ----------
    file : str
        path to COLVAR.dat file.

    Returns
    -------
    cv : list
        collective variables (value of reaction coordinate) at each frame.
    bias : list
        bias potentials for each frame of trajectory.

    """

    df = plumed.read_as_pandas(file)
    cv = df.rc.to_list()
    bias = df.bias.to_list()

    return cv, bias


def get_bin_data(dat_file, traj_file, target_mol):
    """


    Parameters
    ----------
    dat_file : str
        path to COLVAR.dat file.
    traj_file : str
        path to ase .traj file.
    target_mol : Atoms
        ASE molecule to reference RMSD calculation against.

    Returns
    -------
    list
        zipped list of the reaction coord for each frame with the corresponding
        rmsd value.

    """

    trajs = read(traj_file, index=":")

    assert len(trajs) == 10001

    rc_list, bias = read_dat_file(dat_file)

    bin_rmsds = get_all_rmsd(target_mol, trajs)

    return list(zip(rc_list, bin_rmsds))


def plot_rmsds(rmsds):
    """


    Parameters
    ----------
    rmsds : list
        list of rmsd values for trajectory.

    """

    plt.plot(rmsds)
    plt.xlabel("Reaction Coordinate ($\mathrm{\AA}$)")
    plt.ylabel("RMSD ($\mathrm{\AA}$)")
    plt.title("RMSD vs DFT TS")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.gca().invert_xaxis()
    plt.show()
