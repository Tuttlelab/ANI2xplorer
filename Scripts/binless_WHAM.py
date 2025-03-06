#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 16:31:41 2025

@author: bwb16179
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d
import glob
import natsort
import tqdm
import plumed
import numpy as np


class BinlessWHAM:
    def __init__(
        self,
        distances,
        bias_energies,
        r0_values,
        kappa,
        temperature,
        max_iterations=10000,
        tolerance=1e-8,
    ):
        self.distances = distances
        self.bias_energies = bias_energies
        self.r0_values = r0_values
        self.kappa = kappa
        self.beta = 1 / (8.617333262145e-5 * temperature)  # 1/k_B T in eV^-1
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.n_windows = 30

    def run_wham(self, n_points=200):
        # Sample reaction coordinate points evenly over the observed range
        all_distances = np.concatenate(self.distances)
        x_min, x_max = np.min(all_distances), np.max(all_distances)
        reaction_coords = np.linspace(x_min, x_max, n_points)

        # Estimate probability distributions using KDE with adaptive bandwidth
        kde_list = []
        n_samples = [len(d) for d in self.distances]
        bw_factors = np.clip(
            1 / np.sqrt(n_samples), 0.1, 0.5
        )  # Avoid overfitting
        for i, dists in enumerate(self.distances):
            kde = gaussian_kde(dists, bw_method=bw_factors[i])
            kde_list.append(kde)

        kde_values = np.array(
            [kde(reaction_coords) for kde in kde_list]
        )  # (n_windows, n_points)

        # Bias energy evaluation at selected reaction coordinates
        bias_at_x = np.array(
            [
                0.5
                * (
                    bias_k[0] * (reaction_coords - bias_k[1]) ** 2
                )  # Harmonic bias U = 0.5*k*(x-x0)^2
                for bias_k in self.bias_energies
            ]
        )  # Shape: (n_windows, n_points)

        # Initialize free energy estimates
        F_k = np.zeros(self.n_windows)

        print("\nStarting WHAM iterations...")
        for iteration in range(self.max_iterations):
            F_k_old = F_k.copy()

            # Compute unbiased probabilities
            numerator = kde_values.sum(axis=0)  # Shape: (n_points,)
            exp_weights = np.exp(
                -self.beta * (bias_at_x - F_k[:, np.newaxis])
            )  # Shape: (n_windows, n_points)
            denominator = np.sum(
                kde_values * exp_weights, axis=0
            )  # Shape: (n_points,)

            # Enforce minimum probability to avoid log(0)
            P_i = np.maximum(numerator / np.maximum(denominator, 1e-12), 1e-12)

            # Update free energies with adaptive weighting
            adaptive_weight = 1 / (1 + np.exp(self.beta * (F_k - F_k_old)))
            F_k = (1 - adaptive_weight) * F_k_old + adaptive_weight * (
                -1 / self.beta * np.log(P_i.sum(axis=0))
            )

            # Check for convergence
            max_diff = np.max(np.abs(F_k - F_k_old))
            if iteration % 1000 == 0:
                print(f"Iteration {iteration}: max ΔF = {max_diff:.6f}")

            if max_diff < self.tolerance:
                print(f"\nWHAM converged after {iteration + 1} iterations")
                break

        if iteration == self.max_iterations - 1:
            print("\nWarning: WHAM did not fully converge!")

        # Compute PMF
        pmf = -1 / self.beta * np.log(P_i)
        pmf -= np.min(pmf)  # Normalize to min(PMF) = 0

        # Apply Gaussian smoothing to stabilize PMF near transition state
        pmf_smoothed = gaussian_filter1d(pmf, sigma=2)

        print(f"\nFinal PMF range: {np.min(pmf):.3f} to {np.max(pmf):.3f} eV")

        plt.figure(figsize=(10, 6))
        plt.plot(reaction_coords, pmf_smoothed * 23.06, color="blue")
        plt.xlabel("Reaction Coordinate ($\mathrm{\AA}$)")
        plt.ylabel("Free Energy (kcal/mol)")
        plt.title("Optimized Binless WHAM PMF")
        plt.tight_layout()
        plt.gca().invert_xaxis()
        plt.show()

        return reaction_coords, pmf_smoothed


def read_dat_file(file_pattern):
    trajectories = sorted(glob.glob(file_pattern))
    cv = []
    bias = []
    for dat_file in tqdm.tqdm(trajectories):
        df = plumed.read_as_pandas(dat_file)
        cv.append(df.rc.to_list())
        bias.append(df.bias.to_list())

    return cv, bias


# Example Usage:
if __name__ == "__main__":

    Run = "10k"
    kappa = 50.0
    Reaction = "Reaction_1A"
    n_windows = 30
    temperature = 298.15

    distances, bias_energies = read_dat_file(
        f"../Data/{Reaction}/full_sim_windows/{Run}/*/COLVAR*.dat"
    )

    r0_range = (2.88, 1.34, 4.4448, 1.79)  # Reaction 1A
    # r0_range=(3.54, 1.36, 2.59, 1.34), # Reaction 1B
    # r0_range=(1.93,0.96,1.60,0.96,3.32,2.05), # Reaction 2

    R1 = np.linspace(r0_range[0], r0_range[1], n_windows)
    R2 = np.linspace(r0_range[2], r0_range[3], n_windows)
    r0_values = R1 + R2

    wham = BinlessWHAM(distances, bias_energies, r0_values, kappa, temperature)
    rc_values, pmf = wham.run_wham()
