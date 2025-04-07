#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 16:31:41 2025

@author: bwb16179
"""
import glob, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import natsort
import tqdm
import plumed
from ase.io import read


class BinlessWHAM:
    def __init__(
        self,
        distances,
        bias_energies,
        r0_values,
        kappa,
        temperature,
        n_windows,
        max_iterations=10000,
        tolerance=1e-8,
    ):
        self.distances = distances
        self.bias_energies = bias_energies
        self.r0_values = r0_values
        self.kappa = kappa
        self.beta = 1 / (
            1.987204259e-3 * temperature  # kcal/mol
        )  # 1/k_B T in eV^-1 8.617333262145e-5
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.n_windows = n_windows

    def run_wham(self, name, n_points=100):
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

        print(f"\nFinal PMF range: {np.min(pmf):.3f} to {np.max(pmf):.3f} eV")

        return reaction_coords, pmf
    
    def get_barrier(self, reaction, pmf_rc_values):
        reactant_minima_ranges = {"Reaction_1A": (4.5, 7.0),
                                  "Reaction_1B": (4.5, 6.0),
                                  "Reaction_2": ( 5.5, 6.5)}
        
        if not reaction in ["Reaction_1A", "Reaction_1B", "Reaction_2"]:
            print("Reaction name not valid")
            return 0
        
        lower, upper = reactant_minima_ranges[reaction]
        minima_energies = [pmf for rc, pmf in pmf_rc_values if lower < rc < upper]

        reaction_minima_energy = np.mean(minima_energies)

        highest = np.max([pmf for rc, pmf in pmf_rc_values[10:]]) #  assuming 100 value pmf, ignore initial relaxation of the system

        print(f"Energy barrier for {reaction} is: ", highest - reaction_minima_energy)



def read_dat_file(file_pattern):
    trajectories = natsort.natsorted(glob.glob(file_pattern))
    cv = []
    bias = []
    for dat_file in tqdm.tqdm(trajectories):
        df = plumed.read_as_pandas(dat_file)
        cv.append(df.rc.to_list())
        bias.append(df.bias.to_list())

    return cv, bias


def update_colvar_files(folder):
    pattern = {
        "#! FIELDS time dR1 dR2 dR3 dR4 rc bb.bias",
        "#! FIELDS time dAB dAC rc bb.bias",
        "#! FIELDS time dAB dAC dAD rc bb.bias",
        "#! FIELDS time dAB dAC rc bias",
    }
    replacement_1 = "#! FIELDS time dAB dAC rc bias\n"
    replacement_2 = "#! FIELDS time dAB dAC dAD rc bias\n"

    for root, _, files in os.walk(folder):
        for filename in files:
            if filename.startswith("COLVAR_d_") and filename.endswith(".dat"):
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, "r+") as file:
                        lines = file.readlines()
                        if lines and lines[0].strip() in pattern:
                            # Check if "dAD" appears in any line of the file - needed for Reaction 2
                            if any("dAD" in line for line in lines):
                                lines[0] = replacement_2
                            else:
                                lines[0] = replacement_1
                            file.seek(0)
                            file.writelines(lines)
                            file.truncate()
                            print(f"Updated file: {filepath}")
                        else:
                            print(
                                f"Skipped file: {filepath} (no matching first line)"
                            )
                except Exception as e:
                    print(f"Error processing file {filepath}: {e}")


def print_full_traj(Run, Reaction):
    # convert traj to extxyz, we dont really need this though
    path = f"../Working_Folder/{Reaction}/{Run}/*/*.traj"
    trajs = glob.glob(path)
    trajs = natsort.natsorted(trajs)
    if "Backward" in path:
        xyz = f"../Trajs/{Reaction}_Backward_Run_{Run}.xyz"
    else:
        xyz = f"../Trajs/{Reaction}_Run_{Run}.xyz"
    for traj in trajs:
        print(traj)
        i = 0
        for frame in read(traj, ":"):
            frame.write(xyz, append=True, format="xyz")
            i += 1


def run_histograms(
    distances,
    bias_energies,
    kappa,
    n_bins=50,
    savefig="histograms.png",
):
    """
    Run WHAM iterations to calculate PMF with improved debugging.
    """
    # Setup histograms
    all_distances = np.concatenate(distances)
    hist_range = (np.min(all_distances), np.max(all_distances))
    print(f"Distance range: {hist_range[0]:.3f} to {hist_range[1]:.3f} Å")

    bin_edges = np.linspace(hist_range[0], hist_range[1], n_bins + 1)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    np.save("R1B_bin_edges.npy", bin_edges)
    np.save("R1B_bin_centres.npy", bin_centers)

    # Calculate and plot histograms for each window
    N_k = []  # Number of samples in each window
    n_ij = []  # Histogram counts

    # plt.figure(figsize=(12, 6))
    for i, dist in enumerate(distances):
        hist, _ = np.histogram(dist, bins=bin_edges)
        N_k.append(len(dist))
        n_ij.append(hist)

    return N_k, n_ij, bin_centers


def plot_results(
    bin_centers, histograms, reaction_coords, pmf_values, reaction_name
):

    fig = plt.figure()
    gs = fig.add_gridspec(2, hspace=0)
    axs = gs.subplots(sharex=True)
    fig.suptitle(name)
    for x in histograms:
        axs[0].plot(bin_centers, x)
    axs[0].set_ylabel("Counts per Window")
    axs[0].set_yscale("log")
    axs[1].plot(reaction_coords, pmf_values * 23.0609)
    axs[1].set_xlabel("Reaction Coordinate ($\mathrm{\AA}$)")
    axs[1].set_ylabel("$\Delta E_{rel}$ (kcal/mol)")
    plt.tight_layout()
    plt.gca().invert_xaxis()
    if not os.path.exists(f"/{reaction_name}.png"):
        plt.savefig(f"/{reaction_name}.png", dpi=600)
    plt.show()


# Example Usage:
if __name__ == "__main__":

    Run = ""
    kappa = 250
    Reaction = "Reaction_1B"
    n_windows = 50
    temperature = 298.15

    update_colvar_files(f"../Working_Folder/{Reaction}/{Run}/") # changes bb.bias to bias so pandas cna read the file

    distances, bias_energies = read_dat_file(
        "path/to/COLVAR*.dat"
    )

    if Reaction == "Reaction_2":
        r0_range = (1.93, 0.96, 1.60, 0.96, 3.32, 2.05)
        R1_dists = np.linspace(1.93, 0.9, num=n_windows)
        R2_dists = np.linspace(1.6, 0.9, num=n_windows)
        R3_dists = np.linspace(3.32, 1.8, num=n_windows)
        r0_values = R1_dists + R2_dists + R3_dists
    elif Reaction == "Reaction_1A":
        r0_range = (2.88, 1.00, 4.4448, 1.50)
        R1 = np.linspace(r0_range[0], r0_range[1], n_windows)
        R2 = np.linspace(r0_range[2], r0_range[3], n_windows)
        r0_values = R1 + R2
    elif Reaction == "Reaction_1B":
        r0_range = (3.54, 1.00, 2.59, 1.00)
        R1 = np.linspace(r0_range[0], r0_range[1], n_windows)
        R2 = np.linspace(r0_range[2], r0_range[3], n_windows)
        r0_values = R1 + R2

    num_samples_per_window, histogram_counts, bin_centers = run_histograms(
        distances, bias_energies, kappa
    )

    wham = BinlessWHAM(
        distances,
        bias_energies,
        r0_values,
        kappa,
        temperature,
        n_windows=n_windows,
    )
    rc_values, pmf = wham.run_wham(f"{Reaction}_{Run}")

    name = Reaction + "_" + Run + "_" + str(kappa)

    plot_results(
        bin_centers, histogram_counts, rc_values, pmf, name
    )

    if not os.path.exists(f"{Reaction}_Run_{Run}.xyz"):
        print_full_traj(Run, Reaction)
