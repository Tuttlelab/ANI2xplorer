# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 18:46:33 2025

@author: Alex
"""
import tqdm, pandas
import numpy as np
from ase.io import read
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pathlib import Path
import sys, os, glob, pickle
import plumed


class WHAMAnalysis:
    def __init__(self, temp=300.0, tolerance=1e-8, max_iterations=10000):
        """
        Initialize WHAM analysis.

        Parameters:
        -----------
        temp : float
            Temperature in Kelvin
        tolerance : float
            Convergence criterion for WHAM iterations
        max_iterations : int
            Maximum number of iterations for WHAM convergence
        """
        self.kB = 8.617333262145e-5  # Boltzmann constant in eV/K
        self.temp = temp
        self.beta = 1 / (self.kB * self.temp)
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def load_trajectories(
        self, traj_pattern, atom1_idx=6, atom2_idx=9, atom3_idx=None
    ):
        """
        Load trajectories and calculate distances between specified atoms.

        Parameters:
        -----------
        traj_pattern : str
            Pattern for trajectory files (e.g., "traj_*.xyz")
        atom1_idx, atom2_idx : int
            Indices of atoms to calculate distance between (0-based)

        Returns:
        --------
        distances : list of numpy.ndarray
            Distances for each trajectory
        energies : list of numpy.ndarray
            Potential energies for each trajectory
        """
        trajectories = sorted(Path().glob(traj_pattern))
        print(trajectories)
        assert len(trajectories) > 0
        distances = []
        energies = []

        for traj_file in tqdm.tqdm(trajectories):
            frames = read(traj_file, ":")
            traj_distances = []
            traj_energies = []

            if atom3_idx == None:
                for frame in frames:
                    pos = frame.get_positions()
                    dist = np.linalg.norm(pos[atom1_idx] - pos[atom2_idx])
                    traj_distances.append(dist)
                    traj_energies.append(frame.get_potential_energy())
            else:
                for frame in frames:
                    pos = frame.get_positions()
                    dAB = np.linalg.norm(pos[atom1_idx] - pos[atom2_idx])
                    dAC = np.linalg.norm(pos[atom1_idx] - pos[atom3_idx])
                    rc = dAB - dAC
                    traj_distances.append(rc)
                    traj_energies.append(frame.get_potential_energy())

            distances.append(np.array(traj_distances))
            energies.append(np.array(traj_energies))

        return distances, energies

    def read_dat_file(self, file_pattern):
        trajectories = sorted(glob.glob(file_pattern))
        cv = []
        bias = []
        for dat_file in tqdm.tqdm(trajectories):
            df = plumed.read_as_pandas(dat_file)
            cv.append(df.rc.to_list())
            bias.append(df.bias.to_list())

        return cv, bias

    def setup_umbrellas(
        self,
        distances,
        kappa,
        r0_range=(1.34, 3.34, 4.44, 1.789),
        n_windows=30,
    ):
        """
        Setup umbrella sampling windows.
        """
        # =============================================================================
        #         if not rc:
        #             r0_values = np.linspace(r0_range[0], r0_range[1], n_windows)
        #             self.r0_values = r0_values  # Store for later use
        # =============================================================================
        R1 = np.linspace(r0_range[0], r0_range[1], n_windows)
        R2 = np.linspace(r0_range[2], r0_range[3], n_windows)
        self.r0_values = R1 + R2

        bias_energies = []
        for dist in distances:
            window_biases = []
            for r0 in self.r0_values:
                # Calculate harmonic bias potential: V = 0.5 * k * (r - r0)²
                bias = 0.5 * kappa * (dist - r0) ** 2
                window_biases.append(bias)
            bias_energies.append(np.array(window_biases))

        return bias_energies

    def run_wham(self, distances, bias_energies, kappa, n_bins=100):
        """
        Run WHAM iterations to calculate PMF with improved debugging.
        """
        # Setup histograms
        all_distances = np.concatenate(distances)
        hist_range = (np.min(all_distances), np.max(all_distances))
        print(f"Distance range: {hist_range[0]:.3f} to {hist_range[1]:.3f} Å")

        bin_edges = np.linspace(hist_range[0], hist_range[1], n_bins + 1)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        np.save("R1A_bin_edges.npy", bin_edges)
        np.save("R1A_bin_centres.npy", bin_centers)
        # sys.exit()

        # Calculate and plot histograms for each window
        N_k = []  # Number of samples in each window
        n_ij = []  # Histogram counts

        # plt.figure(figsize=(12, 6))
        for i, dist in enumerate(distances):
            hist, _ = np.histogram(dist, bins=bin_edges)
            N_k.append(len(dist))
            n_ij.append(hist)

            # Plot histogram for this window
            plt.plot(
                bin_centers,
                hist,
                label=f"Window {i+1} (r0={self.r0_values[i]:.2f}Å)",
            )

        plt.xlabel("Reaction Coordinate ($\mathrm{\AA}$)")
        plt.ylabel("Configurations per Window")
        # plt.title('Histogram of Counts for Each Window')
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        # plt.savefig(savefig)
        plt.gca().invert_xaxis()
        plt.show()

        print("\nSampling statistics:")
        print("Window  Center(Å)  Samples  Mean(Å)   Std(Å)")
        print("-" * 50)
        for i, (n, dist) in enumerate(zip(N_k, distances)):
            print(
                f"{i+1:^6d}  {self.r0_values[i]:^9.3f}  {n:^7d}  {np.mean(dist):^8.3f}  {np.std(dist):^7.3f}"
            )

        histogram_counts = n_ij

        N_k = np.array(N_k)
        n_ij = np.array(n_ij)  # Shape: (n_windows, n_bins)

        # Add small constant to prevent log(0)
        eps = 1e-8
        n_ij = n_ij + eps

        # Initialize free energies
        F_k = np.zeros(len(distances))

        # Pre-calculate bias energies at bin centers
        bias_at_bins = np.zeros((len(distances), len(bin_centers)))
        for k in range(len(distances)):
            r0 = self.r0_values[k]
            bias_at_bins[k] = 0.5 * kappa * (bin_centers - r0) ** 2

        print("\nStarting WHAM iterations...")

        # WHAM iterations
        for iteration in range(self.max_iterations):
            F_k_old = F_k.copy()

            # Calculate unbiased probabilities
            numerator = n_ij.sum(axis=0)  # Shape: (n_bins,)
            denominator = np.zeros(len(bin_centers))

            for k in range(len(distances)):
                # Calculate exp(-beta * (V_bias - F_k)) for each bin
                exp_term = np.exp(-self.beta * (bias_at_bins[k] - F_k[k]))
                denominator += N_k[k] * exp_term

            # Add eps to prevent division by zero
            P_i = np.where(denominator > eps, numerator / denominator, eps)

            # Update free energies
            for k in range(len(distances)):
                exp_term = np.exp(-self.beta * bias_at_bins[k])
                sum_term = np.sum(P_i * exp_term)
                if sum_term > eps:
                    F_k[k] = -1 / self.beta * np.log(sum_term)
                else:
                    F_k[k] = F_k_old[k]  # Keep old value if numerical issues

            # Check convergence
            max_diff = np.max(np.abs(F_k - F_k_old))
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: max ΔF = {max_diff:.6f}")

            if max_diff < self.tolerance:
                print(f"\nWHAM converged after {iteration + 1} iterations")
                break

        if iteration == self.max_iterations - 1:
            print("\nWarning: WHAM did not converge!")

        print("P_i:", P_i)
        # Calculate PMF with numerical stability measures
        valid_idx = (
            P_i > eps
        )  # Only calculate PMF where probability is non-negligible
        pmf = np.full_like(P_i, np.inf)  # Initialize with inf

        # Calculate PMF only for valid points
        pmf[valid_idx] = -1 / self.beta * np.log(P_i[valid_idx])

        if np.any(valid_idx):
            min_valid_pmf = np.min(pmf[valid_idx])
            pmf[valid_idx] -= min_valid_pmf

            print(f"\nPMF statistics:")
            print(f"Valid points: {np.sum(valid_idx)} out of {len(pmf)}")
            print(
                f"PMF range: {np.min(pmf[valid_idx]):.3f} to {np.max(pmf[valid_idx]):.3f} eV"
            )
        else:
            print("\nError: No valid PMF points found!")
            print("Try adjusting these parameters:")
            print("1. Decrease the number of bins (current: {n_bins})")
            print("2. Check window spacing and overlap")
            print("3. Increase sampling in each window")
            print("4. Adjust the temperature or force constant")

        return bin_centers, pmf, histogram_counts

    def plot_pmf(self, distances, pmf, output_file="pmf.png"):
        """
        Plot the PMF curve.

        Parameters:
        -----------
        distances : numpy.ndarray
            Distance values
        pmf : numpy.ndarray
            PMF values
        output_file : str
            Output file name for the plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(distances, pmf, "b-", linewidth=2)
        plt.xlabel("Reaction Coordinate")
        plt.ylabel("PMF (eV)")
        plt.title("Potential of Mean Force")
        plt.grid(True)
        # plt.savefig(output_file)
        plt.close()

    def save_structures_to_xyz(
        self, reaction_coords, relevant_coords, traj_pattern, output_file
    ):

        rxn_coords = np.concatenate(reaction_coords)
        rxn_coords = rxn_coords.tolist()

        f = [float(round(x, 2)) for x in relevant_coords.tolist()]

        relevant_indices = [
            i for i, rc in enumerate(rxn_coords) if round(rc, 2) in f
        ]
        assert relevant_indices, "No matching reaction coordinates found!"

        trajectories = sorted(Path().glob(traj_pattern))
        assert len(trajectories) > 0, "No trajectory files found!"

        trajs = []
        for file in trajectories:
            trajs.append(read(file, ":"))

        trajs = [x for x in trajs]

        # Write the relevant structures to the output XYZ file
        for i, frame in enumerate(trajs):
            # Check if the current frame index is in the relevant indices
            if i in relevant_indices:
                frame.write("Results/PMF_structures.xyz", append=True)


import natsort


def print_full_traj(Run, Reaction):
    # convert traj to extxyz, we dont really need this though
    path = f"Working_Folder/ANI/{Reaction}/{Run}/*/*.traj"
    trajs = glob.glob(path)
    trajs = natsort.natsorted(trajs)
    if "Backward" in path:
        xyz = f"Trajs/{Reaction}_Backward_Run_{Run}.xyz"
    else:
        xyz = f"Trajs/{Reaction}_Run_{Run}.xyz"
    for traj in trajs:
        # xyz = traj.replace(".traj", ".xyz")
        # if os.path.exists(xyz):
        # continue
        print(traj)
        i = 0
        for frame in read(traj, ":"):
            frame.write(xyz, append=True, format="xyz")
            i += 1


# Initialize WHAM analysis
wham = WHAMAnalysis(temp=298.15)

Run = "10k"
kappa = 50.0
Reaction = "Reaction_1A"

distances, energies = wham.read_dat_file(
    f"../Data/{Reaction}/full_sim_windows/{Run}/*/COLVAR*.dat"
)

# Setup umbrella sampling windows
bias_energies = wham.setup_umbrellas(
    distances,
    kappa=kappa,
    r0_range=(2.88, 1.34, 4.4448, 1.79),  # Reaction 1A
    # r0_range=(3.54, 1.36, 2.59, 1.34), # Reaction 1B
    # r0_range=(1.93,0.96,1.60,0.96,3.32,2.05), # Reaction 2
    n_windows=30,
)

# Run WHAM analysis
bin_centers, pmf, histogram_counts = wham.run_wham(
    distances, bias_energies, kappa, n_bins=100
)

# sys.exit()

# Plot results
wham.plot_pmf(bin_centers, pmf)

data = pandas.DataFrame()
data["bin_centers"] = bin_centers
data["pmf"] = pmf
print(data)

valid_mask = ~np.isnan(pmf) & ~np.isinf(pmf)
data = data.iloc[valid_mask]

data = data.dropna()

fig = plt.figure()
gs = fig.add_gridspec(2, hspace=0)
axs = gs.subplots(sharex=True)
for x in histogram_counts:
    axs[0].plot(bin_centers, x)
axs[0].set_ylabel("Counts per Window")
axs[1].plot(data["bin_centers"], data["pmf"] * 23.06)
axs[1].set_xlabel("Reaction Coordinate ($\mathrm{\AA}$)")
axs[1].set_ylabel("F (kcal/mol)")
plt.tight_layout()
plt.gca().invert_xaxis()

plt.plot(data["bin_centers"], data["pmf"] * 23.06)
plt.xlabel("Reaction Coordinate ($\mathrm{\AA}$)")
plt.ylabel("Helmholtz Free Energy (kcal/mol)")
plt.gca().invert_xaxis()

distances_all = np.array(distances).flatten()
suspect_geoms = np.where((distances_all >= 3.961) & (distances_all <= 3.9605))[
    0
]
