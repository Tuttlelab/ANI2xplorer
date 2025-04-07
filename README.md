# 🧪 ANI2xplorer  
_Extended Sampling & Umbrella Sampling for Reaction Barriers with the ANI2x Machine Learning Potential_  

![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)  
![Python](https://img.shields.io/badge/python-3.12%2B-blue)  
![Status](https://img.shields.io/badge/status-active-green)  

## 📌 Overview  
**ANI2xplorer** is a machine learning-driven workflow for exploring reaction pathways with ANI2x. This project employs ANI2x as a calculator through the atomic simulation enviroment (ASE) and the associated PLUMED calculator to drive umbrella sampling of organic reactions.   

## 🚀 Features  
- **ANI2x Potential**: Efficient and accurate quantum chemistry approximations  
- **Umbrella Sampling**: Sample along reaction coordinates by employing bias potentials 
- **Automated Workflow**: Easy setup for diverse organic reactions  
- **Visualization Tools**: Plot reaction pathways and barriers  

## 🛠 Installation  

Clone this repository:  
```bash
git clone https://github.com/your-username/ANI2xplorer.git
cd ANI2xplorer

```
## :file_folder: Repository Structure
```
|   
|-- Scripts/                       # Contains scripts for running simulations and analysis
|   |-- 3D_Energy_Traj_Plots.py    # Plots numpy arrays against reaction coordinates of simualtions
|   |-- WHAM.py                    # Perform WHAM analysis on the PLUMED umbrella sampling outputs
|   |-- Get_bin_RMSD_data.py       # Get the Root Mean Squared Deviation (RMSD) for all structure in a single simualtion window compared to the first frame of the window
|   |-- Get_bin_energy_data.py     # Get the highest energy window and the information regarding energies in each window
|   |-- Get_traj_ANI_Es.py         # Get the ANI-2x single point energy from a full simulation trajectory
|   |-- Run_umbrella_sampling.py   # Run the umbrella sampling simualtion for a single reaction
|   |-- Write_dat_file.py          # Write the PLUMED input files (.data files) for a each window of a simulation
|
```
