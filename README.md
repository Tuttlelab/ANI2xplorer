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
|-- Data/   
|   |-- Reaction_X/              # Directory for each reaction studied   
|   |   |-- trajectories/        # Stores full trajectory XYZ files from simulations   
|   |   |-- full_sim_windows/    # Contains full simulation windows at different sampling resolutions   
|   |   |   |-- 1k/              # 1000 steps per window   
|   |   |   |   |-- window_X     # one directory for each window in the simulation   
|   |   |   |   |   |-- plumed.dat          # File containing inputs for bias potential, reaction coordinate, etc.   
|   |   |   |   |   |-- *COLVAR*.dat        # Output file from simulation, contains reaction coordinate vlaue, bias potential, etc.   
|   |   |   |   |   |-- molecule.xyz        # XYZ file containing starting structure of specific window   
|   |   |   |   |   |-- trajectory.traj     # ASE trajectory file written from simulation - from ase.io import read, read("trajectory.traj", index=":")   
|   |   |   |-- 10k/             # 10000 steps per window   
|   |   |-- energies/            # numpy files (.npy) contianing ANI2x single point energies of each frame in each trajectory   
|   
|-- Scripts/                     # Contains scripts for running simulations and analysis   
```
