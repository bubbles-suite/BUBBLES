.:BioNano-cgMD:.
================

BioNano-cgMD (Biological Nanoscale coarse-grain Molecular Dynamics) is a set of tools designed to simulate the interactions and kinetics of Nanoparticles in biological enviornments such as aqueous solutions containing proteins.



# Features

+ Supporting CUDA-capable devices (nVidia)
+ Molecular Dynamics
+ Berendsen thermostat for equilibration
+ Langevin thermostat for dynamics
+ Tabulated Potentials involving 2-body and 3-body interactions
+ Concentration control algorithms
+ Up to 1 Nanoparticle in simulation box
+ Up to 3 kinds of different proteins

## TODO

+ Arbitrary number of Nanoparticles
+ Arbitrary number of kinds of proteins
+ Patchy Nanoparticles and proteins
+ Proteins defined as groups of beads


# Documentation

The documentation for this project is not ready yet and will be available in a future

# Installation

Just download the project package and uncompress it.
Inside the project root directory:
```
cd src/
./compile.sh
cd ..
```
If the compilation is succesful, a binary with the name *mdgpu_np* should have been created

# Running the code

Run the executable followed by the configuration data file
```
./mdgpu_np setup.dat
```
Running the simulation in background, unattached from the user:
```
nohup ./mdgpu_np setup.dat &
```


# Quick guide

# Examples
