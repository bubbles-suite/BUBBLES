.:BioNano-cgMD:.
================

BioNano-cgMD (Biological Nanoscale coarse-grain Molecular Dynamics) is a set of tools designed to simulate the interactions and kinetics of Nanoparticles in biological enviornments of aqueous solutions containing proteins.

# Features

+ Supporting CUDA-capable devices (nVidia)
+ Implicit solvent Molecular Dynamics
+ Berendsen thermostat for equilibration
+ Langevin thermostat for dynamics
+ DLVO interaction potentials for colloidal particles
+ Tabulated potentials involving 2-body and 3-body interactions
+ Concentration control algorithms
+ Up to 1 Nanoparticle in simulation box
+ Up to 3 kinds of different proteins

## TODO

+ Arbitrary number of Nanoparticles
+ Arbitrary number of kinds of proteins
+ Patchy Nanoparticles and proteins
+ Proteins defined as groups of beads

# Installation

First check that the system is running a CUDA-enabled device: https://developer.nvidia.com/cuda-gpus.
Next, install the nVidia propietary drivers and the CUDA toolkit:
http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux/

Download the project package and uncompress it.
Inside the project main directory, run the following in a terminal:
```
$ cd src/
$ ./compile.sh
$ cd .. 
```
If the compilation ends succesfully, a binary with the name **mdgpu_np** should have been created

# Quick guide

## Creating the initial configuration of a simulation

Edit the configuration script **genConfig.py** and run it:
``` 
$ ./genConfig.py
```
or alternatively
```
$ python genConfig.py
```

This will create a file called **init.dat** inside the directory **config** which is the input of the simulation.

## Running the simulation

Edit the simulation setup file called **setup.dat**
Run the executable followed by the configuration data file
```
$ ./mdgpu_np setup.dat
```
Alternatively, run the simulation in background, unattached from the user:
```
$ nohup ./mdgpu_np setup.dat &
```

# Examples

## Running a simulation with one kind of proteins

## Running a simulation with multiple kinds of proteins

# Documentation

The documentation for this project is not ready yet and will be available in a future


