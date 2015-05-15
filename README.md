.:BUBBLES:.
================

BUBBLES ("BUBBLES is a User-friendly Bundle for Bio-nano Large-scale Efficient Simulations") is a set of tools designed to simulate the interactions and kinetics of Nanoparticles in biological enviornments of aqueous solutions containing proteins.

# Features

+ Supporting CUDA-capable devices (nVidia GPU's)
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
+ Higher detail protein models
+ Input custom effective potentials (tabulated)

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

## Using VMD to visualize snapshots and trajectories

The program creates trajectories with periodic snapshots of the simulation in the plaintext format **\*.xyz**
which can be visualized on-the-fly with **VMD**. This software is a powerful visualization tool freely available for download at <a href="http://www.ks.uiuc.edu/Research/vmd/">http://www.ks.uiuc.edu/Research/vmd/</a>

Enter the directory **Utils**, and execute VMD:
```
$ cd Utils/
$ vmd
```

In the VMD Main window go to `>File>Load Viasualization State...`. Load one of the vmd scripts inside the Utils directory, these scripts are pre-configured to visualize a system of 3 proteins:
+ viewstate.vmd: shows a trajectory in linear time, which is more suitable for visualizing long-time dynamics
+ viewstateLog.vmd: shows a trajectory in logarithmic time, which is more suitable for visualizing the dynamics at early time-steps


# Examples

## Running a simulation with one kind of proteins

Follow the instructions in the wiki section https://github.com/bubbles-suite/BUBBLES/wiki/Examples#51-running-a-simulation-with-one-kind-of-proteins

## Running a simulation with multiple kinds of proteins

Follow the instructions in the wiki section https://github.com/bubbles-suite/BUBBLES/wiki/Examples#51-running-a-simulation-with-one-kind-of-proteins

# Documentation

The documentation for this project is available as a wiki inside this repository at https://github.com/bubbles-suite/BUBBLES/wiki/


