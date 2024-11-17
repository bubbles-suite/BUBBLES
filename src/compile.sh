#!/bin/bash

# The -DSC flag switches the Soft-Corona interactions ON, remove it from the command to compile without these interactions (to set them OFF)

nvcc -DSC -D_FORCE_INLINES -use_fast_math -O2 -lcurand --ptxas-options -v mdgpu_np.cu -o mdgpu_np

cp mdgpu_np ..
