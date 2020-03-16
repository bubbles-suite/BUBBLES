#!/bin/bash

# The -DSC flag switches the Soft-Corona interactions, remove it from the command to compile without these interactions

nvcc -DSC -D_FORCE_INLINES -arch=sm_30 -use_fast_math -O2 -lcurand --ptxas-options -v mdgpu_np.cu -o mdgpu_np

cp mdgpu_np ..
