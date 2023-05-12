#!/bin/bash

### Job name
#SBATCH --job-name=Poromechanics_PY

### runtime
#SBATCH -t 1-12:00:00

### partition (compute or gpu)
#SBATCH -p compute

### number of compute nodes
#SBATCH --nodes=1

### memory for the job
#SBATCH --mem=32GB

#### allow other jobs on the node
##SBATCH --exclusive

### output filename
#SBATCH --output=PoroMechSubsurface.out

### MPI ranks per node
#SBATCH --ntasks-per-node=1

### up to 4 threads per MPI rank
#SBATCH --cpus-per-task=8

# Load python module and numpy, scipy + matplotlib, specific to cluster
module load 2022r2 openmpi python py-numpy py-mpi4py py-pip py-scipy py-matplotlib

export OMP_NUM_THREADS=8

# true runs Heterogeneous (complex test case), false runs simple test case
if true
then
for T in {0..550..50} 
do
    python3 Poromechanics2D_HeterogenTest.py $T
done
else
for T in 0 100 200 300 400 500 600 
do
    python3 Poromechanics2D_PistonTest.py $T
done
fi
