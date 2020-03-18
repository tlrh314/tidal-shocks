#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./tjob.out.%j
#SBATCH -e ./tjob.err.%j
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J submit.sh
# Queue (Partition):
#SBATCH --partition=p.24h
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#
#SBATCH --mail-type=all
#SBATCH --mail-user=timoh@rzg.mpg.de
#
# Wall clock limit:
#SBATCH --time=08:00:00

# MCMC fit
# python src/tlrh_profiles.py -gc "NGC 104" -Nw 32 -Ns 50000 -Nb 1000 

# Test stability of GC in isolation
nice -n 19 python src/galpy_amuse.py -gc "NGC 104" -T 10 -dt 5 --isolation
