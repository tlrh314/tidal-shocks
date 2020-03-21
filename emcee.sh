#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./tjob.%j.out
#SBATCH -e ./tjob.%j.err
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J emcee.sh
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
#SBATCH --time=24:00:00

gc_name=$*
echo "Running emcee.sh for gc_name: ${gc_name}"
python src/emcee_wrapper.py -gc "${gc_name}" -m "king"   -Nw 32 -Ns 50000 -Nb 5000
python src/emcee_wrapper.py -gc "${gc_name}" -m "wilson" -Nw 32 -Ns 50000 -Nb 5000
python src/emcee_wrapper.py -gc "${gc_name}" -m "limepy" -Nw 32 -Ns 50000 -Nb 5000
python src/emcee_wrapper.py -gc "${gc_name}" -m "spes"   -Nw 32 -Ns 50000 -Nb 5000
