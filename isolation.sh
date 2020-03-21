#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./tjob.%j.out
#SBATCH -e ./tjob.%j.err
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J isolation.sh
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
echo "Running isolation.sh for gc_name: ${gc_name}"
export OMP_NUM_THREADS=16
python src/test_stability_in_isolation.py -gc "${gc_name}" -m "king" -N 1000 -t 1000 --Nsnap 100
