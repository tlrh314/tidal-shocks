#!/bin/bash -l
#SBATCH -o ./tjob.%j.out
#SBATCH -e ./tjob.%j.err
#SBATCH -D ./
#SBATCH -J test.sh
#SBATCH -N 1
#SBATCH --ntasks-per-node=40
#SBATCH --mail-type=all
#SBATCH --mail-user=timoh@rzg.mpg.de
#SBATCH --time=00:02:00

set -e

export PYTHONUNBUFFERED=1
# export AMUSE_MPD_CHECK=0
export I_MPI_DEBUG=10
 
if [[ $(hostname -s) = freya* ]]; then
    echo "Freya"

    if [ $(which conda) == "/u/timoh/conda-envs/tidalshocks/bin/conda" ]; then
        echo "Conda env tidalshocks already loaded"
    else 
        echo "Loading Conda env tidalshocks"
        setup_tidalshocks
    fi

    # set_interactive
fi

env >> env_in_sbatch_${SLURM_JOBID}
which mpiexec
which python
pwd
# python -c "import mpi4py; print(mpi4py)"
echo "python test_amuse.py"
mpirun -n 1 python test_amuse.py
echo "done with mpiexec python test_amuse.py"
exit 0

echo -e "\nRunning mpiexec /w mpi4py test w/o hostfile"
mpiexec -n 4 python -m mpi4py.bench helloworld
echo "mpiexec exit code: $?"  # unfortunately 1

# echo -e "\nRunning mpiexec /w mpi4py test with a hostfile"
# mpiexec --hostfile hostfile_mpich2 -n 5 python -m mpi4py.bench helloworld
# echo "mpiexec exit code: $?"  # should be 0

echo "Check Hydra thingy"

echo -e "\nRunning mpiexec.hydra /w mpi4py test w/o hostfile"
mpiexec.hydra -n 4 python -m mpi4py.bench helloworld
echo "mpiexec exit code: $?"  # unfortunately 1

# echo -e "\nRunning mpiexec.hydra /w mpi4py test with a hostfile"
# mpiexec.hydra --hostfile hostfile_mpich2 -n 5 python -m mpi4py.bench helloworld
# echo "mpiexec exit code: $?"  # should be 0

which srun

echo "Check srun"

echo -e "\nRunning srun /w mpi4py test w/o hostfile"
srun -n 4 python -m mpi4py.bench helloworld
echo "srun exit code: $?"  # unfortunately 1
