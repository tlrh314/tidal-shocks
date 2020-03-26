#!/bin/bash
set -e

which mpiexec
python -c "import mpi4py; print(mpi4py)"

echo -e "\nRunning mpiexec /w mpi4py test w/o hostfile"
mpiexec -n 1 python -m mpi4py.bench helloworld
echo "mpiexec exit code: $?"  # unfortunately 1

echo -e "\nRunning mpiexec /w mpi4py test with a hostfile"
mpiexec --hostfile hostfile_mpich2 -n 5 python -m mpi4py.bench helloworld
echo "mpiexec exit code: $?"  # should be 0

echo "Check Hydra thingy"

echo -e "\nRunning mpiexec.hydra /w mpi4py test w/o hostfile"
mpiexec.hydra -n 5 python -m mpi4py.bench helloworld
echo "mpiexec exit code: $?"  # unfortunately 1

echo -e "\nRunning mpiexec.hydra /w mpi4py test with a hostfile"
mpiexec.hydra --hostfile hostfile_mpich2 -n 5 python -m mpi4py.bench helloworld
echo "mpiexec exit code: $?"  # should be 0
