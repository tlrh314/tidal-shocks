#!/bin/bash
unset -e

echo -e "\nRunning mpiexec /w mpi4py test w/o hostfile"
mpiexec -n 5 python -m mpi4py.bench helloworld
echo "mpiexec exit code: $?"  # unfortunately 1

echo -e "\nRunning mpiexec /w mpi4py test with a hostfile"
mpiexec --hostfile hostfile -n 5 python -m mpi4py.bench helloworld
echo "mpiexec exit code: $?"  # should be 0
