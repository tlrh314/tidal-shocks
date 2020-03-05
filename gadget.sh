#!/bin/bash
set -e

NOW=$(date +"%Y%m%dT%H%M")
echo "1. Creating a new simulation folder: $NOW"
if [ -d runs/$NOW ]; then
    echo -e "  ERROR: folder $NOW exists\n"
    exit 1
else
    mkdir runs/$NOW
    echo -e "  SUCCESS: folder $NOW created\n"
fi

echo "2. Compiling Gadget3"
cd P-Gadget3
if [ -f compileGadget.log ]; then 
    rm compileGadget.log;
fi
make -j >> compileGadget.log 2>&1
echo -e "Done compiling P-Gadget3\n"
for fname in Makefile Config.sh gadget3.par P-Gadget3 compileGadget.log; do
    if [ ! -f $fname ]; then 
        echo -e "  ERROR: folder fname does not exist\n"
        exit 1
    else
        cp $fname ../runs/$NOW
    fi
done
rm compileGadget.log;

echo -e "\n3. Starting simulation"
cd ../runs/$NOW
OMP_NUM_THREADS=1 nice -n 19 mpirun -np 4  --use-hwthread-cpus \
  ./P-Gadget3 gadget3.par >> runGadget.log 2>&1
wait
echo "Done running simulation"
