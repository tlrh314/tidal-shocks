#!/bin/bash

set -e

NOW=$(date +"%Y%m%dT%H%M")
if [ -d runs/$NOW ]; then
    echo -e "ERROR: folder $NOW exists\n"
    exit 1
else
    mkdir $NOW
    echo "SUCCESS: folder $NOW created\n"
fi

echo -e "Compiling nbody6tt\n"
cd ../nbody6tt/Ncode
make nbody6
cp nbody6 ../../tidal-shocks/runs/$NOW
cd ../../tidal-shocks

echo -e "Running with the following inbins\n"
cp inbins runs/$NOW
cd runs/$NOW
cat inbins


# time nbody6 <input >out &
# wait
