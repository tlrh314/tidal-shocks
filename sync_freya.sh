#!/bin/bash
set -e

if [[ $(hostname -s) = freya* ]]; then
    echo "Freya"
elif [[ $(hostname -s) = ZoltaN ]]; then
    echo "ZoltaN"
    rsync -auHxv --progress freya:/u/timoh/phd/tidalshocks/out/ out/
elif [[ $(hostname -s) = ChezTimo* ]]; then
    echo "MBP"
fi
