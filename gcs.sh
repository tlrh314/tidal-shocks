#!/bin/bash
set -e


# What"s up with the order of the GCs? (This comes from deB19 Appendix A)
gcs=(
    "NGC 104"  "NGC 288"  "NGC 362"  "NGC 1261" "Pal 1"    "NGC 1851" 
    "NGC 1904" "NGC 2298" "NGC 2419" "NGC 2808" "NGC 3201" "NGC 4147"
    "NGC 4590" "NGC 5024" "NGC 5053" "NGC 5139" "NGC 5272" "NGC 5286"
    "NGC 5466" "NGC 5634" "NGC 5694" "IC 4499"  "NGC 5824" "NGC 5897" 
    "NGC 5904" "NGC 5986" "NGC 6093" "NGC 6121" "NGC 6101" "NGC 6144"
    "NGC 6139" "NGC 6171" "NGC 6205" "NGC 6229" "NGC 6218" "NGC 6235"
    "NGC 6254" "NGC 6266" "NGC 6273" "NGC 6284" "NGC 6293" "NGC 6341"
    "NGC 6325" "NGC 6333" "NGC 6352" "NGC 6366" "NGC 6362" "NGC 6388"
    "NGC 6402" "NGC 6397" "NGC 6426" "NGC 6496" "NGC 6539" "NGC 6541"
    "IC 1276"  "NGC 6569" "NGC 6584" "NGC 6624" "NGC 6626" "NGC 6637"
    "NGC 6652" "NGC 6656" "Pal 8"    "NGC 6681" "NGC 6715" "NGC 6717"
    "NGC 6723" "NGC 6752" "NGC 6779" "NGC 6809"            "Pal 11"   
    "NGC 6864" "NGC 6934" "NGC 6981" "NGC 7006" "NGC 7078" "NGC 7089" 
    "NGC 7099" "Pal 12" "NGC 7492"
)

for gc in "${gcs[@]}"; do
    job="./emcee.sh ${gc}"
    echo "Submitting job: ${job}"
    exec $job
    break
done
