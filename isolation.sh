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
export OMP_NUM_THREADS=8
for seed in 1337; do
    for N in 1000; do
        for softening in 0.1 1.0 10 0.01; do
            for model in "king"; do
                echo $seed $N $softening $model

                python src/test_stability_in_isolation.py -gc "${gc_name}" \
                    -m "$model" -N $N --softening $softening  \
                    -t 100 --Nsnap 100 -c "gadget2" --seed $seed -np 8

                dir="${model}_isolation_${N}_${softening}_${seed}"
                if [ ! -d "out/ngc104/ngc104_${dir}" ]; then 
                    mkdir "out/ngc104/ngc104_${dir}"
                fi
                for f in $(ls "out/ngc104/ngc104_${dir}_"*); do
                    mv $f "out/ngc104/ngc104_${dir}/"
                done
                break
            done
            break
        done
        break
    done
    break
done
