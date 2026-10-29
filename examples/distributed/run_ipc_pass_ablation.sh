#!/usr/bin/env bash
# Run inside an existing one-node/four-DCU allocation; never starts extra jobs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/dev_old/env.sh
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_coll='^hcoll,ucc'
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
test -n "${SLURM_JOB_ID:-}"
result_root="${1:-build-old/ipc-ablation-20261014}"
for round in 1 2; do
    # Reverse the case order in round two; each case itself alternates A/B/C.
    if [[ "$round" == 1 ]]; then
        kinds=(rs ar ag); sizes=(2048 4096 8192 16384)
    else
        kinds=(ag ar rs); sizes=(16384 8192 4096 2048)
    fi
    for kind in "${kinds[@]}"; do
        for size in "${sizes[@]}"; do
            case_dir="$result_root/round$round/$kind$size"
            mkdir -p "$case_dir"
            # Resume without overwriting completed measurements.
            if [[ -s "$case_dir/rank0.json" && -s "$case_dir/rank1.json" &&
                  -s "$case_dir/rank2.json" && -s "$case_dir/rank3.json" ]]; then
                echo "PRESERVED round=$round kind=$kind size=$size"
                continue
            fi
            timeout --signal=TERM --kill-after=15s 900s \
                mpirun -np 4 --oversubscribe --bind-to none --stdin none \
                python examples/distributed/benchmark_ipc_pass_ablation.py \
                --kind "$kind" --size "$size" --warmup 30 --repeats 30 \
                --output "$case_dir" > "$case_dir/run.log" 2>&1
            for rank in 0 1 2 3; do test -s "$case_dir/rank$rank.json"; done
            echo "COMPLETED round=$round kind=$kind size=$size"
        done
    done
done
python examples/distributed/summarize_ipc_pass_ablation.py "$result_root"
