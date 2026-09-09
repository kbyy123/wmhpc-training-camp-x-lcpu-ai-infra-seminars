#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
OUT="$PWD/results/2026-09-09"
timeout -k 5 180 .venv/bin/python kernels/lower_matmul.py > "$OUT/tilelang_compile.txt" 2>&1
cd cuda
make -j2 bin/m4_gemm/00_naive_ladder bin/m4_gemm/inspect_pipeline > "$OUT/build_extra.txt" 2>&1
timeout -k 5 90 bin/m4_gemm/00_naive_ladder > "$OUT/naive.txt"
bin/m4_gemm/inspect_pipeline > "$OUT/resources.txt"
METRICS=sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,launch__registers_per_thread,launch__shared_mem_per_block,launch__occupancy_limit_shared_mem
for item in 'quant m5_lowprec/03b_nvfp4_quant nvfp4_quant_kernel 442' 'probe m5_lowprec/03c_ceiling_probe probe_kernel 0' 'pipeline m4_gemm/pipeline_s3 gemm_pipeline 0'; do
    read -r label exe name skip <<< "$item"
    timeout -k 5 150 ncu --csv --kernel-name-base function --kernel-name "regex:$name" --launch-skip "$skip" --launch-count 1 --metrics "$METRICS" "bin/$exe" > "$OUT/ncu_$label.csv" 2>&1
done
# Profile the chosen large shape, skipping autotuning (15 candidates * 40 launches).
timeout -k 5 180 ncu --csv --kernel-name-base function --kernel-name 'regex:fused_rms_kernel' --launch-skip 600 --launch-count 1 --metrics "$METRICS" bin/m5_lowprec/04_fused_rms_nvfp4 16384 8192 > "$OUT/ncu_fused.csv" 2>&1
nvidia-smi --query-compute-apps=pid,name --format=csv > "$OUT/gpu_after_full.txt"
printf 'PROFILE SUITE PASS\n'
