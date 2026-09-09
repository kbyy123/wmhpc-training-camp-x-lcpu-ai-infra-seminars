#!/usr/bin/env bash
# Run inside the GPU allocation: srun --jobid=JOB -n1 bash assignment02/run_requested.sh
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
OUT="$PWD/results/2026-09-09"
mkdir -p "$OUT"
nvidia-smi > "$OUT/gpu_environment.txt"
nvidia-smi --query-compute-apps=pid,name --format=csv > "$OUT/gpu_before_full.txt"
nvcc --version > "$OUT/nvcc.txt"
.venv/bin/python -m pytest tests/test_block_scale.py -v > "$OUT/block_scale.txt"
.venv/bin/python kernels/quant_outlier.py > "$OUT/quant_outlier.txt"
.venv/bin/python kernels/thin_roofline.py > "$OUT/thin_prediction.txt"
cd cuda
make -j4 -B bin/m4_gemm/01_tiled bin/m4_gemm/02_tma bin/m4_gemm/00_naive_ladder \
    bin/m5_lowprec/03a_encode_check bin/m5_lowprec/03b_nvfp4_quant bin/m5_lowprec/test_fp4_gemm \
    bin/m5_lowprec/03c_ceiling_probe bin/m5_lowprec/04_fused_rms_nvfp4 bin/m5_lowprec/test_quant_edges \
    bin/m4_gemm/05_thin_gemm > "$OUT/build_baseline.txt" 2>&1
for target in m5_lowprec/03a_encode_check m5_lowprec/03b_nvfp4_quant m5_lowprec/test_fp4_gemm m5_lowprec/03c_ceiling_probe; do
    timeout -k 5 90 bin/$target
done > "$OUT/basic.txt"
timeout -k 5 90 bin/m5_lowprec/test_quant_edges > "$OUT/quant_edges.txt"
timeout -k 5 90 bin/m4_gemm/00_naive_ladder > "$OUT/naive.txt"
for name in 01_tiled 02_tma; do
    timeout -k 5 90 bin/m4_gemm/$name 4096 4096 4096
done > "$OUT/ladder.txt"
for s in 2 3 4 6; do
    STAGES=$s make -sB bin/m4_gemm/03_pipeline
    cp bin/m4_gemm/03_pipeline bin/m4_gemm/pipeline_s$s
    for shape in '128 64 64' '256 128 128' '256 128 320' '1024 1024 1024' '4096 4096 4096' '256 4096 16384'; do
        timeout -k 5 90 bin/m4_gemm/pipeline_s$s $shape
    done
done > "$OUT/stages.txt" 2>&1
for s in 3 6 10; do
    STAGES=$s make -sB bin/m4_gemm/04_pipeline_pair
    cp bin/m4_gemm/04_pipeline_pair bin/m4_gemm/pair_s$s
    for shape in '256 64 64' '256 128 128' '256 128 320' '1024 1024 1024' '4096 4096 4096' '256 4096 16384'; do
        timeout -k 5 90 bin/m4_gemm/pair_s$s $shape
    done
done > "$OUT/pair_stages.txt" 2>&1
cp bin/m4_gemm/pipeline_s3 bin/m4_gemm/03_pipeline
cp bin/m4_gemm/pair_s3 bin/m4_gemm/04_pipeline_pair
timeout -k 5 240 bin/m4_gemm/05_thin_gemm 2250 8000 > "$OUT/thin_gemm.txt"
timeout -k 5 480 bin/m5_lowprec/04_fused_rms_nvfp4 > "$OUT/fused_rms.txt"
cd ..
timeout -k 5 180 .venv/bin/python kernels/lower_matmul.py > "$OUT/tilelang_compile.txt" 2>&1
nvidia-smi --query-compute-apps=pid,name --format=csv > "$OUT/gpu_after_full.txt"
printf 'REQUESTED SUITE PASS\n'
