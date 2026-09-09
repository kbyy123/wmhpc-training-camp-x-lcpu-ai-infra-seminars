"""Write all 63 arithmetic-intensity predictions before running thin GEMM."""
import csv
from pathlib import Path

shapes=[('f_b_proj',1536,128),('q_b_proj',2304,1536),('o_proj',7168,1536),
        ('fused_qkv_a_proj',2112,7168),('in_proj_qkvgfab',6288,7168),
        ('dense_down_proj',7168,8448),('dense_gate_up_proj',16896,7168)]
out=Path(__file__).resolve().parents[1]/'results/2026-09-09/thin_roofs.csv'
out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w',newline='') as f:
    w=csv.writer(f,lineterminator='\n')
    w.writerow(['layer','M','N','K','AI_FLOP_per_B','TC_roof_TFLOPS','BW_roof_TFLOPS','roof_TFLOPS','predicted_bound'])
    for layer,n,k in shapes:
        for m in [1,8,16,64,256,1024,4096,16384,65536]:
            ai=m*n*k/(m*k+n*k+m*n)
            w.writerow([layer,m,n,k,ai,2250,ai*8,min(2250,ai*8),'memory' if ai<281.25 else 'compute'])
print('PASS: wrote 63 roofline predictions; peak=2250 TFLOPS, bandwidth=8000 GB/s')
