"""Audit the complete requested evidence set without requiring another GPU run."""
import json
import re
from pathlib import Path

root=Path(__file__).resolve().parents[1]
r=root/'results/2026-09-09'
def read(name): return (r/name).read_text()
summary={}
for file,count in [('stages.txt',24),('pair_stages.txt',18)]:
    text=read(file)
    assert 'FAIL' not in text
    assert text.count('PASS(bad=0)')==count
    summary[file]={'strict_gemm_passes':count}
basic=read('basic.txt')
assert 'PASS: 202864 values match hardware' in basic and 'FAIL' not in basic
assert basic.count('PASS(bad=0)')==3 and basic.count('maxrel=')==3
summary['nvfp4']={'hardware_encoder_values':202864,'byte_exact_shapes':3,'cublaslt_shapes':3}
assert read('quant_edges.txt').count('PASS bad=0')==6
summary['extra_edge_checks']=6
for name in ['memcheck_pipeline.txt','memcheck_pair.txt']:
    assert 'ERROR SUMMARY: 0 errors' in read(name)
summary['memcheck']='both pipelines: 0 errors'
fused=read('fused_rms.txt')
assert 'FAIL' not in fused and len(re.findall(r'x PASS\(bad=\d+\)',fused))==10
summary['fused_shapes_passed']=10
thin=[line for line in read('thin_gemm.txt').splitlines()
      if len(line.split())==10 and line.split()[1].isdigit()]
assert len(thin)==63
assert len(read('thin_roofs.csv').splitlines())==64
summary['thin_shapes_measured']=63
assert '3 passed' in read('block_scale.txt')
assert 'ratio_with/without=149.40993' in read('quant_outlier.txt')
summary['python_tests']='3 passed; outlier experiment complete'
for target in ['sm_90a','sm_100a']:
    assert target+' PASS device compilation' in read('tilelang_compile.txt')
    for suffix in ['.cu','.host.tir','.device.tir']:
        assert (r/'tilelang'/(target+suffix)).stat().st_size>1000
summary['tilelang_compiled_targets']=['sm_90a','sm_100a']
assert 'PASS bad=0' in read('naive.txt')
for name in ['ncu_quant.csv','ncu_probe.csv','ncu_pipeline.csv','ncu_fused.csv']:
    assert 'sm__throughput.avg.pct_of_peak_sustained_elapsed' in read(name)
assert len(read('gpu_after_full.txt').strip().splitlines())==1
answer=(root/'handout/src/assignment02-answer.md').read_text()
assert not [c for c in answer if ord(c)<32 and c not in '\n\t']
assert answer.count('> | dense_gate_up_proj |')==9
summary['status']='PASS: all requested required code, experiments, reports and 4.4(a) verified'
(r/'validation_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
