"""Problem 6.1: save actual TileLang 0.1.13 CUDA, host/device IR and compile both targets."""
import importlib.util
from pathlib import Path
import tilelang

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('a1_matmul',ROOT.parent/'assignment01/kernels/tilelang_matmul.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def main():
    out=ROOT/'results/2026-09-09/tilelang'
    out.mkdir(parents=True,exist_ok=True)
    f=module.make_matmul(1024,1024,1024,BLOCK_M=128,BLOCK_N=128,BLOCK_K=64,threads=128,num_stages=3)
    (out/'input.tir').write_text(f.script())
    print('tilelang',tilelang.__version__)
    for arch in ['sm_90a','sm_100a']:
        target={'kind':'cuda','arch':arch}
        target=tilelang.tvm.target.Target(target)
        with target:
            artifact=tilelang.lower(f,target=target,enable_device_compile=True)
        (out/(arch+'.cu')).write_text(artifact.kernel_source.rstrip()+'\n')
        (out/(arch+'.device.tir')).write_text(artifact.device_mod.script())
        (out/(arch+'.host.tir')).write_text(artifact.host_mod.script())
        print(arch,'PASS device compilation',len(artifact.kernel_source),'source bytes',flush=True)

if __name__=='__main__': main()
