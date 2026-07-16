"""Reproducible OEFA train-step benchmark and profiler (no long training).

Example: python tools/benchmark_oefa.py --batch 128 --imgsz 640 --warmup 20 --iters 50 --amp --profile
"""

import argparse
import time
from pathlib import Path

import torch
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel

MODELS = {
    "legacy_identity": "ultralytics/cfg/models/oefa/yolo11n-oefa-identity.yaml",
    "legacy_m1": "ultralytics/cfg/models/oefa/yolo11n-oefa-m1.yaml",
    "identity_v2": "ultralytics/cfg/models/oefa/yolo11n-oefa-identity-v2.yaml",
    "m1_v2": "ultralytics/cfg/models/oefa/yolo11n-oefa-m1-v2.yaml",
}


def make_batch(n, size, device):
    return {"img": torch.rand(n,3,size,size,device=device), "batch_idx": torch.arange(n,device=device),
            "cls": torch.zeros(n,1,device=device), "bboxes": torch.tensor([[.5,.5,.2,.2]],device=device).repeat(n,1)}


def main(a):
    if not torch.cuda.is_available(): raise SystemExit("CUDA is required for synchronized benchmark results")
    device=torch.device("cuda"); rows=[]
    names = list(MODELS) if a.model == "all" else [a.model]
    for name in names:
        model=DetectionModel(MODELS[name], nc=a.nc, verbose=False).to(device).train(); model.args=get_cfg()
        model.criterion = model.init_criterion()
        hook_handles=[]
        if a.profile:
            from ultralytics.nn.modules import Detect
            from ultralytics.nn.modules.oefa_v2 import EvidenceTargetGeneratorV2, OEFABoundaryDownsampleV2, OEFAEvidencePredictorV2, OEFAGuidedSamplerV2
            def label(module):
                if isinstance(module, OEFAGuidedSamplerV2): return "P5toP4 sampler" if module.evidence_level == 1 else "P4toP3 sampler"
                if isinstance(module, OEFABoundaryDownsampleV2): return "boundary downsample P3toP4" if module.evidence_level == 0 else "boundary downsample P4toP5"
                if isinstance(module, EvidenceTargetGeneratorV2): return "target generator"
                if isinstance(module, OEFAEvidencePredictorV2): return "evidence predictor"
                if type(module) is Detect: return "Detect"
            for module in (*model.modules(), *model.criterion.target_generator.modules()) if hasattr(model.criterion, "target_generator") else model.modules():
                tag=label(module)
                if tag:
                    def pre(mod, inp, tag=tag): mod._oefa_profiler_scope=torch.profiler.record_function(tag); mod._oefa_profiler_scope.__enter__()
                    def post(mod, inp, out): mod._oefa_profiler_scope.__exit__(None,None,None)
                    hook_handles.extend((module.register_forward_pre_hook(pre),module.register_forward_hook(post)))
        optimizer=torch.optim.SGD(model.parameters(),lr=1e-3); batch=make_batch(a.batch,a.imgsz,device)
        scaler=torch.amp.GradScaler("cuda", enabled=a.amp)
        def step():
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16,enabled=a.amp): loss=model(batch)[0].sum()
            with torch.profiler.record_function("backward") if a.profile else torch.autograd.profiler.record_function("backward"):
                scaler.scale(loss).backward()
            optimizer.step() if not a.amp else scaler.step(optimizer); scaler.update()
        for _ in range(a.warmup): step()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
        if a.profile:
            activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]
            with torch.profiler.profile(activities=activities,profile_memory=True,record_shapes=True) as prof:
                for _ in range(a.iters): step()
            prof.export_chrome_trace(str(Path(a.output)/f"{name}.json"))
            print(prof.key_averages().table(sort_by="self_cuda_time_total",row_limit=40))
        else:
            for _ in range(a.iters): step()
        torch.cuda.synchronize(); elapsed=(time.perf_counter()-start)*1000/a.iters
        rows.append((name,elapsed,torch.cuda.max_memory_allocated()/2**30))
        for handle in hook_handles: handle.remove()
        del model,optimizer,batch; torch.cuda.empty_cache()
    print("model\ttrain_step_ms\tmax_vram_GiB")
    for r in rows: print(f"{r[0]}\t{r[1]:.3f}\t{r[2]:.3f}")


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--model",choices=["all",*MODELS],default="all")
    p.add_argument("--batch",type=int,default=128); p.add_argument("--imgsz",type=int,default=640)
    p.add_argument("--warmup",type=int,default=20); p.add_argument("--iters",type=int,default=50)
    p.add_argument("--nc",type=int,default=80); p.add_argument("--amp",action="store_true"); p.add_argument("--profile",action="store_true")
    p.add_argument("--output",default="oefa_profiles"); a=p.parse_args(); Path(a.output).mkdir(parents=True,exist_ok=True); main(a)
