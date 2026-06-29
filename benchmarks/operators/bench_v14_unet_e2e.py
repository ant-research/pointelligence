"""Production ResUNet (PointCNN++) end-to-end: real ScanNet, fwd-only + fwd+bwd,
version arms v1.0/v1.3/v1.4, small + large batch. v1.4 is the shipped
auto route; pass --include-force-fused to time the diagnostic force route. The seg UNet is the real
workload (conv-heavier than the classification ResNet); this decides whether the
operator win reaches e2e."""
import os, sys, argparse
import numpy as np, torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# pointcept comes from the Pointcept overlay build tree (OVERLAY_ROOT_POINTCEPT on
# the cluster; <repo>/build/Pointcept locally after overlays/Pointcept/build.sh).
_PCEPT = os.environ.get("OVERLAY_ROOT_POINTCEPT", os.path.join(REPO, "build", "Pointcept"))
sys.path.insert(0, _PCEPT)
sys.path.insert(0, REPO)
os.environ.setdefault("POINTELLIGENCE_GEOMETRY_SCHEDULER", "0")
from sparse_engines._dispatch_override import dispatch_mode
from pointcept.models.sparse_unet.unet_pointcnnpp import ResUNet

# Mixed precision: convs run fp16 (fused-gather-sum eligible), BatchNorm runs fp32
# internally with fp16 I/O — the standard pattern, matches production AMP where
# BN stays fp32. Without this, fp16 BN raises "expected Float but found Half".
def _bn_mixed(self, x):
    rm = self.running_mean; rv = self.running_var
    return torch.nn.functional.batch_norm(
        x.float(),
        rm.float() if rm is not None else None,
        rv.float() if rv is not None else None,
        self.weight.float() if self.weight is not None else None,
        self.bias.float() if self.bias is not None else None,
        self.training or not self.track_running_stats,
        self.momentum if self.momentum is not None else 0.0,
        self.eps,
    ).to(x.dtype)
torch.nn.BatchNorm1d.forward = _bn_mixed
# Linear/pool feeders can hand fp32 into fp16 layers in manual mixed precision;
# cast the input to the layer's own dtype (production AMP does this implicitly).
_orig_lin = torch.nn.Linear.forward
def _lin_cast(self, x):
    return _orig_lin(self, x.to(self.weight.dtype))
torch.nn.Linear.forward = _lin_cast

dev = "cuda"
SCAN = os.environ["TIG_BENCH_SCANNET"]
SCENES = [
    "scene0011_00", "scene0015_00", "scene0019_00", "scene0025_00",
    "scene0030_00", "scene0046_00", "scene0050_00", "scene0064_00",
    "scene0086_00", "scene0095_00", "scene0100_00", "scene0131_00",
    "scene0144_00", "scene0153_00", "scene0164_00", "scene0169_00",
    "scene0187_00", "scene0193_00", "scene0207_00", "scene0217_00",
    "scene0221_00", "scene0011_01", "scene0019_01", "scene0249_00"]
GS = 0.025
BASE_ARMS = [("v1.0","force_pt"),("v1.3","force_tig"),("v1.4","auto")]


def dedup_first(c, gs):
    g=(c/gs).long(); u,inv=torch.unique(g,dim=0,return_inverse=True)
    f=torch.full((u.size(0),),c.size(0),device=dev,dtype=torch.long)
    f.scatter_reduce_(0,inv,torch.arange(c.size(0),device=dev),reduce="amin"); return c[f.sort().values]

def make_input(raws, B):
    cs,offs=[],[0]
    for c in raws[:B]:
        d=dedup_first(c,GS); cs.append(d); offs.append(offs[-1]+d.size(0))
    coord=torch.cat(cs,0); N=coord.size(0)
    torch.manual_seed(0); feat=torch.randn(N,1,device=dev,dtype=torch.float16)
    offset=torch.tensor(offs[1:],dtype=torch.long,device=dev)
    return {"feat":feat,"coord":coord,"offset":offset,"grid_size":GS}, N

def timed(fn, it=8, warm=3):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); s,e=torch.cuda.Event(True),torch.cuda.Event(True); s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/it

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--batches",default="2,4")
    ap.add_argument("--include-force-fused", action="store_true",
                    help="also time the diagnostic force_fused_gather_sum route")
    args=ap.parse_args(); batches=[int(b) for b in args.batches.split(",")]
    arms=list(BASE_ARMS)
    if args.include_force_fused:
        arms.append(("v1.4-force","force_fused_gather_sum"))
    raws=[torch.from_numpy(np.load(f"{SCAN}/{s}/coord.npy")).float().to(dev) for s in SCENES[:max(batches)]]
    model=ResUNet(in_channels=1,num_classes=16,base_channels=32,voxel_size=GS,normalize_feature=False).to(dev).to(torch.float16)
    # parity (B=2)
    inp,_=make_input(raws,2)
    outs={}
    for _,mode in [("v1.3","force_tig"),("v1.4","auto")]:
        with torch.no_grad(), dispatch_mode(mode):
            outs[mode]=model(inp).float()
    a,b=outs["force_tig"],outs["auto"]
    rel=(a-b).abs().max().item()/(a.abs().max().item()+1e-6)
    print(f"ResUNet parity auto-v1.4 vs force_tig: rel={rel:.2e} shape={tuple(a.shape)} -> {'PASS' if rel<5e-2 else 'FAIL'}")
    for B in batches:
        inp,N=make_input(raws,B)
        tgt=torch.randint(0,16,(N,),device=dev)
        print(f"\n=== ResUNet base32  B={B}  N={N}  (real ScanNet) ===")
        for regime in ("val","train"):
            line=f"  {regime:5s} | "; base=None
            for aname,mode in arms:
                def run():
                    if regime=="val":
                        with torch.no_grad(), dispatch_mode(mode): model(inp)
                    else:
                        for p in model.parameters(): p.grad=None
                        with dispatch_mode(mode):
                            out=model(inp); torch.nn.functional.cross_entropy(out,tgt).backward()
                try: ms=timed(run)
                except Exception as ex:
                    ms=float("nan"); print(f"    {aname} {regime} ERR {type(ex).__name__}: {str(ex)[:120]}")
                if base is None: base=ms
                line+=f"{aname} {ms:.1f}({base/ms:.2f}x) "
            print(line)

if __name__=="__main__": main()
