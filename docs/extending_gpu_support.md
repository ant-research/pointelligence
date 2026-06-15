# Extending GPU support — porting beyond RTX 5880 Ada / H200

The kernels in this repo (the TIG convolution, the no-atomic segment weight-gradient, the
grouped FSG tiers, the grid-partition disjoint conv) are **developed and tuned on two
GPUs**: a workstation **RTX 5880 Ada** (`sm_89`) and a data-center **H200** (`sm_90`).
They are written in Triton + CUTLASS, so they **run correctly on any recent NVIDIA GPU**
(Ampere `sm_80` and later). What is *tuned* for those two targets is the **dispatch** — a
small set of architecture-dependent gates that choose which kernel variant is fastest. On
an untested GPU the dispatch falls back to the nearest-architecture defaults: **correct,
but not necessarily optimal**. This guide shows how to (1) confirm correctness on your GPU
and (2) re-tune the dispatch for it.

## TL;DR

1. **Correctness is architecture-independent — verify it first.** `pytest` the parity
   batteries; they compare every kernel against an fp64 reference (a hardware-independent
   ground truth). If they pass, the kernels are numerically correct on your GPU and the
   defaults will *work* — you can stop here.
2. **Performance dispatch is tuned for `sm_89` / `sm_90`.** Your GPU inherits the nearest
   tier (`cc < 9.0` → the Ada tier; `cc ≥ 9.0` → the Hopper tier). To get the *best* speed,
   re-measure the crossovers below and add an architecture tier — a ~3-line code change.

## 1. What is, and isn't, architecture-specific

| Component | Architecture-specific? |
|---|---|
| Numerical results (every kernel, every dtype) | **No** — Triton/CUTLASS target the arch; parity vs the fp64 reference holds everywhere |
| Which CUTLASS FSG variant runs (`sm_80` vs `sm_90`) | Auto-selected by arch — no user action |
| The no-atomic **segment vs. atomic** weight-gradient gate | **Yes** — `_seg_gate_params()` |
| The **FI1 plain-store** forward / grad-input / fp32-wgrad gate | **Yes** — `_fi1_wins_here()` |
| The single-cloud **compile vs. fuse** inference crossover | **Yes** — a deployment choice, measured |

## 2. Step 1 — verify correctness (do this first; ~minutes)

```bash
pytest tests/unittest/test_tig_forward.py tests/unittest/test_tig_backward.py \
       tests/unittest/test_conv_with_stride_disjoint_parity.py \
       tests/unittest/test_upsample_recompute_k.py \
       tests/unittest/test_sparse_linalg_fp16.py tests/unittest/test_sparse_linalg_bf16.py \
       tests/unittest/test_input_precision.py tests/unittest/test_fp32_routing_precision.py
```

These check kernel outputs against an fp64 block-diagonal reference — independent of the
hardware. If they pass, the kernels are correct on your GPU; everything below is purely
about speed. If a parity test *fails* (rather than just running slower), that is a bug —
please report it with your GPU model and the failing test.

## 3. Step 2 — the architecture-tuned gates and their defaults

### (a) Segment-vs-atomic weight-gradient gate — `_seg_gate_params()` (`sparse_engines/tig.py`)

The no-atomic segment kernel beats the atomic Split-K only above a joint `(C, K)` threshold
(wide channels or high kernel volume), and that threshold is **architecture-dependent**: a
GPU with cheaper global atomics needs a *tighter* gate, since the atomic path it competes
against is faster there. The shipped values:

```
(C_wide, C_hi) = (256, 128)   on cc < 9.0   (Ada / Ampere tier)
               = (512, 256)   on cc ≥ 9.0   (Hopper tier; cheaper fp32 atomics)
   K_hi = 125
segment iff  C ≥ C_wide  OR  (C ≥ C_hi AND K ≥ K_hi)
```

Your untested GPU inherits whichever tier its compute-capability falls into — a safe
default, not a tuned value.

### (b) FI1 plain-store gate — `_fi1_wins_here()` (`sparse_engines/tig.py`)

FI1 (the fan-in-1 plain-store forward + grad-input, plus the fp32 weight-gradient path) is
enabled on `cc < 9.0` and disabled on Hopper, where the atomic path is faster. An untested
sub-Hopper GPU gets FI1 enabled by default.

### (c) Single-cloud compile-vs-fuse inference crossover

For batch-1 inference, `torch.compile` on the separable feature trunk wins in the
**launch-bound** regime (small/medium models at room-scale scenes), while explicit fusion
wins in the **compute/bandwidth-bound** regime (wide models or apartment-scale clouds). The
crossover point shifts with the GPU's tensor-core throughput. This is a deployment choice,
not a code gate.

## 4. Step 3 — re-tune for your GPU (optional, for maximum performance)

Measure the segment-vs-atomic crossover on **your** GPU, on **real** point clouds:

```bash
TIG_BENCH_SCANNET=/path/to/scannet_v2/val \
    python benchmarks/operators/bench_v13_dispatch_retune.py --batch <your batch>
```

Each `(C, K)` row prints `opt=seg|atomic` (the empirically fastest route) next to
`gate=seg|atomic` (what the current gate picks). Read off the `C` at which `opt` flips to
`seg` to get *your* `(C_wide, C_hi)`. Cross-check with `bench_v13_tig_kernels.py`
(kernel-isolated) and `bench_v13_versions.py` (all generations, train + val). For inference
deployment, sweep the compile/fuse arms with the end-to-end backbone bench.

> **Always benchmark on real point clouds** (set `TIG_BENCH_SCANNET`), never synthetic: the
> segment-vs-atomic crossover depends on the real neighbor fan-in distribution **and** on
> batch size, both of which synthetic clouds get wrong. Use your production batch size.

## 5. Step 4 — add an architecture tier (a ~3-line change)

With your measured crossover in hand, add a branch to `_seg_gate_params()`:

```python
def _seg_gate_params():
    dev = torch.cuda.current_device()
    cc = torch.cuda.get_device_capability(dev)
    if cc >= (9, 0):          # Hopper and newer
        return (512, 256)
    if cc == (8, 0):          # e.g. A100 — your measured values
        return (CW, CH)
    return (256, 128)         # Ada / default
```

If the forward bench shows FI1 behaving differently on your GPU, adjust `_fi1_wins_here()`
the same way. No other code changes are needed — the kernels themselves are
architecture-agnostic; only the routing thresholds are tuned.

## 6. Requirements and caveats

- **Minimum architecture:** Ampere (`sm_80`) or later for the full design (bf16 tensor
  cores + the CUTLASS FSG tiers). fp16-only operation on Turing (`sm_75`) may work but is
  untested.
- **fp32** accumulates in IEEE by default on every architecture (TF32 is opt-in) — no
  per-architecture numerics change.
- **Newer-than-Hopper GPUs** (`cc ≥ 9.0`, e.g. Blackwell) inherit the Hopper dispatch tier
  by default; re-tune as above if their crossover differs.
- The defaults are conservative by design: an unrecognized architecture always lands on a
  *correct* route; re-tuning only changes *which fast path* runs, never the result.

## Citation

```bibtex
@misc{li2025pointcnnperformantconvolutionnative,
  title         = {PointCNN++: Performant Convolution on Native Points},
  author        = {Lihan Li and Haofeng Zhong and Rui Bu and Mingchao Sun and Wenzheng Chen and Baoquan Chen and Yangyan Li},
  year          = {2025},
  eprint        = {2511.23227},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2511.23227}
}
```
