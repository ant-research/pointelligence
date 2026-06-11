"""Surgical third-party warning filters for PNT / PT-v3 training runs.

Imported as the very first statement of ``pointcept/__init__.py`` so the
filters are registered before spconv / torch / numpy are imported, in
*every* process: the launcher, each ``mp.spawn`` DDP rank, and each
spawn-based DataLoader worker. (Spawn children get a fresh interpreter
that re-imports ``pointcept`` via the dataset/model classes, so a filter
registered only in the parent would not reach them — this is why eight
import-time warnings ballooned to ~8000 lines per run.)

Scope discipline — every entry is anchored to a specific third-party
``module=`` or ``message=`` AND ``category``, never a blanket
``ignore::FutureWarning``. A genuinely new warning raised by *our own*
code (pointcept / pnt_* / overlays) is therefore still surfaced. We pin
PyTorch at the current version or newer (project directive: never
downgrade torch), so the deprecation-shaped warnings below describe APIs
inside vendored libraries we don't edit (spconv) — they will never turn
into errors we must act on, so silencing them loses no signal.

Each filter records WHAT it silences, WHERE it comes from, and WHY it is
safe. Drop an entry the day its upstream library is bumped past the
deprecation.
"""

import warnings


def _install() -> None:
    # 1. spconv 2.3.x defines its autograd Functions with the pre-2.4
    #    `torch.cuda.amp.custom_fwd/custom_bwd` decorators. PyTorch >=2.4
    #    emits a FutureWarning pointing at `torch.amp.custom_fwd(...,
    #    device_type='cuda')`. spconv is a vendored binary dependency we
    #    do not patch; the decorator still works. This single warning is
    #    ~99.9% of all training-log warning volume (re-emitted by every
    #    spawned DDP rank + DataLoader worker on each spconv import).
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module=r"spconv\..*",
    )

    # 2. spconv's fused-impl dispatch logs a UserWarning when an AMP run
    #    hands it half-precision input against fp32 weights ("Mismatch
    #    dtype between input and weight ... Cannot dispatch to fused
    #    implementation"). Expected under our fp16 AMP recipe; spconv
    #    falls back to the correct (slower) path. Not our code to fix.
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r"Mismatch dtype between input and weight",
    )

    # 3. torch's legacy `torch.cuda.FloatTensor(...)`-style constructors
    #    are deprecated in favour of `torch.tensor(..., device='cuda')`.
    #    Emitted a handful of times from inside torch/third-party init,
    #    not from pointcept. Message-anchored (the issuing frame is a
    #    torch C-extension, so `module=` is unreliable here).
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r"The torch\.cuda\.\*DtypeTensor constructors are no longer recommended",
    )

    # 4. torch DataLoader pinning path calls `Tensor.pin_memory(device=...)`
    #    whose `device` argument is deprecated in current torch. Internal
    #    to the data pipeline, not our call site.
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=r"The argument 'device' of Tensor\.pin_memory\(\) is deprecated",
    )

    # 5. Python's multiprocessing resource_tracker prints a "leaked
    #    semaphore objects to clean up at shutdown" UserWarning when
    #    spawn workers are torn down at process exit. Cosmetic
    #    shutdown-time noise; the OS reclaims the semaphores. Anchored to
    #    the resource_tracker module so only its shutdown chatter is hit.
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"multiprocessing\.resource_tracker",
    )

    # 6. numpy emits "Mean of empty slice" / "invalid value encountered in
    #    double_scalars" from the mIoU averaging when a semantic class has
    #    zero points in an eval split (per-class IoU = 0/0 -> nan, then
    #    nanmean). Benign and expected for sparse-class eval; the
    #    evaluator already handles the nan. Message-anchored so only these
    #    two metric-math cases are silenced, not all numpy RuntimeWarnings.
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message=r"Mean of empty slice",
    )
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message=r"invalid value encountered in double_scalars",
    )


_install()
