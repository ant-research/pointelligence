"""Structural contract for a triplet index (the (i, j, k) rulebook).

A point convolution's dispatch depends on three structural properties of its
triplet index — k-sortedness, exact-cover (fan-in-1 / fan-out-1), and uniform
per-tap segment length. These are **facts fixed at construction time**: the
triplet builder that produced (i, j, k) already knows them (or can prove them
with a host reduction, which is free inside the builder's
``@torch.compiler.disable`` region).

Historically ``_conv_forward`` *re-derived* them every forward via host syncs
(``is_sorted_cached(k).item()``, ``exact_cover_cached(i/j).item()``). Those
``.item()`` D2H syncs break ``torch.compile`` (Dynamo graph break) and cost a
CPU↔GPU round trip on every call. Carrying the facts as DATA on this object —
produced once by the builder, consumed by the forward without re-derivation —
removes both problems: the forward becomes a pure tensor-shape function that
traces cleanly under ``torch.compile(fullgraph=True)``.

The four fields mirror the ``TigIndex`` contract flags
(``assume_sorted`` / ``exact_cover_out`` / ``exact_cover_in`` /
``uniform_seg_len``) that the dispatch already consumes — this object is just
the typed carrier that ships them from builder to forward.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TripletContract:
    """Construction-time structural facts about a triplet index.

    Attributes:
      k_sorted: triplets are sorted ascending by kernel offset ``k``. Every
        production conv builder emits ``sort_by="k"`` → this is the conv-path
        invariant (the only ``sort_by="i"`` builder, ``max_pool3d``, feeds
        ``indexed_segment_reduce``, never conv). Carried explicitly so an
        external caller with genuinely-unsorted triplets can opt out
        (``k_sorted=False`` → the eager per-triplet path).
      exact_cover_out: every output row in ``[0, n_out)`` receives EXACTLY one
        triplet (fan-in-1 deconv forward) → routes the TIG FI1 plain-store
        forward. Default False (submanifold / strided convs).
      exact_cover_in: every input row in ``[0, n_in)`` appears in EXACTLY one
        triplet (fan-out-1 partition stem) → routes FI1 grad_input and, at
        large K, the dense-GEMM partition engine. Default False.
      uniform_seg_len: when the rulebook is k-sorted with EXACTLY this many
        triplets per kernel tap (the generative stamp construction), the TIG
        seg_offs is closed-form (no searchsorted). None = not uniform
        (radius-search builders are never uniform).
    """
    k_sorted: bool = True
    exact_cover_out: bool = False
    exact_cover_in: bool = False
    uniform_seg_len: Optional[int] = None

    @classmethod
    def submanifold(cls) -> "TripletContract":
        """The radius-search default: k-sorted, neither exact cover, not uniform.

        Holds for every submanifold / strided (non-disjoint) / upsample conv:
        the builder sorts by ``k`` and the triplet count ``T`` differs from both
        ``n`` and ``n_in`` (each output point has a variable, position-dependent
        neighbor count), so neither exact-cover proof can hold and there is no
        host reduction to pay.
        """
        return cls()
