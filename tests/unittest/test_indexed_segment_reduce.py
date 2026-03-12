import sys
import unittest

import torch

import torch._dynamo

# Import your module
from sparse_engines.ops import indexed_segment_reduce

# Increase cache limit
torch._dynamo.config.cache_size_limit = 256


class TestIndexedSegmentReduceExtensive(unittest.TestCase):
    def setUp(self):
        # Set seed for reproducibility
        torch.manual_seed(42)
        self.device = torch.device("cuda")

    def _assert_close_with_detail(
            self, actual, expected, atol=1e-4, rtol=1e-4, info=""
    ):
        try:
            torch.testing.assert_close(
                actual,
                expected,
                atol=atol,
                rtol=rtol,
                check_device=False,
                check_dtype=False,
            )
        except AssertionError:
            diff = (actual - expected).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            max_val = expected.abs().max().item()
            flat_idx = diff.argmax().item()
            val_act = actual.flatten()[flat_idx].item()
            val_exp = expected.flatten()[flat_idx].item()
            msg = (
                f"\n{'='*60}\nFAILURE: {info}\n{'-'*60}\nMax Diff:  {max_diff:.2e}\nMean Diff: {mean_diff:.2e}\n"
                f"Max Val:   {max_val:.2e}\n  Actual:  {val_act}\n  Expect:  {val_exp}\n{'='*60}\n"
            )
            raise AssertionError(msg) from None

    def _run_comparison(self, T, C, K, reduce):
        # 1. Setup Lengths
        raw_lengths = torch.randint(1, 15, (K,), device=self.device)
        raw_lengths[0] = 50
        L = raw_lengths.sum().item()
        lengths = raw_lengths

        # 2. Setup Indices (Unique)
        # We use randperm to ensure no index is repeated. This prevents the case where
        # the same input element (max) appears twice in a segment, which causes
        # the optimized kernel (recompute) to double-count gradients vs the reference.
        if L <= T:
            indices = torch.randperm(T, device=self.device)[:L]
        else:
            # If L > T, we can't be unique globally.
            # We fallback to randint, but this might trigger the duplicate max issue.
            # For this test config (T=1000, L~400), we represent safe territory.
            indices = torch.randint(0, T, (L,), device=self.device)

        # 3. Setup X (Unique Values)
        x = torch.randn(T, C, device=self.device, dtype=torch.float32)

        # Add a small perturbation based on index to ensure every value in X is strictly unique.
        # This prevents value collisions (e.g. x[i] == x[j]) which also causes gradient mismatch.
        x = x + torch.arange(T, device=self.device).unsqueeze(1) * 1e-5

        if reduce in ["min", "max"]:
            x = x * 100.0
        x.requires_grad = True

        # --- Reference Pass ---
        x_ref = x.clone().detach().requires_grad_(True)
        gathered_ref = x_ref[indices]
        # unsafe=True is fine as we trust bounds here
        y_ref = torch.segment_reduce(
            gathered_ref, reduce=reduce, lengths=lengths, unsafe=True
        )
        loss_ref = y_ref.sum()
        loss_ref.backward()

        # --- Triton Pass ---
        x_tri = x.clone().detach().requires_grad_(True)
        y_tri = indexed_segment_reduce(x_tri, reduce, indices, lengths=lengths)
        loss_tri = y_tri.sum()
        loss_tri.backward()

        # --- Validation ---
        self._assert_close_with_detail(
            y_tri, y_ref, atol=1e-4, rtol=1e-4, info=f"Fwd {reduce} C={C}"
        )
        self._assert_close_with_detail(
            x_tri.grad, x_ref.grad, atol=1e-3, rtol=1e-3, info=f"Bwd {reduce} C={C}"
        )

    def test_correctness_small_and_weird_C(self):
        # T=1000, K=50 -> L approx 400. Fits in T (Unique indices possible)
        for c in [1, 3, 4, 7, 8, 13, 16]:
            for reduce in ["sum", "mean", "max", "min"]:
                self._run_comparison(T=1000, C=c, K=50, reduce=reduce)

    def test_correctness_large_C(self):
        for c in [64, 128]:
            for reduce in ["sum", "mean", "max", "min"]:
                self._run_comparison(T=1000, C=c, K=50, reduce=reduce)


class BenchmarkSuite:
    def __init__(self):
        self.device = torch.device("cuda")

    def benchmark_config(self, T, C, K, avg_len, reduce="sum"):
        compiled_op = torch.compile(indexed_segment_reduce)

        lengths = torch.randint(
            max(1, avg_len - 5), avg_len + 5, (K,), device=self.device
        )
        L = lengths.sum().item()

        # For benchmarking, collisions are fine (performance is the same).
        # We revert to standard randint to simulate realistic dense/sparse mix.
        indices = torch.randint(0, T, (L,), device=self.device)

        x = torch.randn(T, C, device=self.device).requires_grad_(True)
        grad_y = torch.randn(K, C, device=self.device)

        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)

        # Warmup
        for _ in range(3):
            indexed_segment_reduce(x, reduce, indices, lengths=lengths)
            compiled_op(x, reduce, indices, lengths=lengths)

        iterations = 20

        # 1. Reference
        torch.cuda.synchronize()
        start_ev.record()
        for _ in range(iterations):
            gathered = x[indices]
            y = torch.segment_reduce(gathered, reduce, lengths=lengths)
        end_ev.record()
        torch.cuda.synchronize()
        fwd_ref = start_ev.elapsed_time(end_ev) / iterations

        accum = 0.0
        for _ in range(iterations):
            gathered = x[indices]
            y = torch.segment_reduce(gathered, reduce, lengths=lengths)
            torch.cuda.synchronize()
            start_ev.record()
            y.backward(grad_y)
            end_ev.record()
            torch.cuda.synchronize()
            accum += start_ev.elapsed_time(end_ev)
            x.grad.zero_()
        bwd_ref = accum / iterations

        # 2. Triton Eager
        torch.cuda.synchronize()
        start_ev.record()
        for _ in range(iterations):
            y = indexed_segment_reduce(x, reduce, indices, lengths=lengths)
        end_ev.record()
        torch.cuda.synchronize()
        fwd_tri = start_ev.elapsed_time(end_ev) / iterations

        accum = 0.0
        for _ in range(iterations):
            y = indexed_segment_reduce(x, reduce, indices, lengths=lengths)
            torch.cuda.synchronize()
            start_ev.record()
            y.backward(grad_y)
            end_ev.record()
            torch.cuda.synchronize()
            accum += start_ev.elapsed_time(end_ev)
            x.grad.zero_()
        bwd_tri = accum / iterations

        # 3. Triton Compiled
        torch.cuda.synchronize()
        start_ev.record()
        for _ in range(iterations):
            y = compiled_op(x, reduce, indices, lengths=lengths)
        end_ev.record()
        torch.cuda.synchronize()
        fwd_tri_c = start_ev.elapsed_time(end_ev) / iterations

        accum = 0.0
        for _ in range(iterations):
            y = compiled_op(x, reduce, indices, lengths=lengths)
            torch.cuda.synchronize()
            start_ev.record()
            y.backward(grad_y)
            end_ev.record()
            torch.cuda.synchronize()
            accum += start_ev.elapsed_time(end_ev)
            x.grad.zero_()
        bwd_tri_c = accum / iterations

        return {
            "T": T,
            "C": C,
            "K": K,
            "L": L,
            "Mode": reduce,
            "Ref Fwd": fwd_ref,
            "Ref Bwd": bwd_ref,
            "Tri Fwd": fwd_tri,
            "Tri Bwd": bwd_tri,
            "TriC Fwd": fwd_tri_c,
            "TriC Bwd": bwd_tri_c,
        }

    def run(self):
        print("\n" + "=" * 160)
        print("PERFORMANCE BENCHMARK: Ref vs Triton vs Triton(Compiled)".center(160))
        print("=" * 160)

        header = (
            f"{'T':<7} {'C':<5} {'K':<10} {'L':<10} {'Mode':<5} | "
            f"{'Ref Fwd':<9} {'Tri Fwd':<9} {'TriC Fwd':<9} | "
            f"{'Ref Bwd':<9} {'Tri Bwd':<9} {'TriC Bwd':<9} | "
            f"{'Ref/TriC Spd (F)':<18} {'Ref/TriC Spd (B)':<18}"
        )
        print(header)
        print("-" * 160)

        Cs_to_test = [1, 3, 4, 7, 8, 13, 16, 32, 64, 128, 256]
        modes = ["sum", "mean", "max", "min"]

        for reduce in modes:
            print(f"--- Group: {reduce.upper()} ---")
            for c in Cs_to_test:
                res = self.benchmark_config(
                    T=50000,
                    C=c,
                    K=250000 * (10 if c < 32 else 1),
                    avg_len=20,
                    reduce=reduce,
                )
                self._print_res(res)
            print("")

    def _print_res(self, r):
        t_fwd_c = max(r["TriC Fwd"], 1e-6)
        t_bwd_c = max(r["TriC Bwd"], 1e-6)

        line = (
            f"{r['T']:<7} {r['C']:<5} {r['K']:<10} {r['L']:<10} {r['Mode']:<5} | "
            f"{r['Ref Fwd']:<9.3f} {r['Tri Fwd']:<9.3f} {r['TriC Fwd']:<9.3f} | "
            f"{r['Ref Bwd']:<9.3f} {r['Tri Bwd']:<9.3f} {r['TriC Bwd']:<9.3f} | "
            f"{r['Ref Fwd']/t_fwd_c:<18.2f} {r['Ref Bwd']/t_bwd_c:<18.2f}"
        )
        print(line)


if __name__ == "__main__":
    print("Running Correctness Tests...")
    suite = unittest.TestLoader().loadTestsFromTestCase(
        TestIndexedSegmentReduceExtensive
    )
    runner = unittest.TextTestRunner(verbosity=1, stream=sys.stdout)
    result = runner.run(suite)
    sys.stdout.flush()
    if result.wasSuccessful():
        bench = BenchmarkSuite()
        bench.run()