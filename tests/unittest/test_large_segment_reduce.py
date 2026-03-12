import sys
import unittest

import torch

from sparse_engines.ops import large_segment_reduce


class TestLargeSegmentReduceCorrectness(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)

        self.configs = [
            {"name": "Standard_Small", "N": 1000, "K": 100, "Cs": [32, 64]},
            {"name": "Target_TwoSegs", "N": 20000, "K": 2, "Cs": [64, 128]},
            {"name": "Target_SingleSeg", "N": 10000, "K": 1, "Cs": [32]},
            {"name": "Edge_TinyC", "N": 5000, "K": 5, "Cs": [1, 3]},
            {"name": "Edge_PrimeC", "N": 5000, "K": 5, "Cs": [5, 127]},
        ]

    def _generate_data(self, N: int, K: int, C: int):
        if K == 1:
            lengths = torch.tensor([N], device=self.device, dtype=torch.long)
        else:
            probs = torch.rand(K, device=self.device)
            lengths = (probs / probs.sum() * N).long()
            lengths[-1] += N - lengths.sum()
            while (lengths == 0).any():
                zero_indices = (lengths == 0).nonzero(as_tuple=True)[0]
                max_idx = torch.argmax(lengths)
                if lengths[max_idx] <= 1:
                    lengths[zero_indices] = 1
                    break
                lengths[zero_indices[0]] += 1
                lengths[max_idx] -= 1
        N = lengths.sum().item()
        x = torch.randn(N, C, device=self.device, requires_grad=True)
        return x, lengths

    def _run_comparison(
        self, N: int, K: int, C: int, reduce_mode: str, atol: float = 1e-4, rtol=1e-4
    ):
        x, lengths = self._generate_data(N, K, C)
        x_ref = x.detach().clone().requires_grad_(True)

        out_triton = large_segment_reduce(x, reduce=reduce_mode, lengths=lengths)
        out_ref = torch.segment_reduce(
            x_ref, reduce=reduce_mode, lengths=lengths, unsafe=False
        )

        is_close = torch.allclose(out_triton, out_ref, atol=atol, rtol=rtol)
        max_diff = (out_triton - out_ref).abs().max().item()

        self.assertTrue(
            is_close,
            msg=f"FWD Fail | Mode: {reduce_mode} | Shape: [N={N}, K={K}, C={C}] | MaxDiff: {max_diff:.2e}",
        )

        grad_output = torch.randn_like(out_triton)
        out_triton.backward(grad_output)
        out_ref.backward(grad_output)

        is_grad_close = torch.allclose(x.grad, x_ref.grad, atol=atol, rtol=rtol)
        max_grad_diff = (x.grad - x_ref.grad).abs().max().item()

        self.assertTrue(
            is_grad_close,
            msg=f"BWD Fail | Mode: {reduce_mode} | Shape: [N={N}, K={K}, C={C}] | MaxDiff: {max_grad_diff:.2e}",
        )

    def test_mode_sum(self):
        for config in self.configs:
            for C in config["Cs"]:
                self._run_comparison(
                    config["N"], config["K"], C, "sum", atol=1e-3, rtol=1e-3
                )

    def test_mode_mean(self):
        for config in self.configs:
            for C in config["Cs"]:
                self._run_comparison(
                    config["N"], config["K"], C, "mean", atol=1e-3, rtol=1e-3
                )

    def test_mode_max(self):
        for config in self.configs:
            for C in config["Cs"]:
                self._run_comparison(
                    config["N"], config["K"], C, "max", atol=1e-5, rtol=1e-5
                )

    def test_mode_min(self):
        for config in self.configs:
            for C in config["Cs"]:
                self._run_comparison(
                    config["N"], config["K"], C, "min", atol=1e-5, rtol=1e-5
                )


class BenchmarkLargeSegmentReduce(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)
        if self.device.type != "cuda":
            self.skipTest("CUDA not available")

        # Config: 1 Million Elements
        # Testing K=8, 16, 32 to see performance across different segment sizes
        self.N = 1_000_000
        self.Ks = [8, 16, 32]
        self.c_values = [1, 3, 4, 7, 13, 16, 32, 64, 128, 256]

        self.compiled_op = torch.compile(large_segment_reduce)

    def _measure_perf(self, func, iters=50, warmup=10):
        for _ in range(warmup):
            func()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            func()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    def _run_benchmark_for_mode(self, mode):
        # Iterate over different K values
        for K in self.Ks:
            print(f"\n{'='*100}")
            print(
                f"BENCHMARK: MODE = {mode.upper()} | N={self.N}, K={K} | (Target: Few Huge Segments)"
            )
            print(f"{'='*100}")

            # New Columnar Format
            header = (
                f"{'C':<4} | "
                f"{'PyTorch(ms)':<11} | {'Triton(ms)':<11} | {'Speedup':<8} | "
                f"{'PyTorch Bwd':<11} | {'Triton Bwd':<11} | {'Speedup':<8}"
            )
            print(header)
            print("-" * len(header))

            for C in self.c_values:
                # Generate Data based on current K
                lengths = torch.tensor([self.N // K] * K, device=self.device)
                lengths[-1] += self.N - lengths.sum()
                x = torch.randn(self.N, C, device=self.device, requires_grad=True)
                grad_out = torch.randn(K, C, device=self.device)

                # PyTorch
                t_fwd = lambda: torch.segment_reduce(
                    x, reduce=mode, lengths=lengths, unsafe=False
                )

                def t_bwd():
                    out = torch.segment_reduce(
                        x, reduce=mode, lengths=lengths, unsafe=False
                    )
                    out.backward(grad_out, retain_graph=True)
                    x.grad.zero_()

                # Triton (Compiled)
                c_op = torch.compile(large_segment_reduce)
                c_fwd = lambda: c_op(x, reduce=mode, lengths=lengths)

                def c_bwd():
                    out = c_op(x, reduce=mode, lengths=lengths)
                    out.backward(grad_out, retain_graph=True)
                    x.grad.zero_()

                # Measure
                ms_tf = self._measure_perf(t_fwd)
                ms_cf = self._measure_perf(c_fwd)
                ms_tb = self._measure_perf(t_bwd)
                ms_cb = self._measure_perf(c_bwd)

                # Speedups
                su_fwd = ms_tf / ms_cf if ms_cf > 1e-6 else 0
                su_bwd = ms_tb / ms_cb if ms_cb > 1e-6 else 0

                print(
                    f"{C:<4} | "
                    f"{ms_tf:<11.4f} | {ms_cf:<11.4f} | {f'{su_fwd:.2f}x':<8} | "
                    f"{ms_tb:<11.4f} | {ms_cb:<11.4f} | {f'{su_bwd:.2f}x':<8}"
                )

            sys.stdout.flush()

    def test_benchmark_01_sum(self):
        self._run_benchmark_for_mode("sum")

    def test_benchmark_02_mean(self):
        self._run_benchmark_for_mode("mean")

    def test_benchmark_03_max(self):
        self._run_benchmark_for_mode("max")

    def test_benchmark_04_min(self):
        self._run_benchmark_for_mode("min")


if __name__ == "__main__":
    # 1. Run Correctness Tests
    print("\n" + "=" * 30 + " RUNNING CORRECTNESS TESTS " + "=" * 30)
    suite_correctness = unittest.TestLoader().loadTestsFromTestCase(
        TestLargeSegmentReduceCorrectness
    )
    runner_correctness = unittest.TextTestRunner(verbosity=2)
    result_correctness = runner_correctness.run(suite_correctness)

    # 2. Run Benchmarks if Correctness passed
    if result_correctness.wasSuccessful():
        print("\n" + "=" * 30 + " RUNNING BENCHMARKS " + "=" * 30)
        suite_benchmark = unittest.TestLoader().loadTestsFromTestCase(
            BenchmarkLargeSegmentReduce
        )
        runner_benchmark = unittest.TextTestRunner(verbosity=0)
        runner_benchmark.run(suite_benchmark)
    else:
        print("\n" + "!" * 80)
        print("SKIPPING BENCHMARKS: Correctness tests failed.")
        print("!" * 80)
        sys.exit(1)
