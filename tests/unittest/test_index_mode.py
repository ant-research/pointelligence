import random

import unittest

import torch

from internals.index_mode import index_mode


def generate_guaranteed_list(l, m):
    """
    Generates a list of length l with integers in [0, m) using a constructive
    method that guarantees strict frequency uniqueness for the top 2 elements.
    """
    # 1. Validation Constraints
    if m < 2:
        raise ValueError("m must be at least 2 to have a second most frequent item.")
    if l < 3:
        raise ValueError(
            "l must be at least 3 to satisfy strictly unique counts (e.g., 2 vs 1)."
        )

    # 2. Initial Random Distribution (Partition l items into m bins)
    counts = [0] * m
    for _ in range(l):
        counts[random.randrange(m)] += 1

    # 3. Mass Shifting Loop: Enforce Constraints
    # We loop until the sorted counts satisfy strict inequality for top elements.
    # Since we strictly move mass 'upwards' (to higher ranks), this is guaranteed to converge.
    while True:
        counts.sort(reverse=True)

        c0 = counts[0]
        c1 = counts[1]
        c2 = counts[2] if m > 2 else 0

        # Check constraints
        gap2_ok = c1 > c2  # 2nd > 3rd (or 0)
        gap1_ok = c0 > c1  # 1st > 2nd

        if gap1_ok and gap2_ok:
            break

        # Fix Violation 1: Ensure 2nd > 3rd
        if not gap2_ok:
            # Move mass from 3rd (or lower) to 2nd
            # If 3rd is 0 (rare edge case), take from 1st (swap logic handled in next iter)
            if counts[2] > 0:
                counts[1] += 1
                counts[2] -= 1
            else:
                # If c2 is 0, c1 is 0. Both empty. Take from top to create existence.
                counts[1] += 1
                counts[0] -= 1

        # Fix Violation 2: Ensure 1st > 2nd
        elif not gap1_ok:
            # Prefer taking from the "tail" (c2, c3...) to preserve c1's height
            taken_from_tail = False
            if m > 2:
                for i in range(2, m):
                    if counts[i] > 0:
                        counts[0] += 1
                        counts[i] -= 1
                        taken_from_tail = True
                        break

            # If tail is empty, we must take from c1 (widens gap by 2)
            if not taken_from_tail:
                counts[0] += 1
                counts[1] -= 1

    # 4. Map counts to actual numbers
    # We shuffle the values [0..m) so the "winner" isn't always the number 0.
    values = list(range(m))
    random.shuffle(values)

    result_list = []
    for i, count in enumerate(counts):
        val = values[i]
        result_list.extend([val] * count)

    random.shuffle(result_list)

    # The counts list was sorted, so values[0] is most frequent, values[1] is second.
    return result_list, values[0], values[1]


class TestIndexMode(unittest.TestCase):
    def setUp(self):
        seed = 9527
        torch.manual_seed(seed)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"

        x_high = 16
        self.target_size = 4
        max_length = 65536
        lengths = torch.randint(
            low=3, high=max_length, size=(self.target_size,), device=self.device
        )
        self.target_indices = torch.repeat_interleave(
            torch.arange(0, self.target_size, device=self.device), repeats=lengths
        )

        x_list = list()
        x_mode_list = list()
        x_mode_wo_0_list = list()
        l_list = list(lengths.cpu())
        for l in l_list:
            labels, a, b = generate_guaranteed_list(l, x_high)
            x_list.extend(labels)
            x_mode_list.append(a)
            x_mode_wo_0_list.append(a if a != 0 else b)

        self.x = torch.tensor(x_list, device=self.device)
        self.x_mode = torch.tensor(x_mode_list, device=self.device)
        self.x_mode_wo_0 = torch.tensor(x_mode_wo_0_list, device=self.device)

        perm = torch.randperm(self.target_indices.shape[0], device=self.device)
        self.target_indices = self.target_indices[perm]
        self.x = self.x[perm]

    def test_torch_vs_triton_vs_native(self):
        x_mode = index_mode(
            self.x, self.target_indices, self.target_size, ignore_value_zero=False
        )
        self.assertTrue(torch.equal(x_mode, self.x_mode))

        x_mode_wo_0 = index_mode(
            self.x, self.target_indices, self.target_size, ignore_value_zero=True
        )
        self.assertTrue(torch.equal(x_mode_wo_0, self.x_mode_wo_0))


if __name__ == "__main__":
    unittest.main()
