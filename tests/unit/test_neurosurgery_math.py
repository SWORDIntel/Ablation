import unittest

import torch

from aegis_lab.editing.neurosurgery.math_ops import (
    apply_constrained_directional_surgery,
    contrast_direction,
    preservation_basis,
)


class TestNeurosurgeryMath(unittest.TestCase):
    def test_contrast_direction_normalized(self):
        keep = torch.tensor([1.0, 0.0, 0.0])
        drop = torch.tensor([1.0, 3.0, 0.0])
        direction, separation = contrast_direction(keep, drop)
        self.assertAlmostEqual(float(direction.norm()), 1.0, places=6)
        self.assertAlmostEqual(separation, 3.0, places=6)

    def test_preservation_basis_is_orthonormal(self):
        q = preservation_basis(torch.randn(8, 6), 3)
        self.assertTrue(torch.allclose(q.T @ q, torch.eye(3), atol=1e-5))

    def test_constrained_surgery_preserves_norms_and_basis(self):
        torch.manual_seed(1)
        weight = torch.randn(6, 6)
        direction = torch.randn(6)
        basis = preservation_basis(torch.randn(12, 6), 2)
        edited = apply_constrained_directional_surgery(weight, direction, 0.7, basis, True)
        self.assertTrue(torch.allclose(weight.norm(dim=1), edited.float().norm(dim=1), atol=1e-5, rtol=1e-5))
        delta = edited.float() - weight.float()
        self.assertLess(torch.max(torch.abs(basis.T @ delta)).item(), 1e-4)


if __name__ == "__main__":
    unittest.main()
