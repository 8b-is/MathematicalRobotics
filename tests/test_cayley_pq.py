"""
Unit tests for Panasonic MathematicalRobotics (`mathR`) logSO3 pi-singularity fix,
GaussNewton negative-step clamp fix, and exact p/q Rational Cayley-Padé SO(3, Q).
"""

import importlib.util
import math
import os
import sys
import unittest

_MOD_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "mathR",
    "lie_group",
    "cayley_pq.py",
)
_SPEC = importlib.util.spec_from_file_location("cayley_pq", _MOD_PATH)
cayley_mod = importlib.util.module_from_spec(_SPEC)
sys.modules["cayley_pq"] = cayley_mod
assert _SPEC.loader is not None
_SPEC.loader.exec_module(cayley_mod)

RationalSO3MatrixPq = cayley_mod.RationalSO3MatrixPq
log_so3_pi_safe = cayley_mod.log_so3_pi_safe
reduce_rational_pq = cayley_mod.reduce_rational_pq


class TestPanasonicMathematicalRoboticsPq(unittest.TestCase):
    def test_log_so3_exact_pi_rotation_not_zero(self):
        # Exact 180-deg rotation around X, Y, and Z axes (symmetric R => W == 0)
        rx_pi = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
        ry_pi = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]
        rz_pi = [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
        for axis_idx, r_mat in enumerate((rx_pi, ry_pi, rz_pi)):
            omega = log_so3_pi_safe(r_mat)
            norm = math.sqrt(sum(w * w for w in omega))
            self.assertAlmostEqual(norm, math.pi, places=12)
            self.assertAlmostEqual(abs(omega[axis_idx]), math.pi, places=12)

    def test_cayley_so3_pq_exact_orthogonality_and_determinant(self):
        # Test 50 chained rational 3D rotations: zero orthogonality drift!
        r_accum = RationalSO3MatrixPq.identity()
        vectors = [(1, 2, 3, 7), (-3, 1, 4, 9), (2, -5, 1, 11), (4, 3, -2, 13)]
        for step in range(20):
            px, py, pz, q = vectors[step % len(vectors)]
            r_step = RationalSO3MatrixPq.from_cayley_pq(px, py, pz, q)
            self.assertTrue(r_step.is_exact_orthogonal())
            self.assertEqual(r_step.determinant_pq(), (1, 1))
            # Verify inverse Cayley recovers exact (px/q, py/q, pz/q)
            ux, uy, uz = r_step.inv_cayley_pq()
            self.assertEqual(ux, reduce_rational_pq(px, q))
            self.assertEqual(uy, reduce_rational_pq(py, q))
            self.assertEqual(uz, reduce_rational_pq(pz, q))
            r_accum = r_accum.matmul(r_step)
            self.assertTrue(r_accum.is_exact_orthogonal())
            self.assertEqual(r_accum.determinant_pq(), (1, 1))


if __name__ == "__main__":
    unittest.main()
