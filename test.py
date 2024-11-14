import linalg
import math
import unittest


# these all pass
class TestVec3(unittest.TestCase):
    def test_add(self):
        v1 = linalg.Vec3(1, 2, 3)
        v2 = linalg.Vec3(4, 5, 6)
        self.assertEqual(v1 + v2, linalg.Vec3(5, 7, 9))

    def test_sub(self):
        v1 = linalg.Vec3(1, 2, 3)
        v2 = linalg.Vec3(4, 5, 6)
        self.assertEqual(v1 - v2, linalg.Vec3(-3, -3, -3))

    def test_mul(self):
        v1 = linalg.Vec3(1, 2, 3)
        self.assertEqual(v1 * 3, linalg.Vec3(3, 6, 9))

    def test_cross(self):
        v1 = linalg.Vec3(1, 2, 3)
        v2 = linalg.Vec3(4, 5, 6)
        self.assertEqual(v1.cross(v2), linalg.Vec3(-3, 6, -3))

    def test_dot(self):
        v1 = linalg.Vec3(1, 2, 3)
        v2 = linalg.Vec3(4, 5, 6)
        self.assertEqual(v1.dot(v2), 32)

    def test_length(self):
        v1 = linalg.Vec3(1, 2, 3)
        self.assertEqual(v1.length(), math.sqrt(14))

    def test_norm_length(self):
        v1 = linalg.Vec3(1, 2, 3).norm()
        self.assertEqual(v1.length(), 1)


# make these tests
class TestMatrix(unittest.TestCase):
    def setUp(self):
        self.mat = linalg.Matrix([1, 2, 3, 4, 5, 6, 7, 8, 9], rows=3)
        self.other = linalg.Matrix([4, 5, 6, 7, 8, 9, 10, 11, 12], rows=3)
        self.rectangular = linalg.Matrix([1, 2, 3, 4, 5, 6], rows=2)

    def test_init(self):
        self.assertEqual(self.mat.rows, 3)
        self.assertEqual(self.mat.cols, 3)
        self.assertEqual(self.mat.data, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def test_matmul(self):
        out  = self.mat @ self.other
        self.assertEqual(out.data, [[48, 54, 60], [111, 126, 141], [174, 198, 222]])

    def test_matmul_rectangle(self):
        out = self.rectangular @ self.other
        self.assertEqual(out.data, [[48, 54, 60], [111, 126, 141]])

    def test_matmul_lu(self):
        L, U = self.mat.lu_decomp()
        product = L @ U
        self.assertEqual(product.data, self.mat.data)

    def test_add(self):
        res = self.mat + self.other
        self.assertEqual(res.data, [[5, 7, 9], [11, 13, 15], [17, 19, 21]])

    def test_sub(self):
        res = self.mat + self.other
        self.assertEqual(res.data, [[5, 7, 9], [11, 13, 15], [17, 19, 21]])

    def test_scalar_mul(self):
        res = self.mat * 2
        self.assertEqual(res.data, [[2, 4, 6], [8, 10, 12], [14, 16, 18]])

    def test_transpose(self):
        res = self.mat.transpose()
        self.assertEqual(res.data, [[1, 4, 7], [2, 5, 8], [3, 6, 9]])

        res = self.rectangular.transpose()
        self.assertEqual(res.data, [[1, 4], [2, 5], [3, 6]])

    def test_eq(self):
        anotha = linalg.Matrix([1, 2, 3, 4, 5, 6, 7, 8, 9], rows=3)
        self.assertTrue(self.mat == anotha)

    def test_inverse(self):
        self.assertRaises(ValueError, self.mat.inverse)
        mat = linalg.Matrix([1, 2, 3, 4, 5, 6, 7, 2, 9], rows=3)
        exp_inv = linalg.Matrix(
            [-11 / 12, 1 / 3, 1 / 12, -1 / 6, 1 / 3, -1 / 6, 3 / 4, -1 / 3, 1 / 12],
            rows=3,
        )

        for i in range(mat.rows):
            for j in range(mat.cols):
                self.assertAlmostEqual(mat.inverse().data[i][j], exp_inv.data[i][j])

    def test_det(self):
        self.assertEqual(self.mat.det(), 0)
        self.assertEqual(self.other.det(), 0)

        mat = linalg.Matrix([1, 2, 3, 4, 5, 6, 7, 8, 3, 2, 3, 2, 3, 1, 7, 8], rows=4)
        self.assertEqual(mat.det(), -24)

    def test_lu_decomp(self):
        L, U = self.mat.lu_decomp()
        self.assertEqual(L.data, [[1, 0, 0], [4, 1, 0], [7, 2, 1]])
        self.assertEqual(U.data, [[1, 2, 3], [0, -3, -6], [0, 0, 0]])

    def test_lu_decomp2(self):
        mat = linalg.Matrix([1, 2, 3, 4, 5, 6, 7, 8, 3, 2, 3, 2, 3, 1, 7, 8], rows=4)
        L, U = mat.lu_decomp()
        exp_L = linalg.Matrix(
            [1, 0, 0, 0, 5, 1, 0, 0, 3, 1, 1, 0, 3, 5 / 4, 4, 1], rows=4
        )
        exp_U = linalg.Matrix(
            [1, 2, 3, 4, 0, -4, -8, -12, 0, 0, 2, 2, 0, 0, 0, 3], rows=4
        )
        self.assertEqual(L.data, exp_L.data)
        self.assertEqual(U.data, exp_U.data)

    def test_identity(self):
        I = linalg.Matrix.identity(3)
        self.assertEqual(I.data, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_elim_step(self):
        step = linalg.EliminationStep(0, 2, 4)
        self.mat.elim_step(step)
        self.assertEqual(self.mat.data, [[1, 2, 3], [4, 5, 6], [3, 0, -3]])

    def test_gaussian_elimination(self):
        out, _ = self.mat.gaussian_elimination()
        self.assertEqual(out.data, [[1, 2, 3], [0, -3, -6], [0, 0, 0]])

    def test_rref(self):
        out, _ = self.mat.gaussian_elimination(rref=True)
        self.assertEqual(out.data, [[1, 0, -1], [0, 1, 2], [0, 0, 0]])

    def test_indexing(self):
        self.assertEqual(self.mat[0], linalg.Matrix([1, 2, 3], rows=1))
        self.assertEqual(self.mat[1], linalg.Matrix([4, 5, 6], rows=1))
        self.assertEqual(self.mat[2], linalg.Matrix([7, 8, 9], rows=1))
        self.assertEqual(self.mat[0:2], linalg.Matrix([1, 2, 3, 4, 5, 6], rows=2))


if __name__ == "__main__":
    unittest.main()
