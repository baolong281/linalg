from collections import deque
from typing import List, Tuple


class Vec3:
    """
    3D Vector class
    """

    def __init__(self, x, y, z):
        """
        Constructor takes data
        """
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        """
        Add two 3d vectors
        """
        if isinstance(other, Vec3):
            return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)
        elif isinstance(other, int) or isinstance(other, float):
            return Vec3(self.x + other, self.y + other, self.z + other)
        else:
            raise TypeError("Can only add a Vec3 or a scalar")

    def __sub__(self, other):
        """
        Subtract two 3d vectors
        """
        if isinstance(other, Vec3):
            return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)
        elif isinstance(other, int) or isinstance(other, float):
            return Vec3(self.x - other, self.y - other, self.z - other)
        else:
            raise TypeError("Can only subtract a Vec3 or a scalar")

    def __mul__(self, other):
        """
        Scalar multiplication for a vector
        """
        if isinstance(other, int) or isinstance(other, float):
            return Vec3(self.x * other, self.y * other, self.z * other)
        else:
            raise TypeError("Can only multiply by a Vec3 or a scalar")

    def __str__(self):
        """
        Print vector
        """
        return f"Vec3({self.x}, {self.y}, {self.z})"

    def __eq__(self, other):
        """
        Compare data from two vectors
        """
        if not isinstance(other, Vec3):
            raise TypeError("Can only compare Vec3 with another Vec3")
        return self.x == other.x and self.y == other.y and self.z == other.z

    def dot(self, other):
        """
        Dot product for two vectors
        """
        if not isinstance(other, Vec3):
            raise TypeError("Can only dot a Vec3 with another Vec3")
        return self.x * other.x + self.y * other.y + self.z * other.z

    def length(self):
        return (self.x**2 + self.y**2 + self.z**2) ** 0.5

    def cross(self, other):
        if not isinstance(other, Vec3):
            raise TypeError("Can only cross a Vec3 with another Vec3")
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def norm(self):
        len = self.length()
        return Vec3(self.x / len, self.y / len, self.z / len)


class EliminationStep:
    def __init__(self, pivot_row, other_row, multiplier, div=False):
        self.multiplier = multiplier
        self.pivot_row = pivot_row
        self.other_row = other_row
        self.div = div

    def __str__(self):
        return f"EliminationStep(pivot={self.pivot_row}, target={self.other_row}, c={self.multiplier})"


class DecompStep:
    def __init__(self, row, col, value):
        self.row = row
        self.col = col
        self.value = value

    def __str__(self):
        return f"DecompStep(row={self.row}, col={self.col}, value={self.value})"


class Matrix:
    def __init__(self, data: List[int | float], rows: int = 0):
        """
        Initialize a matrix from a 1D list of data
        """
        if rows == 0:
            raise ValueError("Must specify rows")

        if len(data) % rows != 0:
            raise ValueError("Data length must be a multiple of rows")

        self.rows = rows
        self.cols = len(data) // rows

        matrix_data = [[0] * self.cols for _ in range(self.rows)]
        for i, n in enumerate(data):
            r = i // self.cols
            c = i % self.cols
            matrix_data[r][c] = n

        self.data = matrix_data

    def __str__(self):
        mat_data = "\n".join([f"{row}" for row in self.data])
        return f"Matrix({mat_data}, rows={self.rows})"

    def __getitem__(self, key):
        """
        Indexing for matrix like numpy
        """
        def adjust_slice(s, max_val):
            if s.start is not None and s.start < 0:
                raise ValueError("Negative indexing not supported")
            if s.stop is not None and s.stop > max_val:
                raise ValueError("Index out of range")
            start = s.start if s.start is not None else 0
            stop = s.stop if s.stop is not None else max_val
            return slice(start, stop)

        if isinstance(key, tuple) and isinstance(key[0], int) and isinstance(key[1], int):
            return Matrix([self.data[key[0]][key[1]]], rows=1)
        elif isinstance(key, int):
            return Matrix(self.data[key], rows=1)
        elif isinstance(key, slice):
            key = adjust_slice(key, self.rows)
            return Matrix([n for row in self.data[key] for n in row], rows=key.stop - key.start)
        elif isinstance(key, tuple) and isinstance(key[0], slice) and isinstance(key[1], slice):
            rslice = adjust_slice(key[0], self.rows)
            cslice = adjust_slice(key[1], self.cols)
            return Matrix([n for row in self.data[rslice] for n in row[cslice]], rows=rslice.stop - rslice.start)
        elif isinstance(key, tuple) and isinstance(key[0], slice) and isinstance(key[1], int):
            rslice = adjust_slice(key[0], self.rows)
            return Matrix([row[key[1]] for row in self.data[rslice]], rows=rslice.stop - rslice.start)
        elif isinstance(key, tuple) and isinstance(key[0], int) and isinstance(key[1], slice):
            cslice = adjust_slice(key[1], self.cols)
            return Matrix([n for n in self.data[key[0]][cslice]], rows=1)
        else:
            raise TypeError("Invalid index type")

    def pad_bottom(self, rows):
        new_data = [n for row in self.data for n in row]
        new_data.extend([0] * (self.cols * rows))
        out = Matrix(new_data, rows=self.rows + rows)
        return out

    def pad_right(self, cols):
        new_data = [n for row in self.data for n in row + ([0] * cols)]
        out = Matrix(new_data, rows=self.rows)
        return out

    def __padding_strassen(self, other):
        curr_mat = self.copy()
        other_mat = other.copy()
        # if the matrix is not square, we need to pad it with zeros
        if curr_mat.rows % 2 != 0:
            curr_mat = curr_mat.pad_bottom(1)
        if other.rows % 2 != 0:
            other_mat = other_mat.pad_bottom(1)

        if curr_mat.cols % 2 != 0:
            curr_mat = curr_mat.pad_right(1)
        if other.cols % 2 != 0:
            other_mat = other_mat.pad_right(1)

        if curr_mat.rows < other_mat.rows:
            curr_mat = curr_mat.pad_bottom(other_mat.rows - curr_mat.rows)
        if other_mat.rows < curr_mat.rows:
            other_mat = other_mat.pad_bottom(curr_mat.rows - other_mat.rows)
        if curr_mat.cols < other_mat.cols:
            curr_mat = curr_mat.pad_right(other_mat.cols - curr_mat.cols)
        if other_mat.cols < curr_mat.cols:
            other_mat = other_mat.pad_right(curr_mat.cols - other_mat.cols)

        return curr_mat, other_mat


    def __matmul__(self, other):
        """
        Matrix multiplication with @ operator
        i.e. mat @ mat2
        Naive implementation
        """

        if self.cols != other.rows:
            raise ValueError("Matrix dimensions must be compatible")

        curr_mat, other_mat = self.__padding_strassen(other)

        def merge_quadrants(a, b, c, d):
            m = a.rows
            n = a.cols

            out = Matrix([0] * (4 * m * n), rows=2 * m)

            for i in range(m):
                for j in range(n):
                    out.data[i][j] = a.data[i][j]
                    out.data[i][j + n] = b.data[i][j]
                    out.data[i + m][j] = c.data[i][j]
                    out.data[i + m][j + n] = d.data[i][j]

            return out

        def split(mat):
            half_col = mat.cols // 2
            half_row = mat.rows // 2
            a = mat[:half_row, :half_col]
            b = mat[:half_row, half_col:]
            c = mat[half_row:, :half_col]
            d = mat[half_row:, half_col:]
            return a, b, c, d

        def multiply_2x2(a, b):
            return Matrix([
                a.data[0][0] * b.data[0][0] + a.data[0][1] * b.data[1][0],
                a.data[0][0] * b.data[0][1] + a.data[0][1] * b.data[1][1],
                a.data[1][0] * b.data[0][0] + a.data[1][1] * b.data[1][0],
                a.data[1][0] * b.data[0][1] + a.data[1][1] * b.data[1][1],
            ], rows=2)

        def strassen(mat1, mat2):
            if mat1.rows == 2 and mat2.rows == 2:
                return multiply_2x2(mat1, mat2)
            if mat1.rows == 2 and mat2.rows == 1:
                return Matrix([mat1.data[0][0] * mat2.data[0][0]], rows=1)


            a, b, c, d = split(mat1)
            e, f, g, h = split(mat2)

            topleft = strassen(a, e) + strassen(b, g)
            topright = strassen(a, f) + strassen(b, h)
            bottomleft = strassen(c, e) + strassen(d, g)
            bottomright = strassen(c, f) + strassen(d, h)

            return merge_quadrants(topleft, topright, bottomleft, bottomright)
            
        out = strassen(curr_mat, other_mat)
        # we have to crop the matrix to the correct size cause the padding
        return out[:self.rows, :other.cols]

    def __eq__(self, other):
        """
        Asserts that two matrices are equal
        """
        if not isinstance(other, Matrix):
            raise TypeError("Can only compare Matrix with another Matrix")
        if self.rows != other.rows or self.cols != other.cols:
            return False
        return all(
            self.data[i][j] == other.data[i][j]
            for i in range(self.rows)
            for j in range(self.cols)
        )

    def __add__(self, other):
        """
        Add two matrices, must be the same size
        """
        if not isinstance(other, Matrix):
            raise TypeError("Can only add a Matrix to another Matrix")
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must be the same size")

        out = Matrix([0] * (self.rows * self.cols), rows=self.rows)

        for i in range(self.rows):
            for j in range(self.cols):
                out.data[i][j] = self.data[i][j] + other.data[i][j]

        return out

    def __sub__(self, other):
        """
        Subtract two matrices by adding the negative of the other matrix
        """
        if not isinstance(other, Matrix):
            raise TypeError("Can only subtract a Matrix from another Matrix")
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must be the same size")

        return self + (other * -1)

    def __mul__(self, other):
        """
        Scalar multiplication for a matrix
        """
        if not isinstance(other, int) and not isinstance(other, float):
            raise TypeError("Can only multiply a Matrix by a scalar")

        out = self.copy()

        for i in range(self.rows):
            for j in range(self.cols):
                out.data[i][j] = self.data[i][j] * other

        return out

    def transpose(self):
        """
        Return the transpose of the matrix
        """
        out = Matrix([0] * (self.rows * self.cols), rows=self.cols)

        for i in range(self.rows):
            for j in range(self.cols):
                out.data[j][i] = self.data[i][j]

        return out

    def gaussian_elimination(
        self, rref=False
    ) -> Tuple[any, List[EliminationStep | DecompStep]]:
        """
        Do gaussian elimination and returna  new matrix and a list of decomp steps.
        Does not return RREF.
        """
        steps = []
        q = deque()

        out = Matrix([n for row in self.data for n in row], rows=self.rows)

        for pivot in range(out.cols):
            val = out.data[pivot][pivot]

            for row in range(pivot + 1, out.rows):
                factor = out.data[row][pivot] / val

                elim_step = EliminationStep(pivot, row, factor)
                q.append(elim_step)

                if rref:
                    steps.append(elim_step)
                else:
                    steps.append(DecompStep(row, pivot, factor))

            while q:
                out.elim_step(q.popleft())

        if rref:
            self.__rref(out, steps, q)

        return out, steps

    def __rref(self, out, steps, q):
        """
        Internal function to do rref after gaussian elimination
        Called by gaussian_elimination
        """

        # First divide all rows to get a 1 in the pivot column
        for pivot in range(out.cols):
            val = out.data[pivot][pivot]

            if val != 0:
                div_step = EliminationStep(pivot, pivot, val, div=True)
                steps.append(div_step)
                out.elim_step(div_step)

        # Eliminate stuff above and below the pivots
        for pivot in range(out.cols):
            val = out.data[pivot][pivot]

            if val == 0:
                continue

            for row in range(out.rows):
                if row == pivot:
                    continue
                factor = out.data[row][pivot] / val
                elim_step = EliminationStep(pivot, row, factor)

                q.append(elim_step)
                steps.append(elim_step)

            while q:
                out.elim_step(q.popleft())

        return out, steps

    def elim_step(self, elim_step: EliminationStep):
        """
        Apply a single elimination step to the matrix
        """
        if elim_step.pivot_row not in range(
            self.rows
        ) or elim_step.other_row not in range(self.rows):
            raise ValueError("Invalid row index in EliminationStep")

        for i in range(self.cols):
            if elim_step.div:
                self.data[elim_step.pivot_row][i] /= elim_step.multiplier
            else:
                self.data[elim_step.other_row][i] -= (
                    self.data[elim_step.pivot_row][i] * elim_step.multiplier
                )

    def apply_elim_steps(self, steps: List[EliminationStep]):
        """
        Apply a list of elimination steps to the matrix
        """
        out = self.copy()
        for step in steps:
            out.elim_step(step)
        return out

    def apply_decomp_steps(self, steps: List[DecompStep]):
        """
        Apply a list of decomp steps to the matrix
        """
        out = self.copy()
        for step in steps:
            out.data[step.row][step.col] = step.value
        return out

    def lu_decomp(self):
        """
        Return tuple of the LU decomposition of the matrix
        """
        I = Matrix.identity(self.rows)
        U = self.copy()

        U, decomp_steps = U.gaussian_elimination()
        L = I.apply_decomp_steps(decomp_steps)
        return L, U

    def copy(self):
        """
        Create a copy of the matrix
        """
        return Matrix([n for row in self.data for n in row], rows=self.rows)

    def det(self):
        """
        Calculate the determinant of the matrix from LU decomposition
        Just products of diagonal elements
        """
        if self.rows != self.cols:
            raise ValueError("Only square matrices can have determinants")
        L, U = self.lu_decomp()
        det = 1
        for i in range(self.rows):
            det *= L.data[i][i] * U.data[i][i]
        return det

    def inverse(self):
        """
        Calculate the inverse of the matrix by doing RREF on the identity matrix
        """
        if self.rows != self.cols:
            raise ValueError("Only square matrices can have inverses")
        if self.det() == 0:
            raise ValueError("Matrix is not invertible")

        _, steps = self.gaussian_elimination(rref=True)
        I = Matrix.identity(self.rows)
        return I.apply_elim_steps(steps)

    @classmethod
    def ones(cls, rows, cols):
        """
        Create a matrix of ones
        """
        return cls([1] * (rows * cols), rows=rows)

    @classmethod
    def zeros(cls, rows, cols):
        """
        Create a matrix of zeros
        """
        return cls([0] * (rows * cols), rows=rows)

    @classmethod
    def identity(cls, rows):
        """
        Create identity matrix of size rows x rows
        """
        mat = cls.zeros(rows, rows)
        for i in range(rows):
            mat.data[i][i] = 1
        return mat
