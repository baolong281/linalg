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

    # users the @ operator
    # i.e. mat @ mat2
    def __matmul__(self, other):
        """
        Matrix multiplication with @ operator
        i.e. mat @ mat2
        Naive implementation
        """

        if self.cols != other.rows:
            raise ValueError("Matrix dimensions must be compatible")

        out = Matrix([0] * (other.cols * self.rows), rows=self.rows)

        for i in range(self.rows):
            for j in range(other.cols):
                for k in range(self.cols):
                    out.data[i][j] += self.data[i][k] * other.data[k][j]

        return out

    def __eq__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("Can only compare Matrix with another Matrix")
        return all(
            self.data[i][j] == other.data[i][j]
            for i in range(self.rows)
            for j in range(self.cols)
        )

    # do this later
    def __add__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("Can only add a Matrix to another Matrix")
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must be the same size")

        out = Matrix([0] * (self.rows * self.cols), rows=self.rows)

        for i in range(self.rows):
            for j in range(self.cols):
                out.data[i][j] = self.data[i][j] + other.data[i][j]

        return out

    # do this later
    def __sub__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("Can only subtract a Matrix from another Matrix")
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must be the same size")

        return self + (other * -1)

    # scalar mul do later
    def __mul__(self, other):
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

    # do later
    # determinant of matrix
    def det(self):
        if self.rows != self.cols:
            raise ValueError("Only square matrices can have determinants")
        L, U = self.lu_decomp()
        det = 1
        for i in range(self.rows):
            det *= L.data[i][i] * U.data[i][i]
        return det

    # matrix inverse
    def inverse(self):
        if self.rows != self.cols:
            raise ValueError("Only square matrices can have inverses")
        if self.det() == 0:
            raise ValueError("Matrix is not invertible")

        rref, steps = self.gaussian_elimination(rref=True)
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
