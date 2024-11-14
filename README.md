# linalg

## usage
```python
import linalg
a = linalg.Matrix([1, 2, 3, 4, 5, 6, 7, 8, 9], rows=3)
b = linalg.Matrix([4, 5, 6, 7, 8, 9, 10, 11, 12], rows=3)

# Outputs new matrix of their product
a @ b

# Determinant
a.det()

# Addition
a + b

# LU decomposition
L, U = a.lu_decomp()

# Inverse matrix
a.inverse()
```

## running tests

```bash
python3 -m unittest test.py
```
