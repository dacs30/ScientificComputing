# %%
import math

# A method that will get a and b, where a is the constant, b the iterations

# Brute force solution
# very simple logic, we are just accumulating the sums and subtracting in the end
def equationMethod(a, b):
    acc = 0
    i = 1
    while (i <= b):
        acc = acc + 0.1
        i = i+1
    return a - acc

print("brute force")
print(equationMethod(1000, 10000))
print(equationMethod(10000, 100000))
print(equationMethod(100000, 1000000))

# Other solutions:

#  Using math.sum

def sumMethod(a, b):
    result = a - sum(0.1 for _ in range(b))
    return abs(result)

print("sum")
print(sumMethod(1000, 10000))
print(sumMethod(10000, 100000))
print(sumMethod(100000, 1000000))

# using fsum
def fsumMethod(a, b):
    result = a - math.fsum(0.1 for _ in range(b))
    return abs(result)

print("fsum")
print(fsumMethod(1000, 10000))
print(fsumMethod(10000, 100000))
print(fsumMethod(100000, 1000000))


#  using numpy
import numpy as np
def numpyMethod(a, b):
    return abs(a - np.sum(np.full(b, 0.1)))

print("numpy")
print(numpyMethod(1000, 10000))
print(numpyMethod(10000, 100000))
print(numpyMethod(100000, 1000000))

# %%
import numpy as np

A = np.array([[1,  2],
              [-1, 1]])

B = np.array([[2, 0],
              [0, 2]])

C = np.array([[2, 0, -3],
              [0, 0, -1]])   # 2×3

D = np.array([[1,  2],
              [2,  3],
              [-1, 0]])      # 3×2

# All vectors are 1D arrays in numpy
x = np.array([1, 0])        # 2×1

y = np.array([0, 1])        # 2×1

z = np.array([1, 2, -1])    # 3×1

# stack them into a 2D array
matricesAB = np.array([A, B])
print("A+B", np.sum(matricesAB, axis=0))

multipliedMatrices= np.array([3*x, -4*y])
print("3x-4y", np.sum(multipliedMatrices, axis=0))

print("Ax", np.matmul(A, x))

subXY = x-y
print("B(x-y)", np.matmul(B, subXY))

print("Dx", np.matmul(D, x))

newDy = np.matmul(D, y)
print("Dy+z", newDy + z)

print("AB", np.matmul(A, B))

print("BC", np.matmul(B, C))

print("CD", np.matmul(C, D))



# %%
# x_{n+1} = px_{n} (1-x_{n})
# p = 0.5

def logisticMap(p, x0, n):
    x = x0
    for i in range(n):
        x = p * x * (1 - x)
    return x

print(logisticMap(0.8, 0.5, 49))
print(logisticMap(1.5, 0.5, 49))
print(logisticMap(2.8, 0.5, 49))
print(logisticMap(3.2, 0.5, 49))
print(logisticMap(3.5, 0.5, 49))
print(logisticMap(3.65, 0.5, 49))


