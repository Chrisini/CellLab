import numpy as np

a = np.full((2,3), 4)
print("a: 2x3 Matrix with 4")
print(a)
print()
b = np.array([[1,2,3],[4,5,6]])
print("b: 2x3 Matrix with given values")
print(b)
print()
c = np.eye(2,3)
print("c: 2x3 identity matrix")
print(c)
print()
d = a + b + c
print("d: sum of matrices")
print(d)
print()

e = np.array([[1,2,3,4,5],
			  [5,4,3,2,1],
			  [6,7,8,9,0],
			  [0,9,8,7,6]])

print("x: Rank 1 = one dimension of row 3:")
x = e[2,:]
print(x)
print("x: Shape:", x.shape)
print()
print("y: Rank 2 = two dimensions of row 3:")
y = e[2:3,:]
print(y)
print("y: Shape: ", y.shape)
print()

f=e
g=e
print("e: rank 2 array")
print(e.dot(2))
print("e")

print("f: sum the rows")
print(np.sum(f, axis=1))
print()

print("g: transponse")
print(g.T)
print()