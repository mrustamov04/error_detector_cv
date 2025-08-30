import numpy as np

#vector 2-D slicing operations
a = np.arange(20).reshape(-1, 10)
print(a)

# #access 5 consecutive elements (start:stop:step)
# print("a[0, 2:7:1] = ", a[0, 2:7:1], ",  a[0, 2:7:1].shape =", a[0, 2:7:1].shape, "a 1-D array")

# #access 5 consecutive elements (start:stop:step) in two rows
# print("a[:, 2:7:1] = \n", a[:, 2:7:1], ",  a[:, 2:7:1].shape =", a[:, 2:7:1].shape, "a 2-D array")

# # access all elements
# print("a[:,:] = \n", a[:,:], ",  a[:,:].shape =", a[:,:].shape)

# # access all elements in one row (very common usage)
# print("a[1,:] = ", a[1,:], ",  a[1,:].shape =", a[1,:].shape, "a 1-D array")
# # same as
# print("a[1]   = ", a[1],   ",  a[1].shape   =", a[1].shape, "a 1-D array")