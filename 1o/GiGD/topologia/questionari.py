import numpy as np

a = np.array((0,1,1,0,1))
b = np.array((0,1,0,1,1))

print(np.count_nonzero(a != b))
#print(np.linalg.norm(a-b))