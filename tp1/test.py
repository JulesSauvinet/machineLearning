

import numpy as np

a = np.array([[1, 2], [3, 4]])
print a

print ""

b = np.array([[5], [6]])
print b


print np.concatenate((a, b), axis=1)