import numpy as np
np.random.seed(205)
n = 4096
print(3000, n)
for _ in range(n):
    print(1000, 5, *np.random.normal([512, 384, 0], 100), *np.random.normal(0, 25, 3))