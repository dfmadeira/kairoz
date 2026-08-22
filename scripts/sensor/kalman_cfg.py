import numpy as np

Q = np.diag([1, 1]) # TODO: this should change adaptively based on the MLP residual

R = np.diag([
    0.05,
    0.01,
    0.10
])

P0 = np.eye(2)
