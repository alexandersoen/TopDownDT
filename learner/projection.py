import numpy as np

from learner.learner import Learner

class Projection(Learner):

    def __init__(self, dim_size):
        self.dim_size = dim_size

    def learn(self, loss):
        best_proj = None
        best_loss = float('inf')

        # Go over each dimension
        for i in range(dim_size):
            proj_v = np.zeros(4)
            proj_v[i] = 1

            # Projection function
            def proj(xs):
                xs = np.array(xs)
                return xs @ proj_v

            cur_loss = loss(proj)

            # Check for new best
            if cur_loss < best_loss:
                best_proj = proj
                best_loss = cur_loss

        return best_loss, best_proj