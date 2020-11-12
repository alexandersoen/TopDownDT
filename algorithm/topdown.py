import copy

from tree.dtree import DecisionNode, DecisionTree
from tree.tree import Direction

class TopDown:

    def __init__(self, learner, criterion):
        self.learner = learner
        self.criterion = criterion

    def __call__(self, samples, t):
        _, ys = samples

        # Initialise T to be the single-leaf tree, majority labelled
        m_label = max(set(ys), label=ys)
        root = DecisionNode(Direction(m_label), None)
        dtree = DecisionTree(root)

        # While T has fewer than t internal nodes
        while dtree.size < t:

            # Setup best  
            best_loss = float('inf')
            best_grow = None
            for l, l_samp in partition(samples, dtree):
                cur_xs, cur_ys = l_samp

                # Probability of reaching leaf and having a 1 at leaf
                w = len(cur_ys) / len(ys)
                q = sum(cur_ys) / len(cur_ys)

                # Loss contribution of editing a single leaf
                def cur_loss_func(h):
                    pred_ys = h(cur_xs)
                    tau = sum(pred_ys) / len(cur_xs)

                    p = sum((1 - p_y) * c_y for (p_y, c_y) in zip(pred_ys, cur_ys)) / len(cur_ys)  # q(l0)
                    r = sum(p_y * c_y for (p_y, c_y) in zip(pred_ys, cur_ys)) / len(cur_ys)  # q(l1)

                    # loss will be negative of the decrease in criterion
                    return - w * (self.criterion(q) - (1 - tau) * self.criterion(p) - tau * self.criterion(r))

                # Learn the best node splitter for current leaf
                loss, h = self.learner.learn(cur_loss_func)
                if loss < best_loss:
                    best_loss = loss
                    best_grow = (l, h)

            # Grow tree as best
            dtree.grow(*best_grow)

        return dtree


def partition(samples, dtree: DecisionTree):
    partition = {}
    name_to_leaf = {l.name: l for l in dtree.leaves}

    # Partition data into tree leaves
    for s in zip(*samples):
        s_x, _ = s
        l_n = dtree(s_x, eval_f=lambda x: x.name)

        partition[l_n].append(s)

    # Return leaf + transformed samples
    return ((name_to_leaf[l_n], list(zip(*l_samp))) for (l_n, l_samp) in partition.items())