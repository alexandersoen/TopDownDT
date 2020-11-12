from enum import Enum
from typing import List

from tree.tree import Node, Tree, Direction

id_value = lambda n: n.value

class DecisionNode(Node):
    def __init__(self, value: Direction, parent: 'Node') -> None:
        super().__init__(value, parent)
        self.h = None

    def __call__(self, x, eval_f=id_value):
        # If a leaf then just pass the value
        if self.is_leaf:
            return eval_f(self)

        # Evaluate the leaves depending on split direction
        if self.h(x) == Direction.LEFT:
            return self.left(x)
        else:
            return self.right(x)

    def split(self, h):
        self.h = h
        self.left = DecisionNode(Direction.LEFT, self)
        self.right = DecisionNode(Direction.RIGHT, self)


class DecisionTree(Tree):
    def __call__(self, x, eval_f=id_value):
        return self.root(x, eval_f=eval_f)

    def grow(self, l: DecisionNode, h):
        l.split(h)