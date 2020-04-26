import numpy as np
from collections import namedtuple
import torch

class Node:
    def __init__(self, prior, parent):
        self.parent = parent
        self.children = {}
        self.s = None

        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.reward = 0
        self.omega = None

    def expanded(self):
        return len(self.children) > 0

    @property
    def v(self):
        return self.value_sum / self.visit_count if self.visit_count else 0

    @property
    def rpv(self):
        return self.reward + .99 * self.value_sum / self.visit_count if self.visit_count else 0

class MCTS:

    def __init__(self, nsim, s0, nact, forward, pred, transforms, c_1=np.sqrt(2), gamma=0.99,
                 temperature=1.):
        self.root = Node(0, None)
        self.nexpanded = 0
        self.maxv = -1000
        self.minv = 1000
        mmnorm = lambda v: (v - self.minv) / (self.maxv - self.minv) if self.maxv > self.minv else v
        mmnorm = lambda v: v

        def do_expand(node, s, r):
            self.nexpanded += 1
            pi, v = pred(s)
            pi = pi.softmax(1).cpu().data.numpy()[0]
            v = transforms.inverse_value_transform(v)
            node.s = s
            node.reward = r
            for a in range(nact):
                node.children[a] = Node(pi[a], node)
            return v

        def do_backup(node, g):
            while node.parent is not None:
                node.parent.value_sum += g
                node.parent.visit_count += 1
                self.minv, self.maxv = min(self.minv, g), max(self.maxv, g)
                g = node.reward + gamma * g
                node = node.parent

        do_expand(self.root, s0, 0)
        noise = np.random.dirichlet([0.25] * nact)
        for a in range(nact):
            self.root.children[a].prior = self.root.children[a].prior * 0.75 + 0.25 * noise[a]

        for sim in range(nsim):
            node = self.root
            # Select down until leaf is reached
            while node.expanded():
                _, action, node = max(
                    ([np.sqrt(node.visit_count) / (child.visit_count + 1) * child.prior
                      + (child.reward + gamma * mmnorm(child.v)
                         if child.visit_count > 0 else 0)],
                     action, child)
                    for action, child in node.children.items())
            sp, r = forward(node.parent.s, torch.eye(nact)[None, action])
            r = transforms.inverse_reward_transform(r)
            v = do_expand(node, sp, r) # expand leaf
            do_backup(node, v)

        weights = np.float32([self.root.children[a].visit_count
                              for a in range(nact)]) ** (1 / temperature) + 1e-4
        weights /= weights.sum()
        self.softmax_pol = torch.tensor(weights)
