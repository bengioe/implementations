import numpy as np
from collections import namedtuple
import torch

class Node2:
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
        self.root = Node2(0, None)
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
                node.children[a] = Node2(pi[a], node)
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

class MCTSwithBootstrap:

    def __init__(self, nsim, s0, nact, forward, pred, transforms,
                 meta_action, meta_forward, c_1=np.sqrt(2), gamma=0.99,
                 temperature=1., bootstrap_depth=4):
        self.root = Node2(0, None)
        self.nexpanded = 0
        self.maxv = -1000
        self.minv = 1000
        self.num_meta = 0
        mmnorm = lambda v: (v - self.minv) / (self.maxv - self.minv) if self.maxv > self.minv else v
        mmnorm = lambda v: v
        ivt = lambda v: transforms.inverse_value_transform(v)
        irt = lambda r: transforms.inverse_reward_transform(r)

        def do_expand(node, s, r):
            self.nexpanded += 1
            pi, v = pred(s)
            pi = pi.softmax(1).cpu().data.numpy()[0]
            v = ivt(v)
            node.s = s
            node.reward = r
            for a in range(nact):
                node.children[a] = Node2(pi[a], node)
            node.omega, node.omega_action = meta_action(s, argmax=True)
            #meta_step = Node2(0, node)
            #node.children[node.omega_action] = meta_step
            #meta_step.s, meta_step.reward = meta_forward(node.s, node.omega)
            #meta_step.value_sum = ivt(pred(meta_step.s)[1])
            #meta_step.reward = irt(meta_step.reward)
            #meta_step.visit_count += 1
            return v

        def do_backup(node, g):
            while node.parent is not None:
                node.parent.value_sum += g
                node.parent.visit_count += 1
                self.minv, self.maxv = min(self.minv, g), max(self.maxv, g)
                g = node.reward + gamma * g
                node = node.parent

        do_expand(self.root, s0, 0)
        noise = np.random.dirichlet([0.25] * len(self.root.children))
        for a, z in zip(self.root.children, noise):
            self.root.children[a].prior = self.root.children[a].prior * 0.75 + 0.25 * z

        for sim in range(nsim):
            node = self.root
            # Select down until leaf is reached
            while node.expanded():
                _, action, node = max(
                    (([np.sqrt(node.visit_count) / (child.visit_count + 1) * child.prior
                      + (child.reward + gamma * mmnorm(child.v)
                         if child.visit_count > 0 else 0)],
                     action, child)
                    for action, child in node.children.items()), key=lambda x:x[0])
            if action == node.parent.omega_action:
                sp, r = meta_forward(node.parent.s, node.parent.omega)
                self.num_meta += 1
            else:
                sp, r = forward(node.parent.s, torch.eye(nact)[None, action])
            r = transforms.inverse_reward_transform(r)
            v = do_expand(node, sp, r) # expand leaf
            do_backup(node, v)

        weights = np.float32([self.root.children[a].visit_count
                              for a in range(nact)]) ** (1 / temperature) + 1e-4
        #weights[self.root.omega_action] += self.root.children['meta'].visit_count
        weights /= weights.sum()
        self.softmax_pol = torch.tensor(weights)
        self.bootstrap_action_target = weights.argmax()
        node = self.root
        reward = torch.tensor(0.0)
        ts = self.root.s
        for i in range(bootstrap_depth):
            _, node = max(
                ((child.visit_count, child)
                 for action, child in node.children.items()), key=lambda x:x[0])
            reward = reward + node.reward
            if node.s is not None:
                ts = node.s
            if not node.expanded():
                break
        self.bootstrap_state_target = ts
        if ts is None:
            import pdb; pdb.set_trace()
        self.bootstrap_reward_target = reward

class MCTS__:

    def __init__(self, nsim, s0, nact, forward, pred, transforms, c_1=np.sqrt(2), gamma=0.99,
                 temperature=1.1):
        self.root = Node(s0)
        self.nexpanded = 0

        def do_expand(node):
            self.nexpanded += 1
            st = node.s.repeat(nact, 1)
            sp, r = forward(st, torch.eye(nact))
            pi, v = pred(sp)
            r = transforms.inverse_reward_transform(r)
            v = transforms.inverse_value_transform(v)
            node.pi = pi.softmax(1).cpu().data.numpy()[0]
            for a in range(nact):
                node.add_child(a, sp[a], v[a], r[a], gamma)

        def do_backup(node):
            g = 0
            while node.parent is not None:
                idx, parent = node.parent
                gp = parent.r[idx] + gamma * g
                parent.w[idx] += gp
                parent.n[idx] += 1
                g = gp
                node = parent

        for sim in range(nsim):
            node = self.root
            # Select down until leaf is reached
            while node.children:
                #node = node.children[np.argmax(node.q + c_1 * np.sqrt(np.sum(node.n)) / node.n)]
                a = node.pi
                b = np.sqrt(np.sum(node.n)) / (np.float32(node.n)+1)
                node = node.children[np.argmax(node.q + c_1 * a * b)]
            do_expand(node) # expand leaf
            for c in node.children:
                do_backup(c) # backup (which does recursive progressive widening)

        weights = np.float32(self.root.n) ** (1 / temperature)
        #weights = np.exp(weights) / temperature
        weights /= weights.sum()
        self.softmax_pol = torch.tensor(weights)


class CMCTS:

    def __init__(self, nsim, s0, expand, forward, pred, c_1=np.sqrt(2), gamma=0.99):
        self.root = Node(s0)
        self.nexpanded = 0

        def do_expand(node):
            self.nexpanded += 1
            act = expand(node.s)
            sp, r = forward(node.s, act)
            pi, v = pred(sp)
            node.add_child(act, sp, v, r, gamma)

        def do_backup(node):
            g = node.w[-1]
            while node.parent is not None:
                idx, parent = node.parent
                parent.w[idx] += parent.r[idx] + gamma * g
                parent.n[idx] += 1
                g = parent.w[idx]
                node = parent

                # progressive widening
                while len(node.children) < 1 + np.sqrt(sum(node.n)):
                    #print('widen', len(node.children), np.log(sum(node.n)))
                    do_expand(node)
                    do_backup(node)

        for sim in range(nsim):
            node = self.root
            # Select down until leaf is reached
            while node.children:
                node = node.children[np.argmax(node.q + c_1 * np.sqrt(np.sum(node.n)) / node.n)]
            do_expand(node) # expand leaf
            do_backup(node) # backup (which does recursive progressive widening)

    def sample(self, temperature=1):
        weights = self.root.q
        weights = np.exp(weights) / temperature
        weights /= weights.sum()
        return self.root.a[np.random.multinomial(1, weights).argmax()]

class Node:
    def __init__(self, s):
        self.s = s
        self.r = []
        self.a = []
        self.w = []
        self.n = []
        self.parent = None
        self.children = []

    @property
    def q(self):
        return np.float32(self.w) / np.float32(self.n)

    def add_child(self, a, sp, v, r, gamma):
        n = Node(sp)
        n.parent = (len(self.a), self)
        self.a.append(a)
        self.w.append(r.item() + gamma * v.item())
        self.r.append(r.item())
        self.n.append(0)
        self.children.append(n)
