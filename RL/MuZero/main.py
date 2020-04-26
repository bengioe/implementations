import time
import numpy as np
import torch
import torch.nn as nn
import minatar
import ray
from tqdm import tqdm
import os
from mcts import MCTS
tfd = lambda x: torch.tensor(x).float().to(device)
tf = lambda x: torch.tensor(x).float()
tn = lambda x: x.data.cpu().numpy()


num_cpus = 6

ray.init(num_cpus=num_cpus-1, include_webui=False, temp_dir=os.environ['SLURM_TMPDIR'])


class DiscreteSupport:
    def __init__(self, min: int, max: int):
        assert min < max
        self.min = min
        self.max = max
        self.range = range(min, max + 1)
        self.size = len(self.range)
        self.arange = torch.arange(min, max+1)

class Transforms:

    def __init__(self):
        self.value_support = DiscreteSupport(-20, 20)
        self.reward_support = DiscreteSupport(-5, 5)

    def scalar_phi_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    @staticmethod
    def scalar_transform(x):
        """ Reference : Appendix F => Network Architecture
        & Appendix A : Proposition A.2 in https://arxiv.org/pdf/1805.11593.pdf (Page-11)
        """
        epsilon = 0.001
        sign = torch.ones(x.shape).float().to(x.device)
        sign[x < 0] = -1.0
        output = sign * (torch.sqrt(torch.abs(x) + 1) - 1 + epsilon * x)
        return output

    def inverse_reward_transform(self, reward_logits):
        return self.inverse_scalar_transform(reward_logits, self.reward_support)

    def inverse_value_transform(self, value_logits):
        return self.inverse_scalar_transform(value_logits, self.value_support)

    def inverse_scalar_transform(self, logits, scalar_support):
        """ Reference : Appendix F => Network Architecture
        & Appendix A : Proposition A.2 in https://arxiv.org/pdf/1805.11593.pdf (Page-11)
        """
        value_probs = torch.softmax(logits, dim=1)
        #value_support = torch.ones(value_probs.shape)
        #value_support[:, :] = torch.tensor([x for x in scalar_support.range])
        value_support = scalar_support.arange.repeat(value_probs.shape[0], 1)
        value_support = value_support.to(device=value_probs.device)
        value = (value_support * value_probs).sum(1, keepdim=True)
        if 1:
            return value

        epsilon = 0.001
        sign = torch.ones(value.shape).float().to(value.device)
        sign[value < 0] = -1.0
        output = (((torch.sqrt(1 + 4 * epsilon * (torch.abs(value) + 1 + epsilon)) - 1) / (2 * epsilon)) ** 2 - 1)
        output = sign * output
        return output

    def value_phi(self, x):
        return self._phi(x, self.value_support.min, self.value_support.max, self.value_support.size)

    def reward_phi(self, x):
        return self._phi(x, self.reward_support.min, self.reward_support.max, self.reward_support.size)

    @staticmethod
    def _phi(x, min, max, set_size: int):
        x.clamp_(min, max)
        x_low = x.floor()
        x_high = x.ceil()
        p_high = (x - x_low)
        p_low = 1 - p_high

        target = torch.zeros(x.shape[0], x.shape[1], set_size).to(x.device)
        x_high_idx, x_low_idx = x_high - min, x_low - min
        target.scatter_(2, x_high_idx.long().unsqueeze(-1), p_high.unsqueeze(-1))
        target.scatter_(2, x_low_idx.long().unsqueeze(-1), p_low.unsqueeze(-1))
        return target

transforms = Transforms()

class Clamp(nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return x.clamp(self.min, self.max)

class Model(nn.Module):
    def __init__(self, nh, nact, env):
        super().__init__()
        self.nh = nh
        self.transforms = transforms
        self.nact = nact

        # s_t -> h_t
        self.obs2hid = nn.Sequential(
            nn.Conv2d(env.nchannels, nh // 4, 3, padding=1), nn.LeakyReLU(),
            nn.Conv2d(nh // 4, nh // 2, 3, padding=1), nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(nh // 2 * 10 * 10, nh), nn.Tanh())#Clamp(0,1))

        # predicts h_t, a_t -> h_tp1, r_t
        self.forwardmodel = nn.Sequential(
            nn.Linear(nh + nact, nh), nn.LeakyReLU(),
            nn.Linear(nh, nh), nn.LeakyReLU(),
            nn.Linear(nh, nh + transforms.reward_support.size))

        # predicts h_t -> v_t, pi_t
        self.hid2pred = nn.Sequential(
            nn.Linear(nh, nh), nn.LeakyReLU(),
            nn.Linear(nh, nh), nn.LeakyReLU(),
            nn.Linear(nh, transforms.value_support.size + nact))

    def mforward(self, x, a):
        h, r = colsplit(self.forwardmodel(torch.cat([x, a], 1)),
                        self.nh, transforms.reward_support.size)
        h = torch.tanh(h)
        #h = h.clamp(0, 1)
        return h, r

    def pred(self, x):
        return colsplit(self.hid2pred(x), self.nact, transforms.value_support.size)

    def get_weights(self):
        d = self.state_dict()
        for k,v in d.items():
            d[k] = v.cpu()
        return d

    def set_weights(self, weights):
        self.load_state_dict(weights)

@ray.remote
class StoreActor:
    def __init__(self):
        self.store = None

    def get(self):
        return self.store

    def set(self, v):
        self.store = v

@ray.remote
class EpisodeWorker:
    def __init__(self, workerid, nh, weight_store):
        self.weight_store = weight_store
        self.workerid = workerid
        self.env = make_env()
        self.nact = self.env.nact
        self.model = Model(nh, self.nact, self.env)

    def do_episode(self, replay):
        self.model.set_weights(ray.get(self.weight_store.get.remote()))
        forward = self.model.mforward
        pred = self.model.pred
        state = self.env.reset()
        done = False
        episode = Episode()
        while not done:
            with torch.no_grad():
                hid = self.model.obs2hid(state)
                v = transforms.inverse_value_transform(pred(hid)[1])
                tree = MCTS(25, hid, self.nact, forward, pred, transforms)
                tree_policy = tree.softmax_pol# * 0.999
                action = tf(np.random.multinomial(1, tn(tree_policy)).argmax()).long()
            statep, r, done, info = self.env.step(action.item())
            episode.add(state[0], action, tf(r), tf(done), tree_policy, v)
            state = statep
        episode.done()
        ray.get(replay.add_episode.remote(episode))
        return self.workerid


def make_env():
    #return TestEnv()
    return MinAtarEnv('breakout')

def main():

    env = make_env()
    nact = env.nact
    nh = 128
    mbsize = 512
    K = 3

    model = Model(nh, nact, env)
    for p in []:#model.parameters():
        if p.ndim == 4:
            k = np.sqrt(6 / (p.shape[0] + p.shape[1]) * np.prod(p.shape[2:]))
            p.data.uniform_(-k,k)
        elif p.ndim == 2:
            k = np.sqrt(6 / (p.shape[0] + p.shape[1]))
            p.data.uniform_(-k,k)
        elif p.ndim == 4:
            p.data.zero_()
    if cuda:
        model.to(device)

    forward = model.mforward
    pred = model.pred

    opt = torch.optim.SGD(model.parameters(), 1e-3, momentum=0.9, weight_decay=1e-4)

    replay = ReplayBuffer.remote(1000)
    wstore = StoreActor.remote()
    ray.get(wstore.set.remote(model.get_weights()))

    mse = lambda a,b: ((a-b)**2).mean()

    #xent = lambda p,t: (p * torch.log(t)).mean() * p.shape[1]
    softmax = nn.Softmax(1)
    logsoftmax = nn.LogSoftmax(1)
    state = env.reset()
    pbar = tqdm(range(200000), smoothing=0.05)
    rema = 0
    lema = np.zeros(5)
    tema = np.zeros(4)
    veema = 0
    loss = tf(0)
    pi_ent = 0

    @ray.remote
    def local_sample(replay, n, k):
        eidx = np.random.randint(0, ray.get(replay.num_episodes.remote()), n)
        episodes = ray.get(replay.get_episodes.remote(eidx))
        lens = np.int32([len(e.s) for e in episodes])
        ts = np.minimum(lens-k, np.int32([e.sample() for e in episodes]))
        return [
            torch.stack([getattr(e, a)[t:t+k] for e,t in zip(episodes,ts)], 1)
            for a in ('s','a','r','nt','pi', 'g')]

    workers = [EpisodeWorker.remote(i, nh, wstore) for i in range(num_cpus-3)]

    episodes = [i.do_episode.remote(replay) for i in workers]

    for step in pbar:

        pbar.update(1)
        if step > 4:
            tema = tema * 0.99  + np.float32([t1-t0,t2-t1, t3-t2, t4-t3])*0.01
            pi_ent = (pi[i] * torch.log(pi[i])).sum(1).mean().item()

        t0 = time.time()
        eps, episodes = ray.wait(episodes, len(episodes), timeout=0 if step > 4 else None)
        if not step % 10:
            wstore.set.remote(model.get_weights())
        for ep in eps:
            workerid = ray.get(ep)
            #replay.add_episode(ep)
            episodes.append(workers[workerid].do_episode.remote(replay))
            if step > 4:
                tr = ray.get(replay.last_episode_reward.remote())
                veema = veema * 0.99 + 0.01 * ray.get(replay.last_episode_verr.remote())
                #with torch.no_grad():
                #    asd = forward(h[:1].repeat(nact, 1), torch.eye(nact).to(device))[1]
                #    asd = transforms.inverse_reward_transform(asd)
            else:
                tr = 0
                asd = 0
            rema = rema * 0.99 + tr * 0.01
            pbar.set_description(
                f'{rema:.1f} {pi_ent:.2f} {veema:.2f} :: '+','.join(f'{i:.3f}' for i in lema)
                +' :: '
                +','.join(f'{i:.3f}' for i in tema))


        if step < 4:
            continue
        if step == 4:
            minibatches = [replay.sample_slices.remote(mbsize, K) for i in range(3)]
            #minibatches = [local_sample.remote(replay, mbsize, K) for i in range(2)]

        t1 = time.time()
        #s, a, r, nt, pi, g, verr = [torch.from_numpy(i).to(device) for i in ray.get(minibatches.pop(0))]
        s0, a, r, pi, g, P = [torch.from_numpy(i).to(device) for i in ray.get(minibatches.pop(0))]
        minibatches.append(replay.sample_slices.remote(mbsize, K))
        #minibatches.append(local_sample.remote(replay, mbsize, K))

        t2 = time.time()
        # s: (K, mbs, nc, w, h)
        # h = model.obs2hid(s[0])
        h = model.obs2hid(s0)
        __g = g
        if 0:
            with torch.no_grad():
                vk = pred(model.obs2hid(s[K-1]))[1][:, 0] * nt[K-1]
            g = [vk] # check this
            for i in range(1, K):
                g.append(g[-1] * gamma * nt[K-i-1] + r[K-i-1])
            g = torch.stack(g[::-1])
            #import pdb; pdb.set_trace()
        losses = []
        _r = r
        _g = g
        r = transforms.reward_phi(r)
        g = transforms.value_phi(g)
        xent = lambda p,q,P=1: (-(p * logsoftmax(q)).sum(1) / P).mean()
        #xent = lambda p,q: (-(p * logsoftmax(q))).mean() * p.shape[-1]
        vinvs = []
        P = P.clamp(1e-1, 1)
        #P = 1
        for i in range(K):
            p, v = pred(h)
            hp, rp = forward(h, mbonehot(a[i], nact))
            l = xent(r[i], rp, P), xent(g[i], v, P), xent(pi[i], p, P)
            #l = xent(r[i], rp), xent(g[i], v), xent(pi[i], p)
            losses.append(sum(l))

            # logging losses:
            rpinv = transforms.inverse_reward_transform(rp).flatten()
            vinv = transforms.inverse_value_transform(v).flatten()
            l2 = l[:2] + (mse(_r[i], rpinv), mse(vinv, _g[i]), xent(pi[i], p))
            vinvs.append(vinv)

            h = hp
        loss = sum(losses) / K
        t3 = time.time()
        loss.backward()
        lema = lema * 0.99 + np.float32([i.item() for i in l2]) * 0.01
        opt.step()
        opt.zero_grad()
        t4 = time.time()

        if step % 100 == 0 and False:
            print()
            print(_r[:, :5])
            print(__g[:, :5])
            print(_g[:, :5])
            print(torch.stack(vinvs)[:, :5])
            print(verr[:, :5])
            ep = ray.get(replay.get_episodes.remote([-1]))[0]
            s = ep.s[:5]
            v = model.pred(model.obs2hid(s.to(device)))[1]
            v = transforms.inverse_value_transform(v).flatten()
            print(v, ep.vpred[:5])
            print()




def colsplit(x, *lens):
    start = np.cumsum((0,)+lens)
    return [x[:, start[i]:start[i+1]] for i in range(len(lens))]

def mbonehot(idx, n):
    return torch.eye(n)[idx].to(device)

class MinAtarEnv:
    def __init__(self, env='seaquest'):
        self.env = minatar.Environment(env, 0)
        self.nchannels = self.env.n_channels
        self.nact = self.env.num_actions()
        self.enumber = 0

    def state(self):
        return tf(self.env.state().transpose(2,0,1))[None, :]# - 0.0825) / 0.275

    def reset(self):
        self.enumber+=1
        self.env.reset()
        return self.state()

    def step(self, a):
        r, done = self.env.act(a)
        return self.state(), r, done, {}





class TestEnv:
    def __init__(self, *a, **k):
        self._state = [0,0]
        self.nchannels = 2
        self.nact = 2

    def obs(self):
        z = torch.zeros((1,2,10,10))
        z[0,0,self._state[0],2] = 1
        z[0,1,2,self._state[1]] = 1
        return z

    def reset(self):
        self._state = [0,0]
        return self.obs()

    def step(self, a):
        r = float(a == 1 and (self._state[1] % 2) == 0 and 3 <= self._state[0] <= 5)
        self._state[a] += 1
        return self.obs(), r, max(self._state) == 7, {}


class Episode:
    def __init__(self):
        self.id = 0
        self.s = []
        self.a = []
        self.r = []
        self.nt = []
        self.pi = []
        self.vpred = []

    def add(self, s, a, r, t, pi, vp):
        self.s.append(s)
        self.a.append(a)
        self.r.append(r)
        self.nt.append(1-t)
        self.pi.append(pi)
        self.vpred.append(vp)

    def done(self):
        self.s = torch.stack(self.s).numpy()
        self.a = torch.stack(self.a).numpy()
        self.r = torch.stack(self.r).numpy()
        self.nt = torch.stack(self.nt).numpy()
        self.pi = torch.stack(self.pi).numpy()
        self.vpred = torch.stack(self.vpred).flatten().numpy()
        self.g = [0]
        t = len(self.r)
        for i in range(t):
            self.g.append(self.r[t-i-1] + gamma * self.g[-1] * self.nt[t-i-1])
        self.g = np.float32(self.g[:0:-1])
        self.sumtree = SumTree(t)
        #self.sumtree2 = SumTreeslow(np.random, t)
        self.verr = abs(self.g - self.vpred)
        #print(self.g)
        #print(self.vpred)
        for i in range(t):
            self.sumtree.set(i, self.verr[i])
            #self.sumtree2.set(i, self.verr[i])

    def sample(self, q=None):
        #idx2 = self.sumtree2.sample(q)
        idx = self.sumtree.sample(q)
        #if idx != idx2:
        #    import pdb; pdb.set_trace()

        return idx, self.sumtree.get(idx) / self.sumtree.total()

@ray.remote
class ReplayBuffer:
    def __init__(self, max_size):
        self.size = 0
        self.max_size = max_size
        self.current_episode = Episode()
        self.episodes = []
        self.total_nep = 0

    def add(self, s, a, r, sp, t, pi):
        self.current_episode.add(s, a, r, t, pi)
        if t:
            self.current_episode.add(sp, a, r*0, t, pi)
            self.current_episode.done()
            self.add_episode(self.current_episode)
            self.current_episode = Episode()

    def add_episode(self, ep):
        self.size += len(ep.s)
        ep.id = self.total_nep
        self.total_nep += 1
        self.episodes.append(ep)
        while self.size > self.max_size:
            self.size -= len(self.episodes[0].s)
            self.episodes = self.episodes[1:]

    def last_episode_reward(self):
        return self.episodes[-1].r.sum().item()

    def last_episode_verr(self):
        return self.episodes[-1].sumtree.total() / self.episodes[-1].s.shape[0]

    def num_episodes(self):
        return len(self.episodes)

    def get_episodes(self, idx):
        return [self.episodes[i] for i in idx]

    @ray.method(num_return_vals=1)
    def sample_slices(self, n, k):
        tu = time.time()
        eidx = np.random.randint(0, len(self.episodes), n)
        lens = np.int32([len(self.episodes[i].s) for i in eidx])
        t0 = time.time()
        #ts = np.int32(np.random.uniform(0, 1, n) * (lens - k))
        q = np.random.random(size=n)
        r = np.float32([self.episodes[i].sample(q[i]) for i in eidx])
        ts = np.minimum(lens-k, np.int32(r[:, 0]))


        t1 = time.time()
        #r = [
        #    np.stack([getattr(self.episodes[i], a)[t:t+k] for i,t in zip(eidx,ts)], 1)
        #    for a in ('s','a','r','nt','pi', 'g', 'verr')]
        rq = ([np.stack([self.episodes[i].s[t] for i,t in zip(eidx,ts)], 0)] +
             [np.stack([getattr(self.episodes[i], a)[t:t+k] for i,t in zip(eidx,ts)], 1)
              for a in ('a','r','pi', 'g')] + [r[:, 1]])
        t2 = time.time()
        #print(t0-tu, t1-t0, t2-t1)
        return rq


class SumTree:

  def __init__(self, rng, size):
    self.rng = rng
    self.nlevels = int(np.ceil(np.log(size) / np.log(2))) + 1
    self.size = size
    self.levels = []
    for i in range(self.nlevels):
      self.levels.append(np.zeros(min(2**i, size), dtype="float32"))

  def sample(self, q=None):
    q = self.rng.random() if q is None else q
    q *= self.levels[0][0]
    s = 0
    for i in range(1, self.nlevels):
      s *= 2
      if self.levels[i][s] < q and self.levels[i][s + 1] > 0:
        q -= self.levels[i][s]
        s += 1
    return s

  def stratified_sample(self, n):
    # As per Schaul et al. (2015)
    return tint([
        self.sample((i + q) / n)
        for i, q in enumerate(self.rng.uniform(0, 1, n))
    ])

  def set(self, idx, p):
    delta = p - self.levels[-1][idx]
    for i in range(self.nlevels - 1, -1, -1):
      self.levels[i][idx] += delta
      idx //= 2

  def get(self, idx):
      return self.levels[-1][idx]

  def total(self):
      return self.levels[0][0]

if __name__ == '__main__':
    import sys
    cuda = 'cuda' in sys.argv
    device = torch.device('cuda') if cuda else torch.device('cpu')
    gamma = 0.99
    main()
