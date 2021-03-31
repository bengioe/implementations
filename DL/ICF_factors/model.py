import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import torch.nn.init as weight_init
import ipdb

nc = 3
ngf = 64
ndf = 64
lrelu = nn.LeakyReLU(0.1)


# class ipdb_layer(nn.Module):
#     def __init__(self):
#         super(ipdb_layer, self).__init__()
#         return
#
#     def forward(self, x):
#         ipdb.set_trace()
#         return x

class Autoencoder(nn.Module):
    def __init__(self, K):
        super(Autoencoder, self).__init__()
        self.K = K
        self.encoder = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, self.K, 4, 1, 0, bias=False),
            nn.Tanh()
            )
        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     K, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            # state size. (nc) x 64 x 64
            nn.Sigmoid()
            )
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #         m.weight.data.normal_(0, 1e-4)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.normal_(1.0, 0.02)
        #         m.bias.data.zero_()



    def forward(self, input):
        h = self.encoder(input).view(input.size(0),-1)
        r = self.decoder(h.view(input.size(0), -1, 1, 1))
        return r, h


class Transition(nn.Module):
    """
    Attribute selector
    Not used in this version
    :input: h and phi
    :output: value of attribute corresponding to phi in h
    """
    def __init__(self, n_in, n_out):
        super(Transition, self).__init__()
        self.fc1 = nn.Linear(n_in, 64)
        self.fc2 = nn.Linear(64, n_out)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                n_in = m.in_features
                n_out = m.out_features
                m.weight.data.normal_(0, math.sqrt(2. / (n_in+n_out)))

    def forward(self, vec):
        vec = self.fc1(vec)
        vec = lrelu(vec)
        vec = self.fc2(vec)
        vec = F.tanh(vec)
        return vec


class Generator(nn.Module):
    """
    Genenerator Network phi-vectors.
    :input: h and z
    :output: a sample of phi
    """
    def __init__(self, args, n_in, n_out):
        super(Generator, self).__init__()
        self.args = args
        # TODO(liamfedus):  Test weight normalization.
        # self.fc1 = nn.utils.weight_norm(nn.Linear(n_in, 500))
        # self.fc2 = nn.utils.weight_norm(nn.Linear(500, n_out))
        self.fc1 = nn.Linear(n_in, args.nhid_generator)
        self.bn1 = nn.BatchNorm1d(args.nhid_generator)
        #self.drop = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(args.nhid_generator, n_out)

        if self.args.phi_space == 'simplex_tanh':
            self.fc22 = nn.Linear(args.nhid_generator, 1)
        #self.bn2 = nn.BatchNorm1d(size_p)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                n_in = m.in_features
                n_out = m.out_features
                m.weight.data.normal_(0, math.sqrt(2. / (n_in+n_out)))

    def forward(self, x):
        #x = self.drop(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = lrelu(x)
        #x = self.drop(x)
        p = self.fc2(x)
        #p = lrelu(p)
        #p = self.bn2(p)
 
        if self.args.phi_space == 'hypersphere':
            p = F.tanh(p)
            norm = p.norm(2, 1)
            p = p / (1e-6+norm.unsqueeze(1))
        elif self.args.phi_space == 'simplex':
            p = F.softmax(p)
        elif self.args.phi_space == 'hypercube':
            p = F.tanh(p)
        elif self.args.phi_space == 'simplex_tanh':
            scal = self.fc22(x)
            p = F.softmax(p)*F.tanh(scal)
        else:
            raise NotImplementedError
        return p



class Policy(nn.Module):
    """
    Pi network.
    :input: h and phi
    :output: probability over actions
    """
    def __init__(self, n_in, n_act, args):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(n_in, args.nhid_policy)
        self.bn1 = nn.BatchNorm1d(args.nhid_policy)
        self.value_head = nn.Linear( args.nhid_policy, 1)
        self.action_head = nn.Linear(args.nhid_policy, n_act)
        #self.action_head2 = nn.Linear(32, n_act)


        self.latent_states = []
        self.saved_actions = []
        self.log_probs_act = []

        for m in self.modules():
            if isinstance(m, nn.Linear):
                n_in = m.in_features
                n_out = m.out_features
                m.weight.data.normal_(0, math.sqrt(2. / (n_in+n_out)))

    def forward(self, x):
        x = self.affine1(x)
        x = self.bn1(x)
        x = lrelu(x)
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores), state_values

    def purge(self):
        self.latent_states = []
        self.saved_actions = []
        self.log_probs_act = []


class PiLSTM(nn.Module):
    """
    Pi network.
    :input: h and phi
    :output: probability over actions
    """
    def __init__(self, n_in, n_act):
        super(PiLSTM, self).__init__()
        self.lstm_actions = nn.LSTMCell(n_in, 32)
        self.affine2 = nn.Linear(32, n_act)


        # nn.init.orthogonal()
        # nn.init.orthogonal()
        # self.lstm_actions.weight_hh.data.normal_(0, 1e-2)
        # self.lstm_actions.weight_ih.data.normal_(0, 1e-2)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                n_in = m.in_features
                n_out = m.out_features
                m.weight.data.normal_(0, math.sqrt(2. / (n_in+n_out)))

    def forward(self, x, hidden):
        # x = self.affine1(x)
        # x = self.bn1(x)
        # x = lrelu(x)
        hx, cx = self.lstm_actions(x, hidden)
        v = self.affine2(hx)
        #x = self.affine3(x)
        #state_values = Variable(torch.ones(1,1).type(dtype))
        return F.softmax(v), (hx, cx)


class ValueFunction(nn.Module):
    """
    Pi network.
    :input: h and phi
    :output: probability over actions
    """
    def __init__(self, n_in):
        super(ValueFunction, self).__init__()
        self.affine1 = nn.Linear(n_in, 256)
        self.bn1 = nn.BatchNorm1d(256)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                n_in = m.in_features
                n_out = m.out_features
                m.weight.data.normal_(0, math.sqrt(2. / (n_in+n_out)))

    def forward(self, x):
        x = self.affine1(x)
        x = self.bn1(x)
        x = lrelu(x)
        return state_values


# Not used in this version
class Attr_Selector(nn.Module):
    """
    Attribute selector
    Not used in this version
    :input: h and phi
    :output: value of attribute corresponding to phi in h
    """
    def __init__(self, n_in, n_out):
        super(Attr_Selector, self).__init__()
        # self.fc1 = nn.Linear(n_in, 32)
        # self.fc2 = nn.Linear(32, n_out)
        self.fc1 = nn.Linear(n_in, n_out)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                n_in = m.in_features
                n_out = m.out_features
                m.weight.data.normal_(0, math.sqrt(2. / (n_in+n_out)))

    def forward(self, vec):
        vec = self.fc1(vec)
        # vec = lrelu(vec)
        # vec = self.fc2(vec)
        vec = F.tanh(vec)
        return vec

class Generator_z(nn.Module):
    """
    Generator of noise network.
    :input: h
    :output: a sample of z
    """
    def __init__(self, n_in, n_out):
        super(Generator_z, self).__init__()
        self.fc1 = nn.Linear(n_in, 32)
        self.fc21 = nn.Linear(32, n_out)
        self.fc21 = nn.Linear(32, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                n_in = m.in_features
                n_out = m.out_features
                m.weight.data.normal_(0, math.sqrt(2. / (n_in+n_out)))

    def forward(self, x, eps=None):
        x = self.fc1(x)
        x = lrelu(x)
        mu = self.fc21(x)
        logvar = self.fc21(x)
        std = logvar.mul(0.5).exp_()
        if eps is None:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu), mu, logvar

def KL_loss(x, mu, logvar):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return KLD

class PolicyEncoder(nn.Module):
    """
    Pi network.
    :input: h and phi
    :output: probability over actions
    """
    def __init__(self, K):
        super(PolicyEncoder, self).__init__()
        self.encoder = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, K, 4, 1, 0, bias=False),
            #nn.Tanh()
            )


        for m in self.modules():
            if isinstance(m, nn.Linear):
                n_in = m.in_features
                n_out = m.out_features
                m.weight.data.normal_(0, math.sqrt(2. / (n_in+n_out)))

    def forward(self, x):
        return self.encoder(x).view(x.size(0),-1)


