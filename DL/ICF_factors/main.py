#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    disentangle_rl.policy_rep
    ~~~~~~~~~~~~~~~~~~~~~

    Actor-critic agent trained with a disentanglement cost on MazeBase.
    Only single actions in this version.
    Attribute selector with policy representation

"""
import argparse
import os
import datetime
import ipdb
import math
import numpy as np
from itertools import count, chain
from collections import namedtuple
import scipy.misc
import io

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.nn.init import xavier_uniform as xavier
from scipy import stats

from wrapper_im import Env
from sklearn.manifold import TSNE
#from logger import Logger
from tensorboardX import SummaryWriter
from model import *

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.neighbors.kde import KernelDensity

from utils import * 
import gzip
import pickle

cmap = plt.get_cmap('viridis')
np.set_printoptions(precision=3, suppress=True)

try:
    import better_exceptions
except ImportError:
    pass

# =====
# CL argumemts
# =====
parser = argparse.ArgumentParser(description='PyTorch ICF')
parser.add_argument('--folder', type=str, default='', metavar='F',
                    help='Name of fodler where to save')
parser.add_argument('--name', type=str, default='', metavar='N',
                    help='name of the session')
parser.add_argument('--max-steps', type=int, default=1, metavar='T', 
                    help='Max number of steps')
parser.add_argument('--learning_rate', type=float, default=1e-4, metavar='lr',
                    help='Model learning rate (default: 1e-4)')
parser.add_argument('--var-mix', type=str, default='bilinear', metavar='V',
                    help='Combination type for h and z')
parser.add_argument('--norm-clip', default=50, metavar='c',
                    help='Type of norm for clipping')
parser.add_argument('--entropy_objective', type=float, default=0.0, metavar='ent',
                    help='Entropy objective coefficient for selectivity (default: 0.0)')
parser.add_argument('--phi_space', type=str, default='simplex', metavar='phi',
                    help="Embedding space for the phi-vectors. One of"
                    "['simplex','hypersphere'])")
parser.add_argument('--selectivity_measure', type=str, default='dot_prod',
                    metavar='sel', help="Selectivity calculation.  One of" 
                    "['cosine', 'dot_prod', 'gaussian']")
parser.add_argument('--K', type=int, default=20, metavar='K',
                    help='Bottleneck of the AE')
parser.add_argument('--mc_samples', type=int, default=256, metavar='M',
                    help='Number of sampled phis')
parser.add_argument('--env-size', type=int, default=4, metavar='M',
                    help='Square size of the enrivonment')
parser.add_argument('--env-light', type=int, default=0, metavar='M',
                    help='Whether switches also turn on/off light')
parser.add_argument('--n-switches', type=int, default=3, metavar='M',
                    help='Default number of switches')
parser.add_argument('--epsilon', type=float, default=0.1, metavar='eps',
                    help='Epsilon greedy parameter (default: 0.1)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--cuda', type=int, default=1, metavar='C',
                    help='Using GPU. 1 for yes')
parser.add_argument('--reset', type=int, default=0, metavar='C',
                    help='If we reset every --reset_freq iter')
parser.add_argument('--reset_freq', type=int, default=1000, metavar='C',
                    help='We reset every --reset_freq iter')
parser.add_argument('--log-interval', type=int, default=500, metavar='L',
                    help='interval between training status logs (default: 50)')
parser.add_argument('--log_dir', type=str, 
        default='/data/lisatmp2/thomasva/logs/cleaner_code/', metavar='log',
        help='Base directory for the logs.')
parser.add_argument('--coeff_AE', type=float, default=1e4, metavar='c', help='Coefficient for loss_AE')
parser.add_argument('--coeff_SEL', type=float, default=2e1, metavar='c', help='Coefficient for lossSEL')
parser.add_argument('--coeff_VAL', type=float, default=1e0, metavar='c', help='Coefficient for lossVAL')
parser.add_argument('--coeff_ENT', type=float, default=1e-4, metavar='c', help='Coefficient for lossENT')
parser.add_argument('--coeff_MB', type=float, default=1e-1, metavar='c', help='Coefficient for lossMB')
parser.add_argument('--coeff_MB_pix', type=float, default=0e0, metavar='c', help='Coefficient for lossMB_pix')
parser.add_argument('--trans', type=int, default=1, metavar='s', help='use transition network or not')
parser.add_argument('--pi_encoder', type=int, default=0, metavar='s', help='Use own encoder for pi or not')
parser.add_argument('--nhid_generator', type=int, default=128)
parser.add_argument('--nhid_policy', type=int, default=64)


args = parser.parse_args()
torch.manual_seed(args.seed)
var_mix = args.var_mix
mc_samples = args.mc_samples

args_param = vars(args)
donotprint = ['folder', 'norm_clip', 'mc_samples', 'cuda',
'log_interval', 'env_size', 'n_switches', 'log_dir',
'coeff_AE', 'coeff_VAL', 'coeff_ENT', 'coeff_SEL',
'coeff_MB_pix']

if var_mix == 'bilinear':
    donotprint.append('var_mix')

now = datetime.datetime.now()
name = f'{now.hour}_{now.minute}'
for arg, val in args_param.items():
    if arg not in donotprint:
        name += f'_{arg}={val}'
print(f'==== Name: {name} ====')

USE_CUDA = torch.cuda.is_available() and args.cuda == 1
if USE_CUDA:
    print('Using GPU...')
    dtype = torch.cuda.FloatTensor
    folder = f'{args.log_dir}/{args.folder}/{now.month}_{now.day}/{name}'
else:
    print('Using CPU...')
    dtype = torch.FloatTensor
    folder = f'./logs/{name}'

# Tensorboard stuff
#logger = Logger(folder)
writer = SummaryWriter(log_dir=folder)
directory = folder+'/im/'
if not os.path.exists(directory):
        os.makedirs(directory)

# == Init ==
# Env creation
env = Env(args.env_size, args.n_switches, args.env_light)

state_raw = env.reset()
state, state_tensor = process_obs(state_raw)
# state = np.rollaxis(state, 2, 0)
# state = state[:, ::2, ::2]

K = args.K # Number of latent variables in the bottleneck of the AE
size_phi = K # Dimension of phi
size_z = 3 # Dimension of the noise
size_attr = 1 # dimenstion of the atrribute (if learned)

if var_mix == 'concat':
    var_layer_size = K + size_z
    hphi_layer_size = K + size_phi
elif var_mix == 'bilinear':
    var_layer_size = K * size_z
    hphi_layer_size = K * size_phi
else:
    print('No valid mixing param')

def entangle_var(h, z, typ):
    n_batch = h.size(0)
    if typ == 'concat':
        return torch.cat([h, z], 1)
    elif typ == 'bilinear':
        return torch.bmm(h.unsqueeze(2), z.unsqueeze(1)).view(n_batch, -1)
    else:
        print('No type')
        return

# Noise Generator
def sample_z(n, taille_z):
    return Variable(torch.randn([n,taille_z]).type(dtype))
    #return Variable(torch.randn([n,taille_z]).type(dtype))

square_loss = torch.nn.MSELoss(size_average=True)
huber_loss = torch.nn.SmoothL1Loss(size_average=True)
l1_loss = torch.nn.L1Loss(size_average=True)
relu = nn.ReLU(True)
# ====

# ====
nb_policies = 30 # nuber of policies displayed in some tensorbord fields
tab_dh = np.zeros((nb_policies, K)) # log of visited dh
dh_actions = ['' for k in range(nb_policies)]

def update_dh(dh, action):
    tab_dh[:-1, :] = tab_dh[1:, :]
    tab_dh[-1, :] = dh
    del dh_actions[0]
    dh_actions.append(env.actions[action])

# Networks
ae = Autoencoder(K)
trans = Transition(hphi_layer_size, K)
phiGen = Generator(args, var_layer_size, size_phi)
piProb = Policy(hphi_layer_size, env.action_space.n, args)

if USE_CUDA:
    ae.cuda()
    phiGen.cuda()
    piProb.cuda()
    trans.cuda()

all_param = chain(ae.parameters(), phiGen.parameters(), piProb.parameters(), trans.parameters())

# Attribute Selector.
if args.selectivity_measure == 'attrsel':
    attrSel = Attr_Selector(hphi_layer_size, size_attr)
    if USE_CUDA:
        attrSel.cuda()
    all_param = chain(all_param, attrSel.parameters())

# Policy Encoder.
if args.pi_encoder == 1:
    piEnc = PolicyEncoder(K)
    if USE_CUDA:
        piEnc.cuda()
    all_param = chain(all_param, piEnc.parameters())

optim_all = optim.Adam(all_param, lr=args.learning_rate)
optim_ae = optim.Adam(ae.parameters(), lr=args.learning_rate, weight_decay=0e-3)
optim_phi = optim.Adam(phiGen.parameters(), lr=args.learning_rate)
optim_pi = optim.Adam(piProb.parameters(), lr=args.learning_rate)

# Things used for plotting in tensorboad
fixed_z = sample_z(nb_policies, size_z)
tsne_z = sample_z(2000, size_z)
n_histo = mc_samples
histo_z = sample_z(n_histo, size_z)
histo_h =  torch.zeros(n_histo, K)
histo_dh = torch.zeros(n_histo, K)
phiphi = torch.zeros(n_histo, n_histo)
phidh = torch.zeros(n_histo, n_histo)
dhdh = torch.zeros(n_histo, n_histo)
histo_act = ['init' for i in range(n_histo)]
histo_act_id = np.array([None for i in range(n_histo)])
histo_states = np.zeros((2000, 64, 64, 3))
i_histo = 0
j_histo =  0
s_histo = 2000
phi_grid = np.zeros((args.env_size,args.env_size,mc_samples,K))
phi_grid_probs = np.zeros((args.env_size,args.env_size,mc_samples,env.action_space.n))


# init
sigma = 1.0 * np.sqrt(K)  # hyperparam for gaussian kernel
num_played = 0
best_reward_ra = 0

def get_real_attr(z_mc, attributes):
    # Id of the attributes looked at for  each phi
    torch_attributes = torch.LongTensor(attributes).cuda()
    id_attr = (z_mc[:,0].unsqueeze(1) / Variable((torch.ones(1,1)/len(env.attributes)).type(dtype))).long()

    res = torch.index_select(torch_attributes, 0, id_attr.data.squeeze())
    return Variable(res).type(dtype).unsqueeze(1)


def plot_phi_grid():
    if K!=2:
        pca = PCA(n_components=2)
        phi_grid_pca = pca.fit_transform(phi_grid.reshape((args.env_size*args.env_size*mc_samples,K))).reshape((args.env_size,args.env_size,mc_samples,2))
    else:
        phi_grid_pca = phi_grid
    f, ax = plt.subplots(args.env_size,args.env_size,sharex=True,sharey=True,figsize=(12,12))
    for x in range(args.env_size):
        for y in range(args.env_size):
            X,Y = phi_grid_pca[x,y,:,0], phi_grid_pca[x,y,:,1]
            try:
                sns.kdeplot(X, Y, cmap="Blues", shade=True, ax=ax[x,y])
            except np.linalg.linalg.LinAlgError as e:
                print(e)
            colors = phi_grid_probs[x,y].dot(np.arange(1,env.action_space.n+1)[:,None]).ravel()
            ax[x,y].scatter(X, Y, c=colors, s=1, marker="o", linewidth=0.5, cmap='cubehelix')

    f.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(f'{folder}/im/{i_episode}_kde_phi_grid.png')
    im = plt.imread(f'{folder}/im/{i_episode}_kde_phi_grid.png')
    writer.add_image('KDE_phi_grid', im, i_episode)

for i_episode in count(1):
    # Reset the environment every n step
    # if i_episode % 1 == 0:
    #     state = env.reset()

    log_probs_traj = 0
    log_probs_mc_traj = 0
    loss_entropy = 0
    loss_AE = 0
    h_penalty = 0
    list_s, list_h, list_hpi, list_a = [], [], [], []

    for t in range(args.max_steps): 
        #sigma = (1-1e-5)*sigma +  .1 * np.sqrt(K)
        i_histo = (i_histo + 1) % n_histo
        j_histo = (j_histo + 1) % s_histo

        # 1. Generate a trajectory
        # ================================
        state, state_tensor = process_obs(state_raw)
        histo_states[j_histo] = state
        sv = Variable(state_tensor.type(dtype))
        reconstruction, h = ae(sv)
        h_rep = h.expand((mc_samples, h.size(1)))
        loss_AE += square_loss(reconstruction, Variable(state_tensor.type(dtype)))
        if args.pi_encoder == 1:
            h_pi = piEnc(sv)
        else:
            h_pi = h

        if t == 0:
            # Generate the phi
            reco_0 = reconstruction
            h_0 = h
            h0pi = h_pi
            z_mc = sample_z(mc_samples, size_z)
            phi_mc = phiGen(entangle_var(h_rep, z_mc, var_mix)) #n x K
            env.observe()
            attr_0 = list(env.attributes)

        # Policy input.
        input_pi = entangle_var(h_pi.expand((mc_samples, K)), h0pi+phi_mc, var_mix)
        probs_mc, V_mc = piProb(input_pi)

        # plotting parenthesis
        phi_grid[env.attributes[0],env.attributes[1]] = phi_mc.data.cpu().numpy()
        phi_grid_probs[env.attributes[0],env.attributes[1]] = probs_mc.data.cpu().numpy()

        # Epsilon-Greedy policy.
        probs = probs_mc[0, :].unsqueeze(0)
        probs = probs * (1 - args.epsilon) + args.epsilon/env.action_space.n
        probs = probs/probs.sum()
        action = probs.multinomial()
        act_id = action.data[0,0]
        probs_a = probs[:, act_id]

        # Action entropy.
        loss_entropy -= torch.mean((probs_mc*torch.log(probs_mc)).sum(1))

        # Get prob of the action taken
        probs_mc_a = (probs_mc[:, act_id]).unsqueeze(1)
        probs_mc_a = probs_mc_a#.clamp(min=1e-4)

        #if i_episode > 1000 and act_id == 4:
        #    ipdb.set_trace()

        log_probs_traj += torch.log(probs_a)
        log_probs_mc_traj += torch.log(probs_mc_a)

        new_state_raw, _, done, played = env.step(act_id)
        # if done:
        #     env.reset()
        list_h.append(h)
        list_hpi.append(h_pi)
        list_s.append(state)
        list_a.append(act_id)
        # if not played and i_episode > 10000:
        #     print(f'Proba {probs}')
        #     print(f'Action {act_id}')
        #     env.render_in_console()

        new_state, new_state_tensor = process_obs(new_state_raw)
        _, hh = ae(Variable(new_state_tensor.type(dtype)))
        hh_pi = hh#piEnc(sv)
        state_raw = new_state_raw
    list_h.append(hh)
    list_hpi.append(hh_pi)
    list_s.append(new_state)
    dh = hh-h_0
    histo_dh[i_histo, :] = dh.data
    histo_h[i_histo, :] = h.data.squeeze()
    histo_act[i_histo] = env.actions[act_id]
    histo_act_id[i_histo] = act_id

    dh_rep = dh.expand((mc_samples, K))
    hh_rep = hh.expand((mc_samples, h.size(1)))
    h0_rep = h_0.expand((mc_samples, K))
    weight_is = torch.exp(log_probs_mc_traj - log_probs_traj).detach()
    weight_is = weight_is/weight_is.sum()

    loss_entropy = (loss_entropy)/args.max_steps
    loss_AE = loss_AE/args.max_steps

    # Form of latent model.
    if args.trans == 1:
        input_trans = entangle_var(h0_rep, phi_mc, var_mix)#.detach()
        h_hat = trans(input_trans)
    else:
        h_hat = (phi_mc + h_0)#.view(1, -1, 1, 1)

    # Model based predictions in latent and/or pixel space.
    mb_reco = ae.decoder(h_hat.view(mc_samples, K, 1, 1))
    if args.coeff_MB_pix:
        loss_MB_AE_pix = square_loss(mb_reco[0], Variable(new_state_tensor).type(dtype))
    else:
        loss_MB_AE_pix = Variable(torch.zeros(1).type(dtype))
    #loss_MB_AE = torch.sum((h_hat-hh)**2)
    loss_MB_AE = torch.sum((h_hat - hh_rep.detach())**2*weight_is)
    #loss_MB_AE = torch.sum((h_hat-hh.detach())**2/(1e-5+torch.abs(h_hat)))

    #numerator = np.exp(-numerator)
    #numerator = attrSel(dh_rep, phi_mc)

    dh_norm = torch.norm(dh, p=2, dim=1)

    if args.selectivity_measure == 'cosine':
        #assert args.phi_space == 'hypersphere' or args.phi_space == 'hypercube'
        if (dh_norm.data == 0).all():
            numerator = Variable(torch.zeros(mc_samples, 1)).type(dtype)
        else:
            #cossim = phi_mc.mm(dh.t()) / (1e-6+dh_norm)
            cossim = dh_norm + phi_mc.mm(dh.t())
            numerator = cossim
    elif args.selectivity_measure == 'dot_prod':
        attr_h = phi_mc.mm(h_0.t())
        attr_hh = phi_mc.mm(hh.t())
        numerator = torch.abs(attr_hh - attr_h)
    elif args.selectivity_measure == 'relu_dot_prod':
        attr_h = phi_mc.mm(h_0.t())
        attr_hh = phi_mc.mm(hh.t())
        numerator = relu(attr_hh - attr_h)
    elif args.selectivity_measure == 'gaussian':
        numerator = gaussianSim(phi_mc, dh_rep, sigma=sigma)
    elif args.selectivity_measure == 'attrsel':
        attr_h = attrSel(entangle_var(h_rep, phi_mc, var_mix))
        attr_hh = attrSel(entangle_var(hh_rep, phi_mc, var_mix))
        numerator = (attr_hh - attr_h)**2
    else:
        raise NotImplementedError

    # Vector of ||dh - phi_i||^2
    #norm_vec =  torch.sum((dh_rep - phi_mc)**2, 1)
    #numerator =  torch.sum((dh_rep - phi_mc)**2, 1)

    # Selectivity reward.
    denominator = (1e-5+numerator).mean()
    r_mc = torch.log(1e-3 + numerator/ denominator)

    # # Conditionally add the loss_entropy only for non-zero dh.
    # if args.entropy_objective > 0:
    #     h_norm = torch.norm(dh, p=1, dim=1)
    #not_played_penalty = 0
        #not_played_penalty = -1
    #     else:
    #         r_mc = r_mc - args.entropy_objective * torch.log(denominator)

    # Importance sampling
    #r_mc = r_mc #+ not_played_penalty

    if not (dh_norm.data == 0).all():
        num_played += played
    # else:
    #     mean = r_mc.mean().detach()
    #     std = torch.sqrt(1e-7+r_mc.var()).detach()
    #     r_mc = (r_mc - mean)/std

    r_mc_is = r_mc * weight_is
    # ================================

    rewards = r_mc_is
    rewards = rewards
    reward = rewards.sum()

    # 3. Update parameters
    # # Value error
    #value_loss = square_loss(V_mc*weight_is, r_mc.detach()*weight_is).mean()
    value_loss = torch.sum((V_mc-r_mc.detach())**2*weight_is)
    
    # Disentanglement loss
    # Autoencoder loss
    # loss_AE = square_loss(reconstruction, Variable(state_tensor.type(dtype)))
    # Penalty on magnitude of dh
    mseloss_dh = square_loss(dh, Variable(torch.zeros(h.size())).type(dtype))
    #mseloss_dh = (mseloss_dh-1/K)**2

    rewards_adv = rewards - V_mc * weight_is
    rewards_adv = rewards_adv

    lossSEL = - (rewards + rewards_adv.detach() * log_probs_mc_traj).sum() 
    loss_h = torch.sqrt(torch.sum(dh**2)/K)
    loss_tot = args.coeff_AE * loss_AE + \
               args.coeff_VAL * value_loss + \
               args.coeff_ENT * loss_entropy + \
               args.coeff_MB * loss_MB_AE + \
               args.coeff_MB_pix * loss_MB_AE_pix

    loss_tot_plot = loss_tot.clone() - args.coeff_SEL * reward
    loss_tot = loss_tot + args.coeff_SEL * lossSEL

    update_dh(dh.data.cpu().numpy(), act_id)

    # Optimizations.
    optim_all.zero_grad()
    loss_tot.backward()
    torch.nn.utils.clip_grad_norm(ae.parameters(), args.norm_clip)
    torch.nn.utils.clip_grad_norm(phiGen.parameters(), args.norm_clip)
    torch.nn.utils.clip_grad_norm(piProb.parameters(), args.norm_clip)
    optim_all.step()
    # if i_episode % 2100 < 2000:
    #     optim_ae.step()
    # else:
    #     optim_phi.step()
    #     optim_pi.step()


    if args.reset == 1 and (done or i_episode % args.reset_freq == 0):
        state = env.reset()
        if i_episode % args.log_interval == 0:
            plot_phi_grid()
        phi_grid_probs *= 0
        phi_grid *= 0

    # Update plots
    if i_episode<=1:
        lossae_ra =  loss_AE.data.cpu().numpy()
        reward_ra = rewards[0].mean().data.cpu().numpy()
        lossvl_ra =  value_loss.data.cpu().numpy()
        lossh_ra =  loss_h.data.cpu().numpy()
        losstot_ra =  loss_tot_plot.data.cpu().numpy()
        #losspilogsel_ra = pilogsel_tab.sum()
    else:
        lossae_ra = lossae_ra*0.99 + loss_AE.data.cpu().numpy()*0.01
        reward_ra = reward_ra*0.99 + rewards[0].mean().data.cpu().numpy()*0.01
        lossvl_ra = lossvl_ra*0.99 + value_loss.data.cpu().numpy()*0.01
        lossh_ra = lossh_ra*0.99 + loss_h.data.cpu().numpy()*0.01
        losstot_ra = losstot_ra*0.99 + loss_tot_plot.data.cpu().numpy()*0.01
        #losspilogsel_ra = losspilogsel_ra*0.99 + pilogsel_tab.sum()*0.01

    if i_episode % (args.log_interval/10) == 0:
        print(i_episode)
        print(np.column_stack((lossae_ra, reward_ra, lossvl_ra, lossh_ra, abs(losstot_ra))))
        print(attr_0)
        print(env.attributes)
        env.render_in_console()

        info = {
            'loss_AE': args.coeff_AE * loss_AE.data[0],
            'loss_MB_AE': args.coeff_MB * loss_MB_AE.data[0],
            'loss_MB_AE_pix': args.coeff_MB_pix * loss_MB_AE_pix.data[0],
            'loss_value': args.coeff_VAL * value_loss.data[0],
            'loss_entropy': args.coeff_ENT * loss_entropy.mean().data[0],
            'loss_tot': loss_tot_plot.data[0],
            'Reward': args.coeff_SEL * reward.data[0],
            'Reward_behavior': r_mc[0].data[0],
            'Advantage': rewards_adv.sum().data[0],
            'weight_is.sum()': weight_is.sum().data[0],
            'weight_is.max()': weight_is.max().data[0],
            '||dh||/sqrt(k)': loss_h.data[0],
            'denominator Z': denominator.data[0],
            'ratio played': num_played/(args.log_interval/10),
        }
        num_played = 0

        for tag, value in info.items():
            writer.add_scalar(tag, value, i_episode)

    if i_episode % args.log_interval == 0:
        # (2) Log values and gradients of the parameters (histogram)
        nets = {'AE':ae, 'G_phi': phiGen, 'Pi': piProb}
        for net_name, net in nets.items():
            for tag, value in net.named_parameters():
                tag = tag.replace('.', '/')
                try:
                    writer.add_histogram(net_name+'/'+tag, to_np(value), i_episode)
                    writer.add_histogram(net_name+'/'+tag+'/grad',
                            to_np(value.grad), i_episode)
                except ValueError:
                    print(f'Error at {net_name}/{tag}')


        #p_emb = phiGen(entangle_var(h_0.expand(nb_policies, K), fixed_z, var_mix))
        #phi_mc

        #####
        for i, hi in enumerate(list_hpi):
            input_pi = entangle_var(hi.expand(mc_samples, K), h0pi+phi_mc, var_mix)
            proba, _ = piProb(input_pi)
            pi_tab = proba[:nb_policies,:].data.cpu().numpy()
            #print(i, pi_tab)
            #writer.heatmap_summary(f'proba_{i}', np.copy(pi_tab), i_episode)
            mat = np.copy(pi_tab)
            rgba_img = cmap(mat)
            rgb_img = np.delete(rgba_img, 3, 2)
            scipy.misc.imsave(f'{folder}/im/{i_episode}_proba_{i}.png', rgb_img)
            im = plt.imread(f'{folder}/im/{i_episode}_proba_{i}.png')
            writer.add_image(f'proba_{i}', im, i_episode)

        p_emb_tab = phi_mc.data.cpu().numpy()

        info = {'phi_values': p_emb_tab[:nb_policies], 'dh': tab_dh[:nb_policies]}
        for tag, heatmaps in info.items():
            mat = np.copy(heatmaps)
            rgba_img = cmap(mat)
            rgb_img = np.delete(rgba_img, 3, 2)
            scipy.misc.imsave(f'{folder}/im/{i_episode}_{tag}.png', rgb_img)
            im = plt.imread(f'{folder}/im/{i_episode}_{tag}.png')
            writer.add_image(f'proba_{i}', im, i_episode)
            #logger.heatmap_summary(tag, heatmaps, i_episode)
        #####


        # viz.heatmap(pi_tab, win=win_pi, opts=opts_pi)
        # viz.heatmap(p_emb_tab, win=win_p, opts=opts_p)
        # viz.line(
        #     X=np.column_stack((i_episode, i_episode, i_episode, i_episode, i_episode, i_episode)),
        #     Y=np.column_stack((lossae_ra, -reward_ra, lossvl_ra, lossh_ra,
        #         abs(losstot_ra), pdh)),
        #     win=win_loss,
        #     update='append'
        # )
        # opts_dh = dict(title='dh '+name, rownames=dh_actions)
        # viz.heatmap(tab_dh, opts=opts_dh, win = win_dh)

        sort_i = np.argsort(histo_act)
        sort_act_init = [histo_act[i] for i in sort_i]
        sort_act = []
        sort_act.append(sort_act_init[0])

        for i, el in enumerate(sort_act_init):
            if i>0:
                lab = '' if sort_act[i-1]==el else el
                sort_act.append(lab)

        # Pass the raw state directly to policy.

        #policy_input = nn.Linear( hphi_layer_size)



        histo_phi = phiGen(entangle_var(h_0.expand(n_histo, K), histo_z, var_mix))
        histo_probs, _ = piProb(entangle_var(h_0.expand(n_histo, K), histo_phi, var_mix))
        histo_phi = histo_phi.data
        # taken_action = np.argmax(histo_probs[0].data.cpu().numpy(), 1)
        # sort_a = np.argsort(taken_action)

        # phidh =gaussianMat(histo_phi, histo_dh, sigma=sigma)
        # phidh = phidh / phidh.mean()
        # phiphi=gaussianMat(histo_phi, histo_phi, sigma=sigma)
        # dhdh = gaussianMat(histo_dh, histo_dh, sigma=sigma)
        #
        # phidh =permute_ind(phidh, sort_a, sort_i)
        # phiphi=permute_ind(phiphi, sort_a, sort_a)
        # dhdh = permute_ind(dhdh, sort_i, sort_i)
        #
        # info = {'sel':phidh, 'phiphi': phiphi, 'dhdh': dhdh}
        # for id_name, mat in info.items():
        #     logger.heat_plot(id_name, i_episode, mat)


        sns_histo_phi = histo_phi.cpu().numpy()
        sns_histo_dh = histo_dh.cpu().numpy()
        sns_histo_h = histo_h.cpu().numpy()
        if K!=2:
            pca = PCA(n_components=2)
            sns_histo_dh = pca.fit_transform(sns_histo_dh)
            #sns_histo_phi = pca.transform(sns_histo_phi)
            sns_histo_phi = pca.fit_transform(sns_histo_phi)
            sns_histo_h = pca.fit_transform(sns_histo_h)
        # ====== phi ========
        sns_plot = sns.jointplot(sns_histo_phi[:, 0], sns_histo_phi[:,1], kind='kde', stat_func=None)
        colors_phi = histo_probs.data.cpu().mm(torch.arange(1, env.action_space.n+1).unsqueeze(1))
        colors_phi = colors_phi.numpy().ravel()
        sns_plot.plot_joint(plt.scatter, c=colors_phi, s=10, linewidth=1, marker="o", cmap='cubehelix')
        sns_plot.savefig(f'{folder}/im/{i_episode}_kde_phi.png', bbox_inches='tight')   # save the figure to file

        im = plt.imread(f'{folder}/im/{i_episode}_kde_phi.png')
        writer.add_image(f'KDE_phi', im, i_episode)

        # ====== phi grid =====
        # plot_phi_grid()


        # ====== dh ========
        mat_act = np.array(histo_act)

        # dh_actions = ['down', 'left', 'pass',  'up', 'right']
        # dh_cm = ['Reds', 'Blues', 'BuGn', 'Purples', 'Greens']
        # for j in range(5):
        #     indices = (mat_act == dh_actions[j])
        sns_plot = sns.jointplot(sns_histo_dh[:, 0], sns_histo_dh[:,1], kind='kde', stat_func=None)
        sns_plot.plot_joint(plt.scatter, c=histo_act_id, s=10, linewidth=1, marker="o", cmap='cubehelix')
        sns_plot.savefig(f'{folder}/im/{i_episode}_kde_dh.png', bbox_inches='tight')   # save the figure to file

        im = plt.imread(f'{folder}/im/{i_episode}_kde_dh.png')
        writer.add_image(f'KDE_dh', im, i_episode)

        # ====== h ========
        fig = plt.figure()
        sns_plot = sns.jointplot(sns_histo_h[:, 0], sns_histo_h[:,1], stat_func=None)
        sns_plot.savefig(f'{folder}/im/{i_episode}_kde_h.png', bbox_inches='tight')   # save the figure to file
        plt.close('all')

        im = plt.imread(f'{folder}/im/{i_episode}_kde_h.png')
        writer.add_image(f'KDE_h', im, i_episode)

    # except Exception as e:
    #     print('Error', e)
    #     pass
    #
        im1 = list_s[0]
        im2 = torch2np(reco_0.squeeze().data.cpu())/255
        im_final = new_state
        im_final_mb = torch2np(mb_reco[0].squeeze().data.cpu())/255
        writer.add_image(f'First_state', im1, i_episode)
        writer.add_image(f'First_state_reconstruction', im2, i_episode)
        for ind, im_s in enumerate(list_s):
            writer.add_image(f'state_{ind}', im_s, i_episode)
        writer.add_image(f'Final_state', im_final, i_episode)
        writer.add_image(f'Final_state_prediction', im_final_mb, i_episode)

        #15 phi final MB
        # writer.add_image(f'phi_predicted_MB', [torch2np(mb_reco[i].squeeze().data.cpu())/255 for i in range(15)], i_episode)
        #
        # #assert np.any(np.isnan(pts_color)), ipdb.set_trace()
        # bidule=TSNE()
        # prob_out = bidule.fit_transform(prob_in.data.cpu().numpy())
        # if K>2:
        #     p_out = bidule.fit_transform(p_in.data.cpu().numpy())
        # else:
        #     p_out = p_in.data.cpu().numpy()
        # viz.scatter(p_out, pts_color, win=win_tsne, opts=dict(title=f't-SNE p embedding {i_episode}. ' + name))
        # viz.scatter(prob_out, pts_color, win=win_tsne2, opts=dict(title=f't-SNE probs {i_episode}. '+name))
    if i_episode % 2000 == 0:
        im_labels = ae.decoder(h_hat.view(mc_samples, K, 1, 1)).cpu()
        writer.add_embedding(histo_phi, global_step=i_episode, label_img=im_labels.data)
        print('Added embedding!')
    if i_episode % 10000 == 0 and reward_ra > best_reward_ra:
        best_reward_ra = reward_ra
        with gzip.open(f'{folder}/params.pkl.gz','wb') as f:
            pickle.dump([unwrap(i) for i in all_param] , f)

    if i_episode == 500000:
        break


    '''
    multistep: the terminal reward is, you changed only one factor at one timestep
    '''
