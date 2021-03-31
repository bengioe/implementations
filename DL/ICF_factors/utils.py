import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#import imageio
#import ipdb
import glob
import re

import torch
from torch.autograd import Variable
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image

# Utils functions to swap between PIL, numpy and torch data
# =====
torch2pil = ToPILImage()
pil2torch = ToTensor()
torch2np = lambda x: np.array(torch2pil(x))
np2pil = lambda x: Image.fromarray((255*x).astype(np.uint8))
pil2np = lambda x: np.array(x)
#np2torch = lambda x: pil2np(torch2pil(x))

def unwrap(x):
    if isinstance(x, Variable) or isinstance(x, torch.Tensor):
        if isinstance(x, Variable):
            x = x.data
            x = x.cpu().numpy()
    if hasattr(x,'shape') and np.prod(x.shape) == 1:
        x = float(x)
    return x


def process_obs(s):
    rsz_im = np2pil(s)
    rsz_im = rsz_im.resize((64, 64), Image.ANTIALIAS)
    state = pil2np(rsz_im)/255
    state_tensor = pil2torch(rsz_im).unsqueeze(0)
    return state, state_tensor

def gaussianSim(phi, dh, sigma=1):
    norm2 = (phi - dh).norm(2, 1, keepdim=True)
    res = torch.exp(-norm2**2/(2*sigma**2))
    return res

def tSim(phi, dh, sigma=1):
    norm2 = (phi - dh).norm(2, 1, keepdim=True)
    res = 1/(1+norm2**2/(2*sigma**2))
    return res

def gaussianMat(phi, dh, sigma=1):
    phi2 = phi.norm(2, 1, keepdim=True).cpu().numpy()
    dh2 =  dh.norm(2, 1, keepdim=True).cpu().numpy()
    phi = phi.cpu().numpy()
    dh = dh.cpu().numpy()
    K = phi2**2 + dh2.T**2
    K = K - 2*phi@dh.T
    return np.exp(-K/(2*sigma**2)).clip(max=10)

def tMat(phi, dh, sigma=1):
    phi2 = phi.norm(2, 1, keepdim=True).cpu().numpy()
    dh2 =  dh.norm(2, 1, keepdim=True).cpu().numpy()
    phi = phi.cpu().numpy()
    dh = dh.cpu().numpy()
    K = phi2**2 + dh2.T**2
    K = K - 2*phi@dh.T
    return 1/(1+K/(2*sigma**2)).clip(max=10)

def sample_u(n=1):
    u = torch.ones(n, 1)
    return Variable(u.type(dtype))
    # sign = np.random.randint(2)
    # if sign == 1:
    #     u = np.random.rand()+eps
    # elif sign == 0:
    #     u = - (np.random.rand()+eps)
    # return torch.ones(1)*u
    u = torch.bernoulli(torch.ones(n, 1)/2)
    u = 2*u-1
    return Variable(u.type(dtype))

def permute_ind(mat, ind, ind2):
    mat = mat[:, ind2]
    mat = mat[ind, :]
    return mat
 
def to_np(x):
    return x.data.cpu().numpy()

    #return Variable(torch.randn([n,taille_z]).type(dtype))

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


def make_big_fig(name):
    keywords = ['sel', 'phiphi', 'dhdh', 'kde_phi', 'kde_dh', 'kde_h']
    files = [[] for i in range(len(keywords))]
    for i, word in enumerate(keywords):
        files[i] = natural_sort(glob.glob(f'./logs/{name}/im/*{word}.png'))
    length = len(files[0])

    titles = ['Selectivity', r'$A(\phi, \phi)$', r'$A(dh, dh)$', r'KDE $\phi$',  \
            r'KDE $dh$', r'KDE $h$']
    print(f'Writing figures')
    for i in tqdm(range(length)):
        i_episode = (i+1)*200
        fig, ax = plt.subplots(2, 3)
        for j, tit in enumerate(titles):
            im = plt.imread(files[j][i])
            plt.subplot(2,3,j+1)
            plt.imshow(im)
            plt.axis('off')
            plt.title(tit)
        plt.suptitle(f'iter {i_episode}')
        fig.savefig(f'./logs/{name}/im/{i_episode}_fullfig.jpg', bbox_inches='tight')
        plt.close(fig)


# def make_mid_fig(name):
#     keywords = ['kde_phi', 'kde_dh']
#     files = [[] for i in range(len(keywords))]
#     for i, word in enumerate(keywords):
#         files[i] = natural_sort(glob.glob(f'./logs/{name}/im/*{word}.png'))
#     length = len(files[0])
#
#     titles = [r'$\phi$',  r'$dh$']
#     print(f'Writing figures')
#     for i in tqdm(range(length)):
#         i_episode = (i+1)*200
#         fig, ax = plt.subplots(1, 2)
#         for j, tit in enumerate(titles):
#             im = plt.imread(files[j][i])
#             plt.subplot(1,2,j+1)
#             plt.imshow(im)
#             plt.axis('off')
#             plt.title(tit)
#         plt.suptitle(f'iter {i_episode}')
#         fig.savefig(f'./logs/{name}/im/{i_episode}_fullfig.jpg', bbox_inches='tight')
#         plt.close(fig)
#
# def make_gif(name):
#     filenames = natural_sort(glob.glob(f'./logs/{name}/im/*fullfig.jpg'))
#     print(f'Writing gif')
#     with imageio.get_writer(f'./logs/{name}/im/fullfig.gif', mode='I') as writer:
#         for filename in tqdm(filenames):
#             image = imageio.imread(filename)
#             writer.append_data(image)
