import scipy.misc
import numpy

import theano
import theano.tensor as T

from theano.tensor.nnet.abstract_conv import AbstractConv2d_gradInputs
from theano.sandbox.rng_mrg import MRG_RandomStreams


def nprand(shape, k):
    return numpy.float32(numpy.random.uniform(-k,k, shape))

def make_param(shape):
    if len(shape) == 1:
        return theano.shared(nprand(shape,0),'b')
    elif len(shape) == 2:
        return theano.shared(nprand(shape, numpy.sqrt(6./sum(shape))), 'W')
    elif len(shape) == 4:
        return theano.shared(nprand(shape, numpy.sqrt(6./(shape[0]*numpy.prod(shape[2:])))), 'W')
    raise ValueError(shape)

def make_fc(nin, nout, act):
    print 'fc',nin,nout, act
    W = make_param((nin, nout))
    b = make_param((nout,))
    prop = lambda x: act(T.dot(x,W)+b)
    return [W,b], prop

def make_conv(nin, nout, fs, act, stride=(1,1)):
    print 'conv',nin,nout,fs,act,stride
    W = make_param((nout, nin, fs, fs))
    b = make_param((nout,))
    prop = lambda x: act(T.nnet.conv2d(x, W,
                                       filter_shape=W.get_value().shape,
                                       border_mode='half',
                                       subsample=stride)
                         + b.dimshuffle('x',0,'x','x'))
    return [W,b], prop
    
def make_conv_transpose(nin, nout, fs, act, stride=(1,1)):
    print 'convT',nin,nout,fs,act,stride
    W = make_param((nin, nout, fs, fs))
    b = make_param((nout,))
    convT = AbstractConv2d_gradInputs(border_mode='half',subsample=stride)
    # not sure what those two last dimensions are supposed to be :(
    prop = lambda x: act(convT(W, x, [x.shape[2]*stride[0], x.shape[3]*stride[1], 1, 1])
                         + b.dimshuffle('x',0,'x','x'))
    return [W,b], prop
                         
def make_stack_of_layers(layers):
    params = [p for i in layers for p in i[0]]
    def prop(x):
        for l in layers:
            x = l[1](x)
        return x
    return params, prop

def Lmb(l):
    return [], l

def make_dcgan(nz, generator, discriminator):
    acts = [T.nnet.relu] * (len(generator)-2) + [T.nnet.sigmoid]
    g_params, gen = make_stack_of_layers(
        [make_fc(nz, generator[0] * 4 * 4, T.nnet.relu),
         Lmb(lambda x: x.reshape((x.shape[0], generator[0], 4, 4)))] +
        [make_conv_transpose(gen_in, gen_out, 3, act, (2,2))
         for gen_in, gen_out,act in zip(generator[:-1],generator[1:],acts)
        ])

    d_params, disc = make_stack_of_layers(
        [make_conv(dis_in, dis_out, 3, T.nnet.relu, (2,2))
         for dis_in, dis_out in zip(discriminator[:-2], discriminator[1:])] +
        [Lmb(lambda x: x.reshape((x.shape[0], -1))),
         make_fc(discriminator[-2] * 4 * 4, discriminator[-1], T.nnet.relu),
         make_fc(discriminator[-1], 1, lambda x:x)])

    return g_params, d_params, gen, disc


def rmsprop(params, grads, lr, epsilon=0.01, decay=0.999, clip=None):
    lr = theano.shared(numpy.float32(lr))
    updates = []
    # shared variables
    mean_square_grads = [theano.shared(i.get_value()*0.+1) for i in params]
    # msg updates:
    new_mean_square_grads = [decay * i + (1 - decay) * T.sqr(gi)
                             for i,gi in zip(mean_square_grads, grads)]
    updates += [(i,ni) for i,ni in zip(mean_square_grads,new_mean_square_grads)]
    # 
    rms_grad_t = [T.sqrt(i+epsilon) for i in new_mean_square_grads]
    # actual updates
    delta_x_t = [lr * gi / rmsi for gi,rmsi in zip(grads, rms_grad_t)]
    if clip is None:
        updates += [(i, i-delta_i)
                    for i,delta_i in zip(params,delta_x_t)]
    else:
        updates += [(i, T.clip(i-delta_i,-clip,clip))
                    for i,delta_i in zip(params,delta_x_t)]
    return updates

def build_model(gen_lr, dis_lr, score_type):

    assert score_type in ['JSD', 'W1'] # jensen shannon divergence or Earth-Mover/Wasserstein-1 distance
    # https://arxiv.org/abs/1406.2661
    # https://arxiv.org/abs/1701.07875
    
    srng = MRG_RandomStreams(seed=1)
    
    nz = 32
    # [4,8,16,32]
    ngen = [64,32,16,1]
    # [32,16,8,4]
    ndisc = [1,16,32,64, 64*16]

    clip = None

    gp, dp, gen, disc = make_dcgan(nz, ngen, ndisc)

    if score_type == 'JSD':
        # by default disc(x) is linear, let's make it a sigmoid
        disc_ = disc
        disc = lambda x: T.nnet.sigmoid(disc_(x))

    x = T.tensor4()
    mbsize = x.shape[0]
    z = srng.normal((mbsize, nz), 0, 1)
    
    xhat = gen(z)
    
    x_score = disc(x)
    xhat_score = disc(xhat)
    
    if score_type == 'JSD':
        g_score = T.sum(T.log(xhat_score))
        d_score = T.sum(T.log(x_score) + T.log(1-xhat_score))
    if score_type == 'W1':
        g_score = T.mean(xhat_score)
        d_score = T.mean(x_score - xhat_score)
        clip = 0.5 # the paper clips to 0.01 but it's not for MNSIT

    g_grads = T.grad(-g_score, gp)
    d_grads = T.grad(-d_score, dp)

    g_updates = rmsprop(gp, g_grads, gen_lr)
    d_updates = rmsprop(dp, d_grads, dis_lr, clip=clip)

    learn_actor = theano.function([x], g_score, updates=g_updates)
    learn_critic = theano.function([x], d_score, updates=d_updates)
    generate = theano.function([mbsize], xhat)
    adverse = theano.function([x],[x_score, T.grad(T.mean(x_score), x)])

    return learn_actor, learn_critic, generate, adverse


def main(gen_lr, dis_lr, score_type, dataEpoch, ncritic):
    print 'building model'
    learn_actor, learn_critic, generate, adverse = build_model(gen_lr, dis_lr, score_type)
    print 'training'
    for epoch in range(50):
        score = numpy.float32([0,0])
        for x in dataEpoch(mbsize=64):
            for i in range(ncritic):
                score[1] += learn_critic(x)
            score[0] += learn_actor(x)
        print epoch, score
        xhat = generate(16).reshape((-1,32,32))
        print xhat.shape,numpy.hstack(xhat).shape
        scipy.misc.imsave('mnist_%s_%02d.png'%(score_type,epoch),numpy.hstack(xhat))
        x0 = numpy.float32(numpy.random.uniform(0,1,(1,1,28,28)))
        xs = []
        for i in range(25*25-1):
            score, grad = adverse(x0)
            xs.append(x0[0].reshape((28,28)))
            x0 = x0 + 0.05 * grad / abs(grad).max()
            if not i % 10: print i, score
        xs.append(x0[0].reshape((28,28)))
        print 25*25, len(xs)
        xs = numpy.hstack(numpy.hstack(numpy.array(xs).reshape((25,25,28,28))))
                          
        scipy.misc.imsave('mnist_adv_%s_%02d.png'%(score_type,epoch), xs)
    print 'done'
    
def mnist32x32():
    import cPickle, gzip
    try:
        train, valid, test = cPickle.load(gzip.open('mnist.pkl.gz','r'))
    except IOError,e:
        print e
        print 'download MNIST from http://deeplearning.net/data/mnist/'
    X = numpy.zeros((train[0].shape[0], 1, 32, 32),'float32')
    X[:, 0, :28, :28] = train[0].reshape((-1,28,28))
    scipy.misc.imsave('mnist.png',numpy.hstack(X[:16,0]))
    
    def epoch(mbsize=32):
        N = X.shape[0]
        idxes = numpy.arange(0,N/mbsize+(1 if N%mbsize else 0))
        numpy.random.shuffle(idxes)
        for i in idxes:
            yield X[mbsize*i:mbsize*(i+1)]
    return epoch

if __name__ == '__main__':
    print 'loading data'
    dataEpoch = mnist32x32()
    #main(0.0001, 0.0001, 'JSD', dataEpoch, ncritic=1)
    main(0.0001, 0.00005, 'W1', dataEpoch, ncritic=5)
