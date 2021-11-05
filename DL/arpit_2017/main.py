import cPickle
import time
from theano_tools.deep import*
from theano_tools import GenericClassificationDataset
import os 
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pp
import matplotlib.cm as cm
cmname = 'jet'
cmap = cm.get_cmap(cmname)


data = GenericClassificationDataset('mnist')

realX = data.train[0].reshape((-1,14,2,14,2)).mean(axis=4).mean(axis=2).reshape((-1,14*14))
randX = numpy.float32(numpy.random.uniform(0,1,(50000,14*14)))

realY = data.train[1]
randY = numpy.int32(numpy.random.randint(0,10,(50000,)))
randYI = numpy.int32(numpy.arange(50000))


def build_model(nhid, nlayers, mbsize, nclasses):
    nin = 14*14
    nout = nhid
    params = []
    for i in range(nlayers+2):
        params += [shared('W', (nin, nout)), shared('b', nout, 'zero')]
        nin = nhid
        if i == nlayers: nout = nclasses
    lr = theano.shared(numpy.float32(0.05))

    def loop(x,y,*params):
        def fprop(o):
            for i in range(nlayers+1):
                o = T.nnet.relu(T.dot(o, params[i*2]) + params[i*2+1])
            o = T.nnet.softmax(T.dot(o, params[-2]) + params[-1])
            return o
        pred = fprop(x)
        yh = T.extra_ops.to_one_hot(y, nclasses)
        loss = T.mean(-T.log(1e-1+pred)*yh,axis=1)
        grads = T.grad(T.mean(loss), params)
        new_params = [(i - lr * gi) for i,gi in zip(params, grads)]
        err = T.sum(T.neq(T.argmax(pred, axis=1), y))

        disc_pred = fprop(theano.gradient.disconnected_grad(x))
        disc_loss = T.mean(-T.log(1e-1+disc_pred)*yh)
        return new_params + [disc_loss, loss, err]

    
    X = T.matrix('X')
    Y = T.ivector('Y')

    n_steps = T.iscalar()
    idxes = T.lvector()
    idxes_ = srng.random_integers([n_steps * mbsize,], low=0,high=X.shape[0]-1)
    xs = X[idxes].reshape((n_steps, mbsize, -1)) #X[idxes]
    ys = Y[idxes].reshape((n_steps, mbsize))#Y[idxes]
    outputs, _ = theano.scan(loop,
                             sequences=[xs,ys],
                             outputs_info=params + [None]*3,
                             allow_gc=False)
    new_params = outputs[:-3]
    disc_losses = outputs[-3]
    losses = outputs[-2]
    errs = outputs[-1]

    grads = T.mean(abs(T.jacobian(disc_losses[T.arange(n_steps-500,n_steps,2)], X)),axis=2)
    end_g = T.mean(abs(T.jacobian(losses[-1], X)),axis=2)
    get_grads = theano.function([X,Y,n_steps],[grads,errs,losses,end_g,ys],givens={idxes:idxes_})
    just_loss = theano.function([X,Y,n_steps],[errs,losses],givens={idxes:idxes_})

    return get_grads, just_loss

def get_data(Nex, realx_prop, realy_prop):
    if realx_prop == '100ex':
        idxes = numpy.random.randint(0,100,Nex)
        X = realX[idxes]
        X += numpy.random.normal(0,0.05,X.shape)
        if realy_prop == 'I':
            Y = randYI[:100]
        else:
            Y = realY[idxes]
        print '100ex',X.shape, Y.shape
    else:
        nreal = int(realx_prop*Nex)
        X = numpy.concatenate([realX[:nreal], randX[nreal:Nex]])
        if realy_prop == 'I':
            Y = randYI[:Nex]
        else:
            nreal = int(realy_prop*Nex)
            Y = numpy.concatenate([realY[:nreal], randY[nreal:Nex]])
    return X,Y

def main(nhid, nlayers, niter, mbsize, Nex, realx_prop, realy_prop):
    print nhid, nlayers, niter, mbsize, Nex, realx_prop, realy_prop
    nclasses = 10
    if realy_prop == 'I':
        nclasses = Nex
    grds, train = build_model(nhid, nlayers, mbsize, nclasses)

    X,Y = get_data(Nex, realx_prop, realy_prop)
    
    g,e,l,eg,ey = grds(X,Y, niter)
    # version 0
    cPickle.dump([g,e,l,eg,ey, [1,nhid,nlayers,niter,mbsize, Nex,realx_prop, realy_prop]], file('results/%s.pkl'%str(uuid.uuid4())[:8],'w'), -1)
    print 'done'




configs = []
### done
# normal situation
"""configs += [[32, 1, n, 32, 1000, 1, 1]
            for n in [500,800,1200,2000,5000,10000,30000,100000]]"""
# very noisy data, mix randX with realX
"""configs += [[32, 1, n, 32, 1000, 0.5, 1]
            for n in [500,800,1200,2000,5000,10000,30000,100000]]"""
# saturate the network with random examples
"""configs += [[16, 2, 30000, 32, 50000, 0, 1]]"""

### not done yet
# 100 ex
#configs += [[16, 2, 30000, 32, 2000, '100ex', 1]]
#configs += [[16, 2, 10000, 32, 2000, '100ex', 1]]
#configs += [[16, 2, 5000, 32, 2000, '100ex', 1]]
# all noise data
#configs += [[32, 1, n, 32, 1000, 0, 1]
#            for n in [500,800,1200,2000,5000,10000,30000,100000]]
# 2 layers
#configs += [[16, 2, n, 32, 1000, 1, 1]
#            for n in [500,800,1200,2000,5000,10000,30000,100000]]
# 2 layers random
#configs += [[16, 2, n, 32, 1000, 0, 1]
#            for n in [500,800,1200,2000,5000,10000,30000,100000]]

# 1 class per example
# realX
#configs += [[32, 1, n, 32, 1000, 1, 'I']
#            for n in [500,800,1200,2000,5000,10000,30000]]#,100000]]
# noiseX
#configs += [[32, 1, n, 32, 1000, 0, 'I']
#            for n in [500,800,1200,2000,5000,10000,30000]]#,100000]]


# 10 classes, varied level of label corruption

configs += [[32, 1, n, 32, 1000, 1, yprop]
            for n in [500,1200,5000,10000,30000]
            for yprop in [0,0.2,0.4,0.6,0.8,1]]#,100000]]


if __name__ == '__main__':
    import multiprocessing as mp
    import uuid
    pool = mp.Pool(8)

    res = []
    
    def ls(d):
        return [os.path.join(d,i) for i in os.listdir(d)]

    results = [cPickle.load(file(p,'r'))
               for p in ls('results')]
    rcfgs = [i[-1][1:] for i in results]
    for i,c in enumerate(configs):
        if c in rcfgs:
            print c, 'already done'
            continue
        
        res.append(pool.apply_async(main, c))# (i+maxn, 32, 1, 100000, 32, 1000, 1, 0)))
    [i.get() for i in res]
