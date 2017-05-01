
import theano
import theano.tensor as T
import theano.gradient

from util import StackModel, ConvLayer, HiddenLayer, rmsprop, shared

import scipy.misc

import sys
sys.path.append('./Arcade-Learning-Environment')
import ale_python_interface

import matplotlib.pyplot as pp

import multiprocessing as mp
import time
import numpy
import ctypes

def make_model(nactions):

    model = StackModel([
        ConvLayer([16,4,8,8],T.nnet.relu,init='glorot',stride=(4,4)),
        ConvLayer([32,16,4,4],T.nnet.relu,init='glorot',stride=(2,2)),
        lambda x: x.reshape((x.shape[0],-1)),
        HiddenLayer(9*9*32, 256, T.nnet.relu,'glorot'),
        HiddenLayer(256, nactions+1, lambda x:x,'glorot')
    ])

    return model

def make_funcs(model, params):


    x = T.tensor3()
    r = T.scalar()
    a = T.iscalar()
    lr = theano.shared(numpy.float32(0.001))
    beta = theano.shared(numpy.float32(0.01))
    
    out = model(x.dimshuffle('x',0,1,2))
    v = out[0,-1]
    pol = T.nnet.softmax(out[:,:-1])[0]

    A = r-v
    
    logpi = T.log(pol[a])
    actor_loss = -logpi * theano.gradient.disconnected_grad(A)
    critic_loss = A**2
    entropy = beta * T.sum(pol * T.log(pol))
    loss = T.sum(actor_loss + 0.5*critic_loss) + entropy

    grads = T.grad(loss, params)
    updates = rmsprop(0.99,0.1)(params, grads, lr)

    train = theano.function([x,a,r],[critic_loss,actor_loss],updates=updates)
    get_pol = theano.function([x],pol)
    get_v = theano.function([x], v)
    get_grads = theano.function([x,a,r], grads)
    update_from_grads = theano.function(grads, updates=updates)

    return train, get_pol, get_v, get_grads, update_from_grads, lr, beta

def plotmeans(x,**kw):
    y = numpy.arange(x.shape[0])
    N = 200
    n = x.shape[0] / N
    N = x.shape[0] / n
    xp = x[:n*N].reshape((N,n)).mean(axis=1)
    if n*N < x.shape[0]: xp = numpy.concatenate((xp, [x[n*N:].mean()]))
    yp = y[:n*N].reshape((N,n)).mean(axis=1)
    if n*N < x.shape[0]: yp = numpy.concatenate((yp, [y[n*N:].mean()]))
    pp.plot(yp,xp,**kw)


def mparray_zero(shape):
    A = mp.Array(ctypes.c_float, numpy.prod(shape))
    A = (A, shape, 'float32')
    return A + (mpasnp(A),)
def mparray(x):
    A = mparray_zero(x.shape)
    a = mpasnp(A)
    a[:] = x
    return A

def mpasnp(x):
    return numpy.frombuffer(x[0].get_obj(),x[2]).reshape(x[1])


def plot_stuff(rewards, running):
    running = mpasnp(running)
    pp.ion()
    pp.show()
    while running>0:
        print 'plot', running
        time.sleep(5)
        pp.clf()
        if len(rewards) < 200:
            pp.plot(rewards)
        else:
            try:
                plotmeans(numpy.float32(rewards))
            except Exception,e:
                print e
        #pp.show(block=False)
        pp.draw()
        pp.pause(0.001)
        pp.savefig('rewards.png')

def main():
    ale = ale_python_interface.ALEInterface()
    ale.loadROM('aleroms/breakout.bin')
    actions = ale.getMinimalActionSet()

    params = shared.bindNew()
    print 'building main model...'
    model = make_model(len(actions))
    train, get_pol, get_v, get_grads, update_from_grads, lr, beta = make_funcs(model, params)

    global_params = [mparray(p.get_value()) for p in params]
    for gp,p in zip(global_params,params):
        p.set_value(gp[3], borrow=True)
    
    nthreads = 4

    thread_grads = [
        [mparray_zero(p.get_value().shape) for p in params]
        for i in range(nthreads)
    ]
    thread_counters = mparray_zero((nthreads,))
    counters = mpasnp(thread_counters)
    running = mparray_zero((1,))
    running[3][0] += nthreads
    print 'starting threads'
    manager = mp.Manager()
    rewards = manager.list()
    for i in range(nthreads):
        p = mp.Process(target=thread_learner,args=(i, global_params, thread_grads[i],
                                                   thread_counters, running, rewards))
        p.start()

    mp.Process(target=plot_stuff, args=(rewards,running)).start()
        

    try:
        last_draw = time.time()
        upds = numpy.zeros(nthreads)
        cnts = numpy.zeros(nthreads)
        while running[3][0]>0:
            for i in range(nthreads):
                if counters[i] > 0:
                    with thread_grads[i][0][0].get_lock():
                        grads = [g[3] / counters[i] for g in thread_grads[i]]
                        update_from_grads(*grads)
                        for g in thread_grads[i]:
                            g[3][:] *= 0
                        upds[i] += counters[i]
                        cnts[i] += 1
                        counters[i] = 0
            if time.time()-last_draw > 2:
                print 'updates:', numpy.mean(upds/cnts), numpy.mean(upds), running[3]
                upds = numpy.zeros(nthreads)
                cnts = numpy.zeros(nthreads)
                last_draw = time.time()

    finally:
        running[3][0] = 0


def thread_learner(thread_idx, global_params, t_grads, t_counters, running, rs):
    try:
        thread_learner_(thread_idx, global_params, t_grads, t_counters, running, rs)
    except Exception,e:
        mpasnp(running)[0] = 0
        raise e
    finally:
        mpasnp(running)[0] = 1
    print thread_idx,'done',mpasnp(running)
    
def thread_learner_(thread_idx, global_params, t_grads, t_counters, running, rs):
    numpy.random.seed(thread_idx)
    thread_counters = mpasnp(t_counters)
    thread_grads = map(mpasnp, t_grads)
    
    
    ale = ale_python_interface.ALEInterface()
    ale.loadROM('aleroms/breakout.bin')
    #ale.setInt('frame_skip', 4)
    ale.setFloat('repeat_action_probability',0)
    ale.setBool('color_averaging', False)
    actions = ale.getMinimalActionSet()

    params = shared.bindNew()
    
    model = make_model(len(actions))
    train, get_pol, get_v, get_grads, update_from_grads, lr, beta = make_funcs(model, params)

    for gp, p in zip(global_params, params):
        p.set_value(mpasnp(gp), borrow=True)
    
    print 'compiled functions'

    def getImg():
        x = ale.getScreenGrayscale()
        return numpy.float32(scipy.misc.imresize(x[:,:,0], (84,84)) / 255.)

    ale.reset_game()
    x = [getImg()] * 4
    t_max = 5
    gamma = 0.99
    loss = 0
    t = 0
    tot_r = 0
    #rs = []
    #pp.ion()
    #pp.show()
    t1000 = time.time()
    frame_0 = ale.getEpisodeFrameNumber()
    for i in range(1000000):
        if running[3][0] < 1:
            break
        traj = []
        for j in range(t_max):
            t+=1
            x = x[1:]+[getImg()]
            pol = get_pol(x)
            a = numpy.int32(numpy.argmax(numpy.random.multinomial(1,pol)))
            r = 0
            for _ in range(4):
                r += numpy.float32(ale.act(actions[a]))
            tot_r += r
            traj.append([x,a,r])
            if ale.game_over(): break
            
        R = 0 if ale.game_over() else get_v(traj[-1][0])
        for x,a,r in traj[:-1][::-1]:
            R = r + gamma*R
            # loss += train(x,a,numpy.float32(R))[0]
            gs = get_grads(x,a,numpy.float32(R))
            #print 'pushing grads', thread_idx, thread_counters
            t0 = time.time()
            with t_grads[0][0].get_lock():
                #print '   ',time.time()-t0,thread_idx
                for g,tg in zip(gs,thread_grads):
                    tg += g
                thread_counters[thread_idx] += 1
            
        if ale.game_over():
            t1001 = time.time()
            
            print 'FPS:', ale.getEpisodeFrameNumber() / (t1001-t1000)
            print '    ', ale.getEpisodeFrameNumber()
            print '    ', thread_idx
            t1000 = time.time()
            beta.set_value(numpy.float32(beta.get_value()*0.99))
            rs.append(tot_r)
            print i, t, loss, tot_r
            print pol, beta.get_value()
            ale.reset_game()
            x = [getImg()] * 4
            loss = 0
            t = 0
            tot_r = 0
            if 0:
                pp.clf()
                if len(rs) < 200:
                    pp.plot(rs)
                else:
                    plotmeans(numpy.float32(rs))
                #pp.show(block=False)
                pp.draw()
                pp.pause(0.001)
        #pp.savefig('rewards.png')
            
        #import pdb; pdb.set_trace()
            
            
        
main()
