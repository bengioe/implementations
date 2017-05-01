
import scipy.misc

import sys
sys.path.append('./Arcade-Learning-Environment')
import ale_python_interface

import matplotlib.pyplot as pp

import theano
import theano.tensor as T
import theano.gradient

from util import StackModel, ConvLayer, HiddenLayer, rmsprop, shared


def make_model(nactions):

    model = StackModel([
        ConvLayer([16,4,8,8],T.nnet.relu,init='glorot',stride=(4,4)),
        ConvLayer([32,16,4,4],T.nnet.relu,init='glorot',stride=(2,2)),
        lambda x: x.reshape((x.shape[0],-1)),
        HiddenLayer(9*9*32, 256, T.nnet.relu,'glorot'),
        HiddenLayer(256, nactions+1, lambda x:x,'glorot')
    ])

    return model

def plotmeans(x,**kw):
    y = numpy.arange(x.shape[0])
    N = 200
    n = x.shape[0] / N
    xp = numpy.concatenate(x[:n*N].reshape((N,n)).mean(axis=1), x[n*N:].mean())
    yp = numpy.concatenate(y[:n*N].reshape((N,n)).mean(axis=1), y[n*N:].mean())
    pp.plot(yp,xp,**kw)

def main():
    ale = ale_python_interface.ALEInterface()
    ale.loadROM('aleroms/pong.bin')
    actions = ale.getMinimalActionSet()

    params = shared.bindNew()
    
    model = make_model(len(actions))

    x = T.tensor3()
    r = T.scalar()
    a = T.iscalar()
    lr = theano.shared(numpy.float32(0.00025))
    
    out = model(x.dimshuffle('x',0,1,2))
    v = out[0,-1]
    pol = T.nnet.softmax(out[:,:-1])[0]

    A = r-v
    
    logpi = T.log(pol[a])
    actor_loss = logpi * theano.gradient.disconnected_grad(A)
    critic_loss = A**2
    loss = T.mean(actor_loss + critic_loss)

    updates = rmsprop(0.999)(params, T.grad(loss, params), lr)

    train = theano.function([x,a,r],[critic_loss,actor_loss],updates=updates)
    get_pol = theano.function([x],pol)
    get_v = theano.function([x], v)

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
    rs = []
    pp.ion()
    pp.show()
    for i in range(100000):
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
            loss += train(x,a,numpy.float32(R))[0]
        if ale.game_over():
            rs.append(tot_r)
            print i, t, loss, tot_r
            print pol
            ale.reset_game()
            x = [getImg()] * 4
            loss = 0
            t = 0
            tot_r = 0
            pp.clf()
            if len(rs) < 200:
                pp.plot(rs)
            else:
                try:
                    plotmeans(numpy.float32(rs))
                except Exception,e:
                    print e
            #pp.show(block=False)
            pp.draw()
            pp.pause(0.001)
    pp.savefig('rewards.png')
            
        #import pdb; pdb.set_trace()
            
            
        
main()
