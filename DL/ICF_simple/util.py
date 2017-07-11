import scipy.misc
import numpy
import inspect
import traceback

import theano
import theano.tensor as T

from theano.tensor.nnet.abstract_conv import AbstractConv2d_gradInputs
from theano.sandbox.rng_mrg import MRG_RandomStreams
from collections import OrderedDict


def nprand(shape, k):
    return numpy.float32(numpy.random.uniform(-k,k, shape))

def make_param(shape):
    if len(shape) == 1:
        return theano.shared(nprand(shape,0),'b')
    elif len(shape) == 2:
        return theano.shared(nprand(shape, numpy.sqrt(6./sum(shape))), 'W')
    elif len(shape) == 4:
        return theano.shared(nprand(shape, numpy.sqrt(6./(shape[1]+shape[0]*numpy.prod(shape[2:])))), 'W')
    raise ValueError(shape)


def _log(*args):
    if _log.on:
        print " ".join(map(str,args))
_log.on = True



class BlockType:
    def __init__(self, inputs=['input'], outputs=['output']):
        self.inputs = inputs
        self.outputs = outputs
    def __call__(self, f):
        self.f = f
        sign = inspect.getargspec(f)
        kwva = dict((a,default) for a,default in zip(sign.args[-len(sign.defaults):], sign.defaults))
        tb = ''.join(traceback.format_stack())
        
        if not 'block' in kwva:
            raise ValueError('block type',f,'does not have a block argument')
        assert kwva['block'] is None, 'block is a reserved argument and should be None'
        
        def make_block(model,name='name',**kwargs):
            block = Block(self, model, name)
            kwargs['block'] = block
            # replace named inputs with their corresponding blocks
            for inp in self.inputs:
                inp_name = kwargs[inp]
                if inp_name in model.blocks:
                    kwargs[inp] = model.blocks[inp_name]
                    block.inputs.append(inp_name)
                else:
                    raise ValueError("Model's toposort is not sorted properly, '%s' requires '%s'='%s' but '%s' cannot be found or comes later"%(name, inp, kwargs[inp], kwargs[inp]))
            # call actual block maker
            self.f(name,**kwargs)
            return block
        
        def meta_make_block(name='name',**kwargs):
            def f(model):
                try:
                    return make_block(model, name, **kwargs)
                except Exception,e:
                    print 'Block was created:'
                    print tb
                    traceback.print_exc()
                    raise e
            return f
        meta_make_block.func_doc = f.func_doc
        return meta_make_block

class Block:
    def __init__(self, blocktype, model, name):
        self.blocktype = blocktype
        self.model = model
        self.name = name
        self.paramList = []
        self.inputs = []
    def param(self, shape):
        p = make_param(shape)
        p.block = self
        self.paramList.append(p)
        self.model.registerParam(p)
        return p
    def __repr__(self):
        return '<Block %s>'%self.name

class Model:
    def __init__(self):
        self.blocks = {}
        self.params = []
    def registerParam(self, p):
        self.params.append(p)
    def build(self, description):
        # we're going to assume that `description` is already
        # correctly correctly sorted
        self.toposort = []
        for maker in description:
            block = maker(self)
            if block.name in self.blocks:
                raise ValueError("Trying to add block '%s' to Model instance but block name already exists"%block.name)
            self.blocks[block.name] = block
            self.toposort.append(block)

    def apply(self, inputs, partial=False):
        activation_cache = inputs
        for block in self.toposort:
            if block.name in inputs:
                continue

            # check if all inputs are available
            skip = False
            for i in block.inputs:
                if i not in activation_cache:
                    if partial:
                        _log('skipping',block.name,'due to partial evaluation')
                        skip = True
                    else: raise ValueError("I tried using block '%s', needed by '%s', but it is missing (maybe you want partial=True?)"%(i,block.name))
            if skip: continue

            # construct arg list and retrive the block's output
            block_inputs = [activation_cache[i] for i in block.inputs]
            outputs = block.output(*block_inputs)

            if outputs is None: continue # the block does not want to be registered
            
            if not isinstance(outputs, list) and not isinstance(outputs, tuple): outputs = [outputs]
            # fill the activation cache with the block's outputs
            activation_cache[block.name] = outputs[0]
            for output, output_name in zip(outputs, block.blocktype.outputs):
                activation_cache[block.name+'.'+output_name] = output
        return activation_cache




            
@BlockType(inputs=[])
def placeholder(name='placeholder', shape=(None, 32, 32, 3), block=None):
    _log('placeholder', name, shape)
    block.output = lambda: None
    block.output_shape = shape
    
@BlockType(outputs=['output','preact'])
def fc(name='fclayer', input='input', nout=128, act=T.tanh, block=None):
    """Build a fully connected layer
    name -- block name
    input -- input block's name
    nout -- number of outgoing units
    act -- the activation function
    """
    nin = input.output_shape[1]
    W = block.param((nin, nout))
    b = block.param((nout,))
    W.name += name; b.name += name;
    prop = lambda x: (act(T.dot(x,W)+b), T.dot(x,W)+b)
    block.output = prop
    block.output_shape = input.output_shape[0], nout
    _log('fc',name,nin, nout, act, block.output_shape)


@BlockType()
def conv(name='conv',input='input', nout=32, fs=5, act=T.nnet.relu, stride=(1,1),block=None):
    nin = input.output_shape[1]
    W = block.param((nout, nin, fs, fs))
    b = block.param((nout,))
    W.name += name; b.name += name;
    prop = lambda x: act(T.nnet.conv2d(x, W,
                                       filter_shape=W.get_value().shape,
                                       border_mode='half',
                                       subsample=stride)
                         + b.dimshuffle('x',0,'x','x'))
    block.output = prop
    block.output_shape = (input.output_shape[0], nout,
                          input.output_shape[2] / stride[0],
                          input.output_shape[3] / stride[1])
    _log('conv',name,nin,nout,fs,act,stride, block.output_shape)


@BlockType()
def conv_transpose(name='convT',input='input', nout=32, fs=5, act=T.tanh, stride=(1,1),block=None):
    nin = input.output_shape[1]
    W = block.param((nin, nout, fs, fs))
    b = block.param((nout,))
    W.name += name; b.name += name;
    convT = AbstractConv2d_gradInputs(border_mode='half',subsample=stride)
    # not sure what those two last dimensions are supposed to be :(
    prop = lambda x: act(convT(W, x, [x.shape[2]*stride[0], x.shape[3]*stride[1], 1, 1])
                         + b.dimshuffle('x',0,'x','x'))
    block.output = prop
    block.output_shape = (input.output_shape[0], nout,
                          input.output_shape[2] * stride[0],
                          input.output_shape[3] * stride[1])
    _log('convT',name,nin,nout,fs,act,stride, block.output_shape)
                         
@BlockType()
def Lambda(name='lambda', input='input', func=lambda x:x, func_shape=lambda xshape:xshape, block=None):
    block.output = func
    block.output_shape = func_shape(input.output_shape)
    _log('lambda', name, block.output_shape)

def LambdaN(name='lambda*', inputs=['A','B'], func=lambda *x:x, func_shape=lambda *x:x):
    fakekw = dict(('input%d'%i,j) for i,j in enumerate(inputs))
    @BlockType(inputs=fakekw.keys())
    def f(name='name',block=None, **kwargs):
        block.output = func
        block.output_shape = func_shape(*[kwargs[i].output_shape for i in fakekw.keys()])
        _log('lambda*',name,block.output_shape)
        
    return f(name=name, func=func,func_shape=func_shape, **fakekw)
    
@BlockType(inputs=['A','B'])
def concatenate(name='concat', A='A', B='B', axis=1, block=None):
    block.output = lambda a,b:T.concatenate([a,b], axis=axis)
    As = A.output_shape
    Bs = B.output_shape
    new_shape = [0] * len(As)
    assert len(As) == len(Bs), 'A and B must have the same number of dimensions'
    for i,(a,b) in enumerate(zip(As,Bs)):
        if i != axis:
            assert a==b, 'A and B must have the same shape along axis %d, A: %s, B: %s'%(i,As,Bs)
            new_shape[i] = a
        else:
            new_shape[i] = a+b
    block.output_shape = new_shape
    _log('concat', name, axis, As, Bs, block.output_shape)
    
@BlockType()
def image2flat(name='img2flat', input='input', block=None):
    block.output = lambda x:x.reshape([x.shape[0], -1])
    block.output_shape = input.output_shape[0], numpy.prod(input.output_shape[1:])
    _log('image2flat', name, block.output_shape)

@BlockType()
def flat2image(name='flat2img', input='input', shape=(3,32,32), block=None):
    block.output = lambda x:x.reshape([x.shape[0]]+list(shape))
    block.output_shape = [input.output_shape[0]] + list(shape)
    assert numpy.prod(shape) == input.output_shape[1],"image shape does not match this block's input shape"
    _log('flat2image', name, block.output_shape)


@BlockType(inputs=['input','htm1'])
def rnn_core(name='rnn_layer', input='input', htm1='htm1', nhid=32, block=None):
    nin = input.output_shape[-1]
    Wx = block.param((nin, nhid))
    Wh = block.param((nhid, nhid))
    b = block.param((nhid,))
    Wx.name += name; Wh.name += name; b.name += name;
    prop = lambda x,htm1: T.tanh(T.dot(x,Wx)+T.dot(htm1,Wh) + b)
    block.output = prop
    block.output_shape = input.output_shape[0], nhid
    _log('rnn core',name,nin, nhid, block.output_shape)


    
class adam:
    def __init__(self,
                 beta1 = 0.9, beta2 = 0.999, epsilon = 1e-4):
        self.b1 = numpy.float32(beta1)
        self.b2 = numpy.float32(beta2)
        self.eps = numpy.float32(epsilon)

    def __call__(self, params, grads, lr):
        t = theano.shared(numpy.array(2., dtype = 'float32'))
        updates = OrderedDict()
        updates[t] = t + 1

        for param, grad in zip(params, grads):
            last_1_moment = theano.shared(numpy.float32(param.get_value() * 0))
            last_2_moment = theano.shared(numpy.float32(param.get_value() * 0))

            new_last_1_moment = T.cast((numpy.float32(1.) - self.b1) * grad + self.b1 * last_1_moment, 'float32')
            new_last_2_moment = T.cast((numpy.float32(1.) - self.b2) * grad**2 + self.b2 * last_2_moment, 'float32')

            updates[last_1_moment] = new_last_1_moment
            updates[last_2_moment] = new_last_2_moment
            updates[param] = (param - (lr * (new_last_1_moment / (numpy.float32(1.) - self.b1**t)) /
                                      (T.sqrt(new_last_2_moment / (numpy.float32(1.) - self.b2**t)) + self.eps)))

        return list(updates.items())

    
def example():

    # First start with a Model object
    model = Model()
    # then build the blocks in a model
    model.build([
        # think of a placeholder as a T.tensor
        placeholder('input', shape=(None, 28*28)),
        # here we create two fully connected layers
        # the first argument is always the name of the block, it must be unique
        fc('fc1', input='input', nout=128, act=T.nnet.sigmoid),
        # the other arguments depend on the block type
        # generally, `input` is the name of this block's input block
        fc('fc2', input='fc1', nout=10, act=T.nnet.softmax),

        # it's also possible to define lambdas, the downside being you need to specify the shape
        LambdaN('test', inputs=['fc1','fc2'],
                func=lambda x,y:T.concatenate([x,y],axis=0),
                func_shape=lambda x,y:[x[0],y[1]+x[1]])
        # making custom blocks might be a cleaner alternative to lambdas
    ])

    x = T.matrix('x')
    y = T.vector('y')

    # apply the model to x, here {'input':x} means we're attributing x to be the output
    # of the block 'input', but it could be any block, think of it as theano's givens
    forward_pass = model.apply({'input':x})
    # retreive the output of fc2 during the (symbolic) forward pass
    pred = forward_pass['fc2']
    # the previous line retrieves the default output, but a block can have more than one
    # for example the `fc` block has two outputs, 'output' and 'preact' (before the activation)
    preact = forward_pass['fc2.preact']
    # pred is a theano Softmax.0, while preact is `T.dot(x,W) + b`, which is a theano Elemwise{add}.0
    print pred, preact 

    print forward_pass['test']
    
if __name__ == '__main__':
    example()
