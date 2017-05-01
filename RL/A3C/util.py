
import theano
import theano.tensor as T
import numpy
import gzip
import pickle# as pickle
from collections import OrderedDict

def rmsprop(decay, epsilon=1e-3, clip=None):
    def sgd(params, grads, lr):
        if clip is not None:
            grads = [T.clip(i, -clip, clip) for i in grads]
        updates = []
        # shared variables
        mean_square_grads = [theano.shared(i.get_value()*0.+1) for i in params]
        # msg updates:
        new_mean_square_grads = [decay * i + (1 - decay) * T.sqr(gi)
                                 for i,gi in zip(mean_square_grads, grads)]
        updates += [(i,ni) for i,ni in zip(mean_square_grads,new_mean_square_grads)]
        # cap sqrt(i) at epsilon
        rms_grad_t = [T.sqrt(i+epsilon) for i in new_mean_square_grads]
        # actual updates
        delta_x_t = [lr * gi / rmsi for gi,rmsi in zip(grads, rms_grad_t)]
        updates += [(i, i-delta_i)
                    for i,delta_i in zip(params,delta_x_t)]
        return updates
    return sgd


class SharedGenerator:
    def __init__(self):
        self.reset()
        self.init_tensor_x = T.scalar()
        self.init_minibatch_x = numpy.float32(0)
        self.isJustReloadingModel = False
    def reset(self):
        self.param_list = [] # currently bound list of parameters
        self.param_groups = {} # groups of parameters
        self.param_costs = {} # each group can have attached costs
    def bind(self, params, name="default"):
        if type(params)==str:
            self.param_list = self.param_groups[params]
            return
        self.param_list = params
        self.param_groups[name] = params
        if name not in self.param_costs:
            self.param_costs[name] = []
    def bindNew(self, name='default'):
        p = []
        self.bind(p, name)
        return p

    def __call__(self, name, shape, init='uniform', **kwargs):
        #print("init",name,shape,init,kwargs)
        if type(init).__module__ == numpy.__name__: # wtf numpy
            values = init
        elif init == "uniform" or init == "glorot" or init == 'tanh':
            k = numpy.sqrt(6./numpy.sum(shape)) if 'k' not in kwargs else kwargs['k']
            values = numpy.random.uniform(-k,k,shape)
        elif  init == 'relu':
            p = kwargs['inputDropout'] if 'inputDropout' in kwargs and kwargs['inputDropout'] else 1
            k = numpy.sqrt(6.*p/shape[0]) if 'k' not in kwargs else kwargs['k']
            values = numpy.random.uniform(-k,k,shape)
        elif init == "one":
            values = numpy.ones(shape)
        elif init == "zero":
            values = numpy.zeros(shape)
        elif init == 'ortho':
            def sym(w):
                import numpy.linalg
                from scipy.linalg import sqrtm, inv
                return w.dot(inv(sqrtm(w.T.dot(w))))
            values = numpy.random.normal(0,1,shape)
            values = sym(values).real

        else:
            print(type(init))
            raise ValueError(init)
        s = theano.shared(numpy.float32(values), name=name)
        self.param_list.append(s)
        return s

    def exportToFile(self, path):
        exp = {}
        for g in self.param_groups:
            exp[g] = [i.get_value() for i in self.param_groups[g]]
        pickle.dump(exp, open(path,'wb'), -1)

    def importFromFile(self, path):
        exp = pickle.load(open(path,'rb'))
        for g in exp:
            for i in range(len(exp[g])):
                print(g, exp[g][i].shape)
                self.param_groups[g][i].set_value(exp[g][i])
    def attach_cost(self, name, cost):
        self.param_costs[name].append(cost)
    def get_costs(self, name):
        return self.param_costs[name]
    def get_all_costs(self):
        return [j  for i in self.param_costs for j in self.param_costs[i]]
    def get_all_names(self):
        print([i for i in self.param_costs])
        print(self.param_costs.keys())
        return self.param_costs.keys()

    def computeUpdates(self, lr, gradient_method=gradient_descent):
        updates = []
        for i in self.param_costs:
            updates += self.computeUpdatesFor(i, lr, gradient_method)
        return updates

    def computeUpdatesFor(self, name, lr, gradient_method=gradient_descent):
        if name not in self.param_costs or \
           not len(self.param_costs[name]):
            return []
        cost = sum(self.param_costs[name])
        grads = T.grad(cost, self.param_groups[name])
        updates = gradient_method(self.param_groups[name], grads, lr)

        return updates


shared = SharedGenerator()


class HiddenLayer:
    def __init__(self, n_in, n_out, activation, init="glorot", canReconstruct=False,
                 inputDropout=None,name="",outputBatchNorm=False):
        """
        Typical a(Wx+b) hidden layer.
        
        Will init differently given inputDropout
        """
        self.W = shared("W"+name, (n_in, n_out), init, inputDropout=inputDropout)
        self.b = shared("b"+name, (n_out,), "zero")
        self.activation = activation
        self.params = [self.W, self.b]
        if canReconstruct:
            self.bprime = shared("b'"+name, (n_in,), "zero")
            self.params+= [self.bprime]
        if outputBatchNorm:
            self.gamma = shared('gamma', (n_out,), "one")
            self.beta = shared('beta', (n_out,), "zero")
            self.params += [self.gamma, self.beta]
            def bn(x):
                mu = x.mean(axis=0)
                std = T.maximum(T.std(x, axis=0),numpy.float32(1e-6))
                x = (x - mu) / std
                return activation(self.gamma * x + self.beta)
            self.activation = bn
        self.name=name
    def orthonormalize(self):
        import numpy.linalg
        from scipy.linalg import sqrtm, inv

        def sym(w):
            return w.dot(inv(sqrtm(w.T.dot(w))))
        Wval = numpy.random.normal(0,1,self.W.get_value().shape)
        Wval = sym(Wval).real
        self.W.set_value(numpy.float32(Wval))

    def apply(self, *x):
        return self(*x)
    def __call__(self, x, *args):
        return self.activation(T.dot(x,self.W) + self.b)
    def apply_partially(self, x, nin, nout):
        return self.activation(T.dot(x, self.W[:nin,:nout]) + self.b[:nout])
    def reconstruct(self, x):
        return self.activation(T.dot(x,self.W.T) + self.bprime)



class ConvLayer:
    use_cudnn = False
    def __init__(self, filter_shape, #image_shape,
                 activation = lambda x:x,
                 use_bias=True,
                 init="glorot",
                 inputDropout=None,
                 normalize=False,
                 mode="valid",
                 stride=(1,1),
                 use_cudnn=None,
                 batch_normalization=False):
        #print("in:",image_shape,"kern:",filter_shape,)
        #mw = (image_shape[2] - filter_shape[2]+1)
        #mh = (image_shape[3] - filter_shape[3]+1)
        #self.output_shape = (image_shape[0], filter_shape[0], mw, mh)
        #print("out:",self.output_shape,)
        #print("(",numpy.prod(self.output_shape[1:]),")")
        self.normalize = normalize
        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) )
        self.filter_shape = filter_shape
        #self.image_shape = image_shape
        self.activation = activation
        self.mode=mode
        if use_cudnn is not None:
            self.cudnn= use_cudnn
        else:
            self.cudnn = ConvLayer.use_cudnn
        self.stride = stride

        if init=="glorot":
            self.k = k = numpy.sqrt(6./(fan_in+fan_out))
        elif init=="relu":
            if inputDropout:
                self.k = k = numpy.sqrt(6.*inputDropout/(fan_in))
            else:
                self.k = k = numpy.sqrt(6./(fan_in))

        self.use_bias = use_bias

        W = shared('Wconv', filter_shape, 'uniform', k=k)
        self.W = W
        self.params = [W]
        if use_bias:
            b = shared('bconv',filter_shape[0], 'zero')
            self.b = b
            self.params += [b]

        if self.normalize:
            g = shared('normalization_g', (filter_shape[0],), 'one')
            self.g = g
            self.params += [g]

        self.batch_normalization = batch_normalization
        if batch_normalization:
            self.bn = ConvBatchNormalization(filter_shape[0])
    def __call__(self, x, *args):
        if self.normalize:
            W = self.g.dimshuffle(0,'x','x','x') * \
                (self.W - self.W.mean(axis=[1,2,3]).dimshuffle(0,'x','x','x')) /  \
                T.sqrt(T.sum(self.W**2, axis=[1,2,3])).dimshuffle(0,'x','x','x')
        else:
            W = self.W
        #print("conv call:",x,W,self.mode,self.stride)
        #print(x.tag.test_value.shape
        #try:
        #    print(W.tag.test_value.shape)
        #except:
        #    print(W.get_value().shape)
        #print(self.mode)
        #print(self.stride)
        if self.cudnn:
            conv_out = dnn_conv(x,W,self.mode,self.stride)
        else:
            if self.mode == 'half' and 'cpu' in theano.config.device:
                fso = self.filter_shape[2] - 1
                nps = x.shape[2]
                conv_out = T.nnet.conv2d(input=x, filters=W,
                                       filter_shape=self.filter_shape,
                                       border_mode='full',
                                       subsample=self.stride)[:,:,fso:nps+fso,fso:nps+fso]
            else:
                conv_out = T.nnet.conv2d(
                    input=x,
                    filters=W,
                    filter_shape=self.filter_shape,
                    border_mode=self.mode,
                    subsample=self.stride,
                    #image_shape=self.image_shape if image_shape is None else image_shape
                )

        if self.normalize and not shared.isJustReloadingModel:
            mu = T.mean(conv_out, axis=[0,2,3]).eval({shared.init_tensor_x: shared.init_minibatch_x})
            sigma = T.std(conv_out, axis=[0,2,3]).eval({shared.init_tensor_x: shared.init_minibatch_x})
            print("normalizing:",mu.mean(),sigma.mean())
            self.g.set_value( 1 / sigma)
            self.b.set_value(-mu/sigma)

        if hasattr(shared, 'preactivations'):
            shared.preactivations.append(conv_out)

        if 0: # mean-norm
            conv_out = conv_out - conv_out.mean(axis=[0,2,3]).dimshuffle('x',0,'x','x')

        if self.use_bias:
            out = self.activation(conv_out + self.b.dimshuffle('x',0,'x','x'))
        else:
            out = self.activation(conv_out)
        #print("out:", out.tag.test_value.shape)
        
        if self.batch_normalization:
            out = self.bn(out) 

        return out



class StackModel:
    def __init__(self, layers):
        self.layers = layers
        self.params = set()

    def apply(self, *x,**kw):
        return self.__call__(*x,**kw)

    def __call__(self, *x, **kw):
        activations = kw.get('activations', None)
        upto = kw.get('upto', len(self.layers))

        if hasattr(x[0].tag, 'test_value'):
            print("input shape:", x[0].tag.test_value.shape)
        for i,l in enumerate(self.layers[:upto]):
            x = l(*x)
            if hasattr(x,'tag') and hasattr(x.tag, 'test_value'):
                print(l,"output shape:", x.tag.test_value.shape, x.tag.test_value.dtype)
            if activations is not None: activations.append(x)
            if type(x) != list and type(x) != tuple:
                x = [x]
            if hasattr(l, "params") and not hasattr(l, "_nametagged"):
                for p in l.params:
                    p.name = p.name+"-"+str(i)
                    self.params.add(p)
                l._nametagged=True
        self.params = list(self.params)
        return x if len(x)>1 else x[0]

    def reconstruct(self, x, upto):

        h = self(x, upto=upto)
        for l in self.layers[upto-1::-1]:
            h = l.reconstruct(h)
        return h
