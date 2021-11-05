import cPickle
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.cm as cm
jet_cm = cm.get_cmap('jet')
import matplotlib.pyplot as pp
import os
import os.path
import numpy as np

pp.rcParams['text.latex.preamble']=["\\usepackage{lmodern}\n\\usepackage{xfrac}"]
params = {'text.usetex' : True,
          'font.size' : 14,
          'font.family' : 'lmodern',
          'text.latex.unicode': True,
          }
pp.rcParams.update(params)

ext = '.png'

if 0:
    from main_struct import get_data

def ls(d):
    return [os.path.join(d,i) for i in os.listdir(d)]

results = [(p,cPickle.load(file(p,'r')))
           for p in ls('results')]

for r in results:
    print r[0],r[1][-1] # config
results = [r[1] for r in results]
    
def plot_class_sensitivity(rand):
    print 'class sensitivity'
    sens = numpy.zeros((10,10))
    counts = numpy.zeros((10,))
    for r in results:
        if r[-1][0] == 0: continue
        g,e,l,end_grads,ys,cfg = r
        if cfg[0] == 1:
            v,nhid,nlayers,niter,mbsize, Nex,realx_prop, realy_prop = cfg
        if realx_prop != rand:
            continue
        print cfg
        X,Y = get_data(Nex, realx_prop, realy_prop)
        for i in range(mbsize):
            yi = ys[-1][i]
            gi = end_grads[i]
            si = [numpy.mean(gi[Y==cls]) for cls in range(10)]
            sens[yi] = si
            counts[yi] += 1
    print sens
    print counts
    counts[counts==0] += 1
    sens /= counts
#    for i in range(10):
#        for j in range(10):
#            sens[i,j] = sens[j,i]
    pp.matshow(numpy.log(sens))
    pp.savefig('plots/class_sensitivity%s%s'%('_rand' if rand==0 else '', ext))


def plot_class_sensitivity_both():
    print 'class sensitivity'
    f,ax = pp.subplots(1,2,figsize=(16,7))
    for rand in [1,0]:
        sens = numpy.zeros((10,10))
        counts = numpy.zeros((10,))
        for r in results:
            if r[-1][0] == 0: continue
            g,e,l,end_grads,ys,cfg = r
            if cfg[0] == 1:
                v,nhid,nlayers,niter,mbsize, Nex,realx_prop, realy_prop = cfg
            if realx_prop != rand:
                continue
            if realy_prop != 1: continue
            #print cfg
            X,Y = get_data(Nex, realx_prop, realy_prop)
            for i in range(mbsize):
                yi = ys[-1][i]
                gi = end_grads[i]
                si = [numpy.mean(gi[Y==cls]) for cls in range(10)]
                sens[yi] = si
                counts[yi] += 1
        #print sens
        #print counts
        counts[counts==0] += 1
        sens /= counts
#        for i in range(10):
#            for j in range(10):
#                sens[i,j] = sens[j,i]
        print numpy.log(sens).min()/numpy.log(10),numpy.log(sens).max()/numpy.log(10)
        from matplotlib.colors import LogNorm
        #im = ax[1-rand].matshow(numpy.log(sens),norm=LogNorm(vmin=1e-17, vmax=1e-12)) # vmin=-17,vmax=-12.5,
        im = ax[1-rand].matshow(sens,norm=LogNorm(vmin=10**(-7.4), vmax=10**(-5.4))) # vmin=-17,vmax=-12.5,

    f.subplots_adjust(right=0.88)
    cbar_ax = f.add_axes([0.90, 0.15, 0.02, 0.7])
    f.colorbar(im, cax=cbar_ax)
    pp.savefig('plots/emmanuel_class_sensitivity_both%s'%(ext))
                
        
def plot_avg_grads():
    print 'avg grads'
    names = {'[32, 1, 32, 1000, 1, 1]': 'realX_1layer_32hid_1000ex',
             '[16, 2, 32, 1000, 1, 1]': 'realX_2layer_16hid_1000ex',
             '[16, 2, 32, 1000, 0, 1]': 'randX_2layer_16hid_1000ex',
             '[32, 1, 32, 1000, 0.5, 1]': 'semirealX_1layer_32hid_1000ex',
             '[32, 1, 32, 1000, 0, 1]': 'randX_1layer_32hid_1000ex',
             '[16, 2, 32, 2000, \'100ex\', 1]': '100ex_2layer_16hid_2000ex',}
    configs = dict((i,[]) for i in names.values())
    print configs
    for r in results:
        if r[-1][0] == 0:
            g,e,l,end_grads,cfg = r
            n = cfg[3]
        if r[-1][0] == 1:
            g,e,l,end_grads,ys,cfg = r
            n = cfg[3]
        cfg_no_n = str(cfg[1:3]+cfg[4:])
        if cfg_no_n not in names:
            raise ValueError(cfg_no_n)
        name = names[cfg_no_n]
        idxes = [i[1] for i in sorted(zip(g.mean(axis=0), range(len(g[0]))), key=lambda x:x[0])]
        
        configs[name].append([n, g.mean(axis=0)[idxes], g.max(axis=0)[idxes], g.min(axis=0)[idxes]])
    for name, runs in configs.iteritems():
        pp.clf()
        maxn = max([n for n, mean, mmax, mmin in runs])
        for n, mean, mmax, mmin in sorted(runs, key=lambda x:x[0]):
            pp.plot(mean, label='$T = %d$'%n, color=jet_cm(numpy.log(n)/numpy.log(maxn)))
            #pp.fill_between(range(len(mean)), mmax, mmin, color=jet_cm(numpy.log(n)/numpy.log(maxn)), alpha=0.15)
        pp.gca().set_yscale('log')
        pp.ylim(1e-8,1e-3)
        pp.legend(loc='lower center', ncol=2)
        pp.xlabel('dataset axis')
        pp.ylabel('$\\bar{g}_x$')
        pp.savefig('plots/%s%s'%(name,ext))
        
def plot_avg_grads_vs_loss():
    print 'avg grads'
    names = {'[32, 1, 32, 1000, 1, 1]': 'realX_1layer_32hid_1000ex',
             '[16, 2, 32, 1000, 1, 1]': 'realX_2layer_16hid_1000ex',
             '[16, 2, 32, 1000, 0, 1]': 'randX_2layer_16hid_1000ex',
             '[32, 1, 32, 1000, 0.5, 1]': 'semirealX_1layer_32hid_1000ex',
             '[32, 1, 32, 1000, 0, 1]': 'randX_1layer_32hid_1000ex',
             '[16, 2, 32, 2000, \'100ex\', 1]': '100ex_2layer_16hid_2000ex',}
    configs = dict((i,[]) for i in names.values())
    print configs
    for r in results:
        if r[-1][0] == 0:
            g,e,l,end_grads,cfg = r
            n = cfg[3]
        if r[-1][0] == 1:
            g,e,l,end_grads,ys,cfg = r
            n = cfg[3]
        cfg_no_n = str(cfg[1:3]+cfg[4:])
        if cfg_no_n not in names:
            raise ValueError(cfg_no_n)
        name = names[cfg_no_n]
        idxes = [i[1] for i in sorted(zip(g.mean(axis=0), range(len(g[0]))), key=lambda x:x[0])]
        
        configs[name].append([n, g.mean(axis=0)[idxes], g.max(axis=0)[idxes], g.min(axis=0)[idxes]])
    for name, runs in configs.iteritems():
        pp.clf()
        maxn = max([n for n, mean, mmax, mmin in runs])
        for n, mean, mmax, mmin in sorted(runs, key=lambda x:x[0]):
            pp.plot(mean, label='$T = %d$'%n, color=jet_cm(numpy.log(n)/numpy.log(maxn)))
            #pp.fill_between(range(len(mean)), mmax, mmin, color=jet_cm(numpy.log(n)/numpy.log(maxn)), alpha=0.15)
        pp.gca().set_yscale('log')
        pp.ylim(1e-8,1e-3)
        pp.legend(loc='lower left')
        pp.xlabel('dataset axis')
        pp.ylabel('$\\bar{g}_x$')
        pp.savefig('plots/grads_vs_loss_%s%s'%(name,ext))
        

def gini(x):
    print 'gini',x.shape
    return np.sum(abs(x[:,None] - x[None,:])) / (2 * x.shape[0] * x.sum())

def plot_gini():
    print 'gini'
    names = {'[32, 1, 32, 1000, 1, 1]': 'realX_1layer_32hid_1000ex',
             '[16, 2, 32, 1000, 1, 1]': 'realX_2layer_16hid_1000ex',
             '[16, 2, 32, 1000, 0, 1]': 'randX_2layer_16hid_1000ex',
             '[32, 1, 32, 1000, 0.5, 1]': 'semirealX_1layer_32hid_1000ex',
             '[32, 1, 32, 1000, 0, 1]': 'randX_1layer_32hid_1000ex',
             '[16, 2, 32, 2000, \'100ex\', 1]': '100ex_2layer_16hid_2000ex',
             '[32, 1, 32, 1000, 1, \'I\']': 'realX_IY_1layer_32hid_1000ex',
             '[32, 1, 32, 1000, 0, \'I\']': 'randX_IY_1layer_32hid_1000ex',}
    configs = dict((i,[]) for i in names.values())
    print configs
    for r in results:
        if r[-1][0] == 0:
            g,e,l,end_grads,cfg = r
            n = cfg[3]
        if r[-1][0] == 1:
            g,e,l,end_grads,ys,cfg = r
            n = cfg[3]
        cfg_no_n = str(cfg[1:3]+cfg[4:])
        if cfg_no_n not in names:
            raise ValueError(cfg_no_n)
        name = names[cfg_no_n]
        idxes = [i[1] for i in sorted(zip(g.mean(axis=0), range(len(g[0]))), key=lambda x:x[0])]
        
        configs[name].append([n, g.mean(axis=0)[idxes], g.max(axis=0)[idxes], g.min(axis=0)[idxes]])
    markers='so^'
    xs = []
    ys = []
    labels = []
    for name, runs in configs.iteritems():
        if '1layer_32hid_1000ex' not in name:
            continue
        if 'IY' in name:
            continue
        print name
        pp.clf()
        maxn = max([n for n, mean, mmax, mmin in runs])
        st = sorted(runs, key=lambda x:x[0])
        ginis = [gini(mean) for n, mean, mmax, mmin in st]
        ys.append(ginis)
        labels.append(name.replace('_',' '))
        xs.append([n for n, mean, mmax, mmin in st])
    labels = ['real data', '50\% real data', 'random data']
    colors = 'bgr'
    for x,y,l,m,c in zip(xs,ys,labels,markers,colors):
        pp.plot(x,y,m+'-',label=l,c=c,mec=c)
        #for n, mean, mmax, mmin in sorted(runs, key=lambda x:x[0]):
        #    pp.plot(mean, label='$T = %d$'%n, color=jet_cm(numpy.log(n)/numpy.log(maxn)))
        #    #pp.fill_between(range(len(mean)), mmax, mmin, color=jet_cm(numpy.log(n)/numpy.log(maxn)), alpha=0.15)
    pp.gca().set_xscale('log')
    pp.xlim(x[0],x[-1])
    pp.legend(loc='best')#loc='lower center', ncol=2)
    pp.xlabel('number of SGD steps (log scale)')
    pp.ylabel('Gini coefficient of $\\bar{g}_x$ distribution')
    pp.savefig('plots/emmanuel_gini_overall%s'%(ext,))

    xs = []
    ys = []
    labels = []
    for name, runs in configs.iteritems():
        if 'IY' not in name:
            continue
        print name
        pp.clf()
        maxn = max([n for n, mean, mmax, mmin in runs])
        st = sorted(runs, key=lambda x:x[0])
        ginis = [gini(mean) for n, mean, mmax, mmin in st]
        ys.append(ginis)
        labels.append(name.replace('_',' '))
        xs.append([n for n, mean, mmax, mmin in st])
    labels = ['real data', 'random data']
    markers = 's^'
    colors = 'br'
    for x,y,l,m,c in zip(xs,ys,labels,markers,colors):
        pp.plot(x,y,m+'-',label=l,c=c,mec=c)
        #for n, mean, mmax, mmin in sorted(runs, key=lambda x:x[0]):
        #    pp.plot(mean, label='$T = %d$'%n, color=jet_cm(numpy.log(n)/numpy.log(maxn)))
        #    #pp.fill_between(range(len(mean)), mmax, mmin, color=jet_cm(numpy.log(n)/numpy.log(maxn)), alpha=0.15)
    pp.gca().set_xscale('log')
    pp.xlim(x[0],x[-1])
    pp.legend(loc='best')#loc='lower center', ncol=2)
    pp.xlabel('number of SGD steps (log scale)')
    pp.ylabel('Gini coefficient of $\\bar{g}_x$ distribution')
    pp.savefig('plots/emmanuel_gini_IY%s'%(ext,))

def plot_gini_vs_propy():
    print 'gini vs propy'
    configs = dict(((i/5.,n),[]) for i in range(6)
                   for n in [500,1200,5000,10000,30000])
    print configs
    for r in results:
        if r[-1][0] == 0:
            g,e,l,end_grads,cfg = r
            n = cfg[3]
        if r[-1][0] == 1:
            g,e,l,end_grads,ys,cfg = r
            n = cfg[3]
        print cfg,n
        cfg_no_n = str(cfg[1:3]+cfg[4:])
        #idxes = [i[1] for i in sorted(zip(g.mean(axis=0), range(len(g[0]))), key=lambda x:x[0])]
        
        configs[(cfg[-1],n)] = [n, g.mean(axis=0)]#, g.max(axis=0)[idxes], g.min(axis=0)[idxes]]
    markers='so^.><'
    colors = 'bgrcmy'
    xs = []
    ys = []
    labels = [] 
    for i in range(6):
        xs = [500,1200,5000,10000,30000]
        ys = []
        for n in xs:
            propy = i/5.
            ys.append(gini(configs[(propy,n)][1]))
        pp.plot(xs,ys,markers[i]+'-',c=colors[i],label='%d\\%% random Ys'%(100-propy*100))
        # print i
        if not i :print '\t'.join(map(str,xs))
        print '\t'.join(map(str,ys))
    pp.gca().set_xscale('log')
    pp.xlim(xs[0],xs[-1])
    pp.ylim(0,0.7)
    pp.legend(loc='best')#loc='lower center', ncol=2)
    pp.xlabel('number of SGD steps (log scale)')
    pp.ylabel('Gini coefficient of $\\bar{g}_x$ distribution')
    pp.savefig('plots/emmanuel_gini_vspropy_overall%s'%(ext,))

    pp.clf()
    for i in range(6):
        xs = [500,1200,5000,10000,30000]
        ysreal = []
        ysfake = []
        for n in xs:
            propy = i/5.
            ysreal.append(gini(configs[(propy,n)][1][:int(1000*propy)]))
            ysfake.append(gini(configs[(propy,n)][1][int(1000*propy):]))
        pp.plot(xs,ysreal,markers[i]+'-',c=colors[i],label='%d\\%% random Ys'%(100-propy*100))
        pp.plot(xs,ysfake,markers[i]+'--',c=colors[i])#,label='%d%% random Ys'%(100-propy*100))
        # print i
        if not i :print '\t'.join(map(str,xs))
        print '\t'.join(map(str,ys))
    pp.gca().set_xscale('log')
    pp.xlim(xs[0],xs[-1])
    pp.legend(loc='best')#loc='lower center', ncol=2)
    pp.xlabel('number of SGD steps (log scale)\nDashed is Gini of fake examples, full is Gini of real examples')
    pp.ylabel('Gini coefficient of $\\bar{g}_x$ distribution')
    pp.savefig('plots/emmanuel_gini_vspropy_diff%s'%(ext,))

    pp.clf()
    for i in range(6):
        xs = [500]#[500,1200,5000,10000,30000]
        ysreal = []
        ysfake = []
        for n in xs:
            propy = i/5.
            g = configs[(propy,n)][1]
            indexes = sorted(range(1000), key=lambda j: g[j])
            y = np.ones(1000)
            y[:int(propy*1000)] = 0
            g = g[indexes]
            print g.mean()
            pp.scatter(range(1000), g,5,cmap='RdYlGn',marker=',', c=y[indexes],edgecolors=(0,0,0,0))
            #ysreal.append(gini([:int(1000*propy)]))
            #ysfake.append(gini(configs[(propy,n)][1][int(1000*propy):]))
        #pp.plot(xs,ysreal,markers[i]+'-',c=colors[i],label='%d\\%% random Ys'%(100-propy*100))
        #pp.plot(xs,ysfake,markers[i]+'--',c=colors[i])#,label='%d%% random Ys'%(100-propy*100))
        # print i
        if not i :print '\t'.join(map(str,xs))
        print '\t'.join(map(str,ys))
    pp.ylim(1e-8,1e-6)
    pp.gca().set_yscale('log')
    pp.xlim(0,1000)
    #pp.legend(loc='best')#loc='lower center', ncol=2)
    #pp.xlabel('number of SGD steps (log scale)\nDashed is Gini of fake examples, full is Gini of real examples')
    #pp.ylabel('Gini coefficient of $\\bar{g}_x$ distribution')
    pp.savefig('plots/emmanuel_distr_vspropy_diff%s'%(ext,))



#plot_class_sensitivity(0)
#plot_class_sensitivity(1)
#plot_class_sensitivity_both()
#plot_avg_grads()
#plot_gini()
plot_gini_vs_propy()
