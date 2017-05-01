import multiprocessing as mp
import time
import numpy
import ctypes

def f(i,x):
    L,ls = x
    print L,ls
    l = numpy.frombuffer(L.get_obj(),'float32')
    for j in range(10):
        l[i] += j
    print i,l

def main():
    L = mp.Array(ctypes.c_float, 10)
    l = numpy.frombuffer(L.get_obj(),dtype='float32')

    for i in range(10):
        p = mp.Process(target=f, args=(i, (L,l.shape)))
        p.start()
        #f(i,L)
    time.sleep(1)
    print 'end',l
    
print 'name:',__name__
if __name__ == '__main__':
    main()

