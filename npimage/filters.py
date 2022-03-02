import numpy as np

# general convolve framework
def convframe(input, weight, output=None, init=0,
        mode='reflect', buffertype=None, keeptype=True, func=None):
    if output is None:
        output = np.zeros(input.shape, buffertype or input.dtype)
        output[:] = input if init is None else init
    buf = np.zeros_like(output)
    coreshp = weight.shape; coremar = np.array(weight.shape)//2
    padimg = np.pad(input, [(i,i) for i in coremar], mode=mode)

    rcs = np.mgrid[tuple([slice(i) for i in coreshp])]
    rcs = rcs.reshape(input.ndim, -1).T
    for idx, w in zip(rcs, weight.ravel()):
        start, end = idx, idx + input.shape
        s = [slice(i,j) for i,j in zip(start, end)]
        buf[:] = padimg[tuple(s)]
        func(buf, output, w)
    return output.astype(input.dtype) if keeptype else output

# split convolve in axis
def axisframe(img, core, mode='reflect', f=None):
    dtype = img.dtype
    for i in range(len(core)):
        shape = np.ones(img.ndim, dtype=np.int8)
        shape[i] = -1
        if core[i].size == 1:
            img = img * core[i]
            continue
        c = core[i].reshape(shape)
        print(c.shape)
        img = f(img, c, output=None, mode=mode, keeptype=False)
    return img.astype(dtype)

def convolve(input, weight, output=None, mode='reflect', keeptype=True):
    def f(buf, output, w): buf *= w; output += buf
    return convframe(input, weight, output, 0, mode, 'float32', keeptype, f)

def uniform_filter(img, size=3, mode='reflect'):
    if not hasattr(size, '__len__'): size = [size] * img.ndim
    def core(s):
        if s<=1: return np.array([1])
        return  np.ones(s).astype('float32')/s
    cores = [core(i) for i in size]
    return axisframe(img, cores, mode, convolve)
    
def gaussian_filter(img, sig=2, mode='reflect'):
    if not hasattr(sig, '__len__'): sig = [sig] * img.ndim
    def core(s):
        if s==0: return np.array([1])
        x = np.arange(-int(s*2.5+0.5), int(s*2.5+0.5)+1)
        return np.exp(-x**2/2/s**2)/s/(2*np.pi)**0.5
    cores = [core(i) for i in sig]
    return axisframe(img, cores, mode, convolve)

def _maximum_filter(input, weight=None, output=None, mode='reflect', keeptype=True): 
    def f(buf, output, w):
        if w>0: np.maximum(buf, output, out=output)
    return convframe(input, weight, output, None, mode, None, keeptype, f)

def maximum_filter(input, size=None, footprint=None, output=None, mode='reflect', keeptype=True):
    if not footprint is None:
        return _maximum_filter(input, footprint, output, mode)
    if not hasattr(size, '__len__'): size = [size]*input.ndim
    cores = [np.ones(i, 'bool') for i in size]
    return axisframe(input, cores, mode, _maximum_filter)

def _minimum_filter(input, weight=None, output=None, mode='reflect', keeptype=True): 
    def f(buf, output, w):
        if w>0: np.minimum(buf, output, out=output)
    return convframe(input, weight, output, None, mode, None, keeptype, f)

def minimum_filter(input, size=None, footprint=None, output=None, mode='reflect', keeptype=True):
    if not footprint is None:
        return _minimum_filter(input, footprint, output, mode)
    if not hasattr(size, '__len__'): size = [size]*input.ndim
    cores = [np.ones(i, 'bool') for i in size]
    return axisframe(input, cores, mode, _minimum_filter)

if __name__ == '__main__':
    from skimage.data import camera
    import matplotlib.pyplot as plt
    
    img = camera()
    simg = minimum_filter(img, footprint=np.ones((10,10)))
    plt.imshow(simg, cmap='gray')
    plt.show()
