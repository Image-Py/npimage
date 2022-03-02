import numpy as np
from .filters import convframe

def binary_dilation(input, weight=None, output=None, keeptype=False): 
    def f(buf, output, w):
        if w>0: output |= buf
    return convframe(input, weight, output, None, 'reflect', 'bool', keeptype, f)

def binary_erosion(input, weight=None, output=None, keeptype=False): 
    def f(buf, output, w):
        if w>0: output &= buf
    return convframe(input, weight, output, None, 'reflect', 'bool', keeptype, f)

def binary_opening(input, weight=None, output=None, keeptype=False): 
    a = binary_erosion(input, weight, output)
    return binary_dilation(a, weight, output)

def binary_closeing(input, weight=None, output=None, keeptype=False): 
    a = binary_dilation(input, weight, output)
    return binary_erosion(a, weight, output)

if __name__ == '__main__':
    from skimage.data import camera
    import matplotlib.pyplot as plt
    
    img = camera() > 128
    simg = binary_closeing(img, weight=np.ones((10,10)))
    plt.imshow(simg, cmap='gray')
    plt.show()
