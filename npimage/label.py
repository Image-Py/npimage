import numpy as np

# build all the conflict vector
def build_conflict(img, idx, cell):
    drc = np.array(np.where(cell)).T-1
    strides = np.cumprod((1,)+img.shape[:0:-1])[::-1]
    dis = np.array(drc) @ strides
    idxb = (idx + dis[:,None]).ravel()
    idxa = np.hstack([idx]*len(dis))
    idxab = np.vstack([idxa, idxb]).T
    return idxab

# solve conflict one step
def step(img, confs):
    cmap = img.ravel()[confs]
    cmap = np.sort(cmap, axis=1)
    msk = cmap[:,0] > 0 # 不指向0
    msk &= cmap[:,0] < cmap[:,1] # 不指向自己

    if msk.sum()==0: return None, []
    confs = confs[msk]
    cmap = cmap[msk]
    lut = np.arange(img.max()+1)
    lut[cmap[:,1]] = cmap[:,0]

    i = 0
    while True:
        i += 1; # print(i)
        oldlut = lut
        lut = lut[lut]
        if np.sum(oldlut!=lut)==0: break
    return lut, confs

# relabel the image from 1,2,3...
def relabel(img):
    vs, idx = np.unique(img, False, True)
    return idx.reshape(img.shape), len(vs)-1

# solve conflict
def solve(img, confs):
    while True:
        lut, confs = step(img, confs)
        if len(confs)==0: break
        img = lut[img]
    img, n = relabel(img)
    return img[(slice(1,-1),)*img.ndim], n

# label image
def label(img, structure=None):
    if structure is None:
        structure = np.ones((3,)*img.ndim)
    cell = structure.copy()
    cell.ravel()[cell.size//2:] = 0
    padimg = np.pad(img, 1).astype('int32')
    idx = np.where(padimg.ravel())[0].astype('int32')
    padimg.ravel()[idx] = np.arange(1, len(idx)+1)
    confs = build_conflict(padimg, idx, cell)
    return solve(padimg, confs)
    
if __name__ == '__main__':
    from imageio import imread
    import scipy.ndimage as ndimg
    import matplotlib.pyplot as plt
    import scipy.ndimage as ndimg
    from time import time

    data = np.array([
        [0,1,0,0,0,0,0,0],
        [0,1,1,0,0,0,1,1],
        [0,0,0,0,1,0,1,0],
        [0,0,0,1,1,1,0,0],
        [0,0,0,0,1,0,0,0]])

    print('\ntest data:\n', data)
    lab1, n1 = label(data, np.array([[1,1,1],[1,1,1],[1,1,1]]))
    print('\n8-connect label:\n', lab1)
    
    lab2, n2 = label(data, np.array([[0,1,0],[1,1,1],[0,1,0]]))
    print('\n4-connect label:\n', lab2)
