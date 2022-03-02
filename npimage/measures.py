import numpy as np

def sum(x, labels=None, index=None):
    if labels is None: return x.sum()
    if index is None:
        labels, index = np.minimum(labels, 1), 1
    bins = np.zeros(labels.max()+1, dtype=x.dtype)
    np.add.at(bins, labels, x)
    return bins[index]

def mean(x, labels=None, index=None):
    if labels is None: return x.mean()
    if index is None:
        labels, index = np.minimum(labels, 1), 1
    acc = np.bincount(labels.ravel(), x.ravel())
    acc /= np.bincount(labels.ravel(), minlength=len(acc))
    return acc[index]

def minimum(x, labels=None, index=None):
    if labels is None: return x.min()
    if index is None:
        labels, index = np.minimum(labels, 1), 1
    bins = np.zeros(labels.max()+1, dtype=x.dtype)
    bins[:] = x.max()
    np.minimum.at(bins, labels, x)
    return bins[index]

def maximum(x, labels=None, index=None):
    if labels is None: return x.min()
    if index is None:
        labels, index = np.minimum(labels, 1), 1
    bins = np.zeros(labels.max()+1, dtype=x.dtype)
    bins[:] = x.min()
    np.maximum.at(bins, labels, x)
    return bins[index]

def variance(x, labels=None, index=None):
    ex2 = mean(x**2, labels, index)
    e2x = mean(x, labels, index)**2
    return ex2 - e2x
    
def center_of_mass(x, labels=None, index=None):
    rcs = np.mgrid[tuple([slice(i) for i in x.shape])]
    weight = sum(x, labels, index)
    s = np.array([sum(i*x, labels, index) for i in rcs])
    return (s / weight).T

def find_objects(lab, maxlab=0):
    index = np.arange(1, (maxlab or lab.max())+1)
    rcs = np.mgrid[tuple([slice(i) for i in lab.shape])]
    rcs1 = [minimum(i, lab, index) for i in rcs]
    rcs2 = [maximum(i, lab, index) for i in rcs]
    for i in rcs2: i += 1
    arr = np.array([rcs1, rcs2]).transpose(2,1,0)
    return [tuple([slice(*j) for j in i ]) for i in arr]

if __name__ == '__main__':
    import scipy.ndimage as ndimg
    
    x =   np.array([1,2,3,4,5,6,7,8])
    lab = np.array([0,1,1,2,2,2,0,2])
    x.shape = lab.shape = (2,-1)
    index = np.array([2,0,1])

    print('sum test:')
    print(ndimg.sum(x, lab, index))
    print(sum(x, lab, index))

    print('mean test:')
    print(ndimg.mean(x, lab, index))
    print(mean(x, lab, index))

    print('minimum test:')
    print(ndimg.minimum(x, lab, index))
    print(minimum(x, lab, index))

    print('maximum test:')
    print(ndimg.maximum(x, lab, index))
    print(maximum(x, lab, index))

    print('variance test:')
    print(ndimg.variance(x, lab, index))
    print(variance(x, lab, index))

    print('center_of_mass test:')
    print(ndimg.center_of_mass(x, lab, index))
    print(center_of_mass(x, lab, index))

    print('find_objects test:')
    print(ndimg.find_objects(lab))
    print(find_objects(lab))
    
