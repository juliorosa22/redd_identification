from basicLib import *
import matplotlib.pyplot as plt

### Funcoes que geram as combinações para os rotulos dos sinais
#a matriz de combinação é a forma canonica de representar os dados
def genComb(v,i,my_one,N,K,max_p,l_comb): #funcao que gera a matriz de combinacoes de N aparelhos K a K
    if my_one < K:
        for j in range(max_p):
            v[i+j]=1
            genComb(v,(i+j+1),my_one+1,N,K,max_p-j,l_comb)
            v[i+j]=0
    else:
        for j in range(max_p):
            v[i+j]=1
            l_comb.append(np.array(v,dtype = np.float64))
            v[i+j]=0


def combMatrix(min,N):#N representa o numero total de elementos a serem combinados
    comb_M = np.zeros((1,N),dtype = np.float64)
    l=[]
    v=[0 for j in range(N)]
    genComb(v,0,1,N,min,(N-min+1),l)
    comb_M = np.array(l)
    '''
    for i in range(min,N+1):
        l=[]
        v=[0 for j in range(N)]
        genComb(v,0,1,N,i,(N-i+1),l)

        comb_M=np.append(comb_M ,np.array(l,dtype = np.float64),0)
    comb_M=comb_M[1:]
    '''
    return comb_M

def labelTransformRaw2Int(label):
    idx=np.where(label>0)[0]
    bases = 2**idx
    num_hot = bases.sum()-1
    return num_hot

def labelGetIntLabels(Mlabels):
    lhot=[]
    for i in range(Mlabels.shape[0]):
        comb=Mlabels[i]
        nhot=labelTransformRaw2Int(comb)
        lhot.append(nhot)
    vhot=np.array(lhot)
    return vhot


def detect_onset(x, threshold=0, n_above=1, n_below=0,threshold2=None, n_above2=1, show=False, ax=None):
    """Detects onset in data based on amplitude threshold.

    Parameters
    ----------
    x : 1D array_like
        data.
    threshold : number, optional (default = 0)
        minimum amplitude of `x` to detect.
    n_above : number, optional (default = 1)
        minimum number of continuous samples >= `threshold`
        to detect (but see the parameter `n_below`).
    n_below : number, optional (default = 0)
        minimum number of continuous samples below `threshold` that
        will be ignored in the detection of `x` >= `threshold`.
    threshold2 : number or None, optional (default = None)
        minimum amplitude of `n_above2` values in `x` to detect.
    n_above2 : number, optional (default = 1)
        minimum number of samples >= `threshold2` to detect.
    show  : bool, optional (default = False)
        True (1) plots data in matplotlib figure, False (0) don't plot.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    inds : 1D array_like [indi, indf]
        initial and final indeces of the onset events.

    Notes
    -----
    You might have to tune the parameters according to the signal-to-noise
    characteristic of the data.

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectOnset.ipynb

    
    """

    x = np.atleast_1d(x).astype('float64')
    # deal with NaN's (by definition, NaN's are not greater than threshold)
    x[np.isnan(x)] = -np.inf
    # indices of data greater than or equal to threshold
    inds = np.nonzero(x >= threshold)[0]
    if inds.size:
        # initial and final indexes of almost continuous data
        inds = np.vstack((inds[np.diff(np.hstack((-np.inf, inds))) > n_below+1], \
                          inds[np.diff(np.hstack((inds, np.inf))) > n_below+1])).T
        # indexes of almost continuous data longer than or equal to n_above
        inds = inds[inds[:, 1]-inds[:, 0] >= n_above-1, :]
        # minimum amplitude of n_above2 values in x to detect
        if threshold2 is not None and inds.size:
            idel = np.ones(inds.shape[0], dtype=bool)
            for i in range(inds.shape[0]):
                if np.count_nonzero(x[inds[i, 0]: inds[i, 1]+1] >= threshold2) < n_above2:
                    idel[i] = False
            inds = inds[idel, :]
    if not inds.size:
        inds = np.array([])  # standardize inds shape for output
    if show and x.size > 1:  # don't waste my time ploting one datum
        _plot(x, threshold, n_above, n_below, threshold2, n_above2, inds, ax)

    return inds