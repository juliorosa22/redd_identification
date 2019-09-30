from basicLib import *
#from REDD_management import *


def discrete_dt(v2,v1):
    sv=v2-v1
    return (abs(sv[0]/sv[1]))

def discrete_dt2(y2,y1,t2,t1):
    return float((y2-y1)/(t2-t1))

def discrete2(y2,y1):
    return float((y2-y1))

def check_edge(dt1,dt2,epsl,epsr):

    if abs(dt1)>epsl and abs(dt2) >epsr:
        return True
    #if dt2>epsr:
    #    return True
    else:
        return False

def findsignalStates(signal,t,epsl,epsr):

    if signal.shape[0] == t.shape[0]:

        v_segs=[]
        v_tsegs=[]
        seg = []
        tseg=[]
        seg.append(signal[0])
        tseg.append(t[0])

        for i in range(1,signal.shape[0]):
            if i < signal.shape[0] -1:
                #vo=np.array([signal[i-1], t[i-1]],np.float64)
                #va=np.array([signal[i], t[i]],np.float64)
                #vn=np.array([signal[i+1], t[i+1]],np.float64)
                dt1=discrete2(signal[i],signal[i-1])
                dt2=discrete2(signal[i+1],signal[i-1])
                is_edge = check_edge(dt1,dt2,epsl,epsr)
                if is_edge:
                    n = len(seg)
                    array_seg= np.array(seg,dtype=np.float64)
                    array_t = np.array(tseg,dtype=np.int32)
                    p_med= array_seg.mean()
                    v_segs.extend(np.full(n,p_med,dtype=np.float64))
                    v_tsegs.extend(array_t)

                    del seg[:]
                    del tseg[:]
                    seg=[]
                    tseg=[]
                    seg.append(signal[i])
                    tseg.append(t[i])
                else:
                    seg.append(signal[i])
                    tseg.append(t[i])

            else:
                seg.append(signal[i])
                tseg.append(t[i])
                n = len(seg)
                array_seg= np.array(seg,dtype=np.float64)
                array_t = np.array(tseg,dtype=np.int32)
                p_med= array_seg.mean()
                v_segs.extend(np.full(n,p_med,dtype=np.float64))
                v_tsegs.extend(array_t)
                del seg[:]
                del tseg[:]


        v_segs = np.array(v_segs,dtype= np.float64)
        v_tsegs = np.array(v_tsegs,dtype= np.float64)
        return v_segs,v_tsegs

    else:
        print('Invalid sizes of signal and time')

def verify_sequence(wsize,v_sig):
    fv = v_sig[0]
    vseq = np.arange(fv,fv+wsize)
    i_check = np.equal(v_sig,vseq)
    return i_check

def getGofCoef(obs,expect):

    v_diff=obs-expect
    v_square= v_diff*v_diff
    term =v_square/expect
    gof = term.sum()
    return gof

def getGofEdge(signal,t,treshold,wsize):
    p_seg =[]
    t_seg=[]
    p_edges=[]
    t_edges=[]
    #p_seg.extend(signal[:wsize])
    #t_seg.extend(t[:wsize])
    #t_aux.extend(np.zeros(wsize))
    for i in range(signal.shape[0]-(wsize)):
        npred = i+wsize
        nd = (i+1)+wsize
        pw = signal[i:npred]
        dw = signal[(i+1):nd]
        cf = getGofCoef(dw,pw)
        if cf>treshold:
            #cf.std()

            p_edges.append(signal[nd-1])
            t_edges.append(t[nd-1])
            '''
            if len(p_seg)==wsize:
                p_edges.append(p_seg[1])
                t_edges.append(t_seg[1])
                del p_seg[:]
                del t_seg[:]
            else:
                p_seg.append(signal[nd-1])
                t_seg.append(t[nd-1])
            '''
    tv = np.array(t_edges,dtype=np.float64)
    pv = np.array(p_edges,dtype = np.float64)
    return pv,tv

def getDtEdges(signal,sig_t,epsl,epsr):
    #print(signal.shape)
    edges =[]
    times=[]

    for i in range(1,signal.shape[0] -1):
        #dt1=discrete2(signal[i],signal[i-1])
        #dt2=discrete2(signal[i+1],signal[i-1])
        dt1=discrete_dt2(signal[i],signal[i-1],sig_t[i],sig_t[i-1])
        dt2=discrete_dt2(signal[i+1],signal[i-1],sig_t[i+1],sig_t[i-1])
        is_edge = check_edge(dt1,dt2,epsl,epsr)
        if is_edge:
            edges.append(signal[i])
            times.append(sig_t[i])

    edges = np.array(edges)
    times = np.array(times)
    return edges,times

def getMostSigPower(signal,t,treshold,wsize,max_len,min_len):
    print('FILTRANDO3')
    v_msignal = np.zeros(signal.shape[0],dtype = np.float64)
    sig_t = np.arange(signal.shape[0])
    p_seg =[]
    t_seg=[]
    order_seg=[]
    flag_upper = False
    flag_desc = False

    #p_seg.extend(signal[:wsize])
    #t_seg.extend(t[:wsize])
    #t_aux.extend(np.zeros(wsize))
    for i in range(signal.shape[0]-(wsize)):
        npred = i+wsize
        nd = (i+1)+wsize
        pw = signal[i:npred]
        dw = signal[(i+1):nd]
        cf = getGofCoef(dw,pw)
        i_first =[]
        i_last=[]
        if cf>treshold:
            if not flag_upper:
                if len(i_first)<wsize:
                    i_first.append(nd-1)
                else:
                    v_i=verify_sequence(wsize,np.array(i_first))
                    idx=np.where(v_i == False)[0]
                    if len(idx>0):
                        i_first = [ i_first[idx[k]] for k in range(len(idx)) ]
                    else:
                        flag_upper = True
            elif  not flag_desc == False :
                if len(i_last)<wsize:
                    i_last.append(nd-1)
                else:
                    v_i=verify_sequence(wsize,np.array(i_last))
                    idx=np.where(v_i == False)[0]
                    if len(idx>0):
                        i_last = [ i_last[idx[k]] for k in range(len(idx)) ]
                    else:
                        flag_desc = True
            else:
                print('aki')
                v_aux =signal[i_first[0]:i_last[0]+1]
                if v_aux.shape[0] > max_len and v_aux.shape[0]< min_len :
                    i_first = i_last
                    del i_last[:]
                    flag_desc = False
                else:
                    v_msignal[i_first[0]:i_last[0]+1] = v_aux
                    del i_first[:]
                    del i_last[:]
                    flag_upper = False
                    flag_desc = False


    return v_msignal,sig_t

def filter_matrix(Msig):
    print('FILTRANDO2')
    t = np.arange(3600)
    #print(t.shape)
    M_fs =np.empty((1,Msig.shape[1]))
    #print(Msig.shape)
    for i in range(Msig.shape[0]):
        s = Msig[i]
        fs,tf = findsignalStates(s,t,20,50)
        fs = np.reshape(fs,(1,fs.shape[0]))
        M_fs = np.append(M_fs,fs,0)

    return M_fs[1:]


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
