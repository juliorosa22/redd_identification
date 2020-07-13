from basicLib import *
#import pandas as pd
from tools_csv import *
from filters import *
from apiFilter import *
#from tsfresh.feature_extraction import extract_features, MinimalFCParameters
from pylab import *


global_nsamples=3600
n_max_signals = 500



dir_path = '/home/julio/Framework_Projeto/Data_sets/ANN_data'
global_dsetPath =dir_path+'/Unified_Databases'



#######
#funcoes utilizadas na soma dos sinais e normalizacao

def norm_sig2(sig,sz):
    n = sig.shape[0]

    if n>sz:
        sig = sig[:sz]
    nzeros = sz - sig.shape[0]
    rsz = math.ceil(nzeros/2)
    lsz = math.floor(nzeros/2)
    nsig = np.append(np.zeros(lsz),np.append(sig,np.zeros(rsz)))
    return nsig


def checkAdjust(ip,ls,rs,max_size):
    return (ip-ls)>=0 and (ip+rs)<max_size

def adjustSig(sig,sz):#ajusta o maior sinal para o tamanho do menor para q a soma seja realizada
    ic = np.random.randint(sig.shape[0])
    ls = math.floor(sz/2)
    rs =math.ceil(sz/2)
    while not checkAdjust(ic,ls,rs,sig.shape[0]):
        ic = np.random.randint(sig.shape[0])
    newsig = sig[(ic -ls):(ic+rs)]
    return newsig


def smartSum(s1,s2,norm_size):
    #print(s1.shape)
    #print(s2.shape)
    ix1 = np.where(s1>15)[0]
    ix2 = np.where(s2>15)[0]
    vs1=s1[ix1[0]:ix1[-1]+1]
    vs2=s2[ix2[0]:ix2[-1]+1]
    vmin = vs1
    vmax = vs2
    if vmin.shape[0] > vmax.shape[0]:
        aux  =vmax
        vmax = vmin
        vmin = aux
    if vmin.shape[0] < math.floor(vmax.shape[0]*0.8):
        #print('ajusta')
        #ajusta o maior sinal para o tamanho do menor
        vmax = adjustSig(vmax,vmin.shape[0])
    vmin = norm_sig2(vmin,norm_size)
    vmax = norm_sig2(vmax,norm_size)
    sumv = vmin+vmax
    #sumv = np.reshape(sumv,(1,sumv.shape[0]))
    return sumv

def plotV(v,name,i_n):
        nd = v.shape[0]
        s = v
        t =  np.linspace(0,nd,nd)#/3600#
        plt.figure(i_n)
        plt.plot(t,s)
        plt.title(name)
        plt.ylabel('Power(W)')
        plt.xlabel('t(hour)')

def checkFitSig(sig,win,i_pos):
        win_centroid = math.floor(win.shape[0]/2)
        szr =win[win_centroid:].shape[0]
        szl=win[:win_centroid].shape[0]
        rsig = sig[i_pos:].shape[0]
        lsig = sig[:i_pos].shape[0]
        return rsig >= szr and lsig >=szl

def fit(sig,win,i_pos):

    nv= np.zeros(sig.shape[0],dtype = np.float64)
    win_centroid = math.floor(win.shape[0]/2)
    szr =win[win_centroid:].shape[0]
    szl=win[:win_centroid].shape[0]
    nv[i_pos:i_pos+szr] =win[win_centroid:]
    nv[i_pos-szl :i_pos] =win[:win_centroid]
    nv = sig+nv
    nv = np.reshape(nv,(1,nv.shape[0]))
    return nv

def fast_conv(s1,s2):
    ms_1 = np.where(s1>10)[0]
    ms_2 = np.where(s2>10)[0]
    sig=[]
    win=[]
    #acc_win =[]
    acc_sig=[]

    if ms_1.shape[0]> ms_2.shape[0]   :
        win = s2[ms_2[0]:ms_2[-1]+1]
        sig = s1
        acc_sig=ms_1
        #acc_win=ms_2
    else:
        win = s1[ms_1[0]:ms_1[-1]+1]
        sig = s2
        acc_sig=ms_2
        #acc_win=ms_1

    c_aux = np.arange(acc_sig[0],acc_sig[-1]+1)
    #sig_centroid = c_aux[math.floor(c_aux.shape[0]/2)]
    sig_centroid =math.floor(sig.shape[0]/2)
    win_centroid=math.floor(win.shape[0]/2)
    l_pos=[]
    '''
    figure(1)
    plt.plot(sig)
    figure(2)
    plt.plot(win)
    '''


    #rint(sig_centroid,win_centroid)
    #plt.show()
    while len(l_pos)<2:
        p = np.random.randint(sig_centroid-50,sig_centroid+50)
        r=checkFitSig(sig,win,p)
        if r and p not in l_pos:
            l_pos.append(p)

    while len(l_pos)<7:
        p = np.random.randint(acc_sig[0],sig_centroid)
        if checkFitSig(sig,win,p) and p not in l_pos:
            l_pos.append(p)

    while len(l_pos)<12:
        p = np.random.randint(sig_centroid,acc_sig[-1])
        if  checkFitSig(sig,win,p) and p not in l_pos:
            l_pos.append(p)

    m_sum = np.empty((1,sig.shape[0]))
    for p in l_pos:
        nv=fit(sig,win,p)
        m_sum = np.append(m_sum,nv,0)
    m_sum=m_sum[1:]
    #print('msum shape: ',m_sum.shape)
    return m_sum


def norm_sig(sig,size):
    n = sig.shape[0]

    if n>size:
        sig = sig[:size]
    nzeros = size - sig.shape[0]
    rsz = math.ceil(nzeros/2)
    lsz = math.floor(nzeros/2)
    nsig = np.append(np.zeros(lsz),np.append(sig,np.zeros(rsz)))
    #nsig =np.reshape(nsig,(1,nsig.shape[0]))
    return nsig

def windowSliding(signal,window_size,ds):
    l_win_signals=[]
    if signal.shape[0]>window_size:
        sz=signal.shape[0]
        for i in range(sz):
            i_begin=i*ds
            i_end=i_begin+window_size
            if i_begin>sz or (sz-i_begin) < math.floor(0.7*window_size):

                break
            else:
                cut_sig = []
                if i_end>sz:
                    cut_sig = signal[i_begin:sz]
                    cut_sig = np.append(cut_sig,np.zeros((window_size - cut_sig.shape[0])))
                else:
                    cut_sig = signal[i_begin:i_end]
                l_win_signals.append(cut_sig)

        sliding_signals = np.vstack(l_win_signals)
        return sliding_signals
    else:
        #cut_sig = np.append(signal,np.zeros((window_size - signal.shape[0])))
        cut_sig = norm_sig(signal,window_size)
        cut_sig = np.reshape(cut_sig,(1,cut_sig.shape[0]))
        return cut_sig

def getDevSliding(M_dev,window_size,ds):
    sl_matrix = np.empty((1,window_size))
    for i in range(M_dev.shape[0]):
        sig = M_dev[i]
        i_pfw=np.where(sig>10)[0]
        pfw_sig = sig[i_pfw[0]:i_pfw[-1]]
        rM = windowSliding(pfw_sig,window_size,ds)
        sl_matrix = np.append(sl_matrix,rM,0)
    return sl_matrix[1:]


def normMatrix(M):
    Mn = np.empty((1,M.shape[1]))
    for i in range(M.shape[0]):
        s = M[i]
        ns = norm2(s,3600)
        Mn = np.append(Mn,ns,0)
    return Mn[1:]

def getMS(m_select):
    newM = []
    for i in range(m_select.shape[0]):
        lm = m_select[i]

        mi = np.where(lm>20)[0]
        if mi.shape[0]>0:
            vm = lm[mi[0]:mi[-1]]
            tm = np.arange(vm.shape[0])
            vm,tm = findsignalStates(vm,tm,20,50)
            vm = norm_sig(vm,m_select.shape[1])
            newM.append(vm)
    newM = np.vstack(newM)
    return newM

#isig indica no vetor do sinal quais posições são maiores que zero


### manipulam os sinais dos aparelhos para criar o banco de dados geral


def recSumNormMatrix(recDev,l_devs,norm_size,idev):
    np.random.shuffle(recDev)
    dev2 = l_devs[idev]
    dev2 =getRandomSigs(dev2,norm_size)
    np.random.shuffle(dev2)
    sumDevs =recDev+dev2
    if idev < (len(l_devs)-1):
        return recSumNormMatrix(sumDevs,l_devs,norm_size,(idev+1))
    else:
        return sumDevs

def genNSumsOfSignals(l_devs,norm_size,n_sums):

    times = math.ceil(n_sums/norm_size)
    rM =np.empty((1,l_devs[0].shape[1]))
    #print('times:',times)
    for i in range(times):
        Mdev=getRandomSigs(l_devs[0][:],norm_size)
        recSum=recSumNormMatrix(Mdev,l_devs,norm_size,1)
        rM = np.append(rM,recSum,0)
    return rM[1:]


##################
#Funcoes responsaveis por manipular sinais separadamente
##################
def reshapeFileData(source_file,filter_op=False): #funcao que usa um sinal do banco Redd e divide-o em sinais com espaçamentos de 1h
    signal_p=get_REDD_P_array(source_file)
    if filter_op:
        t = np.linspace(0,signal_p.shape[0],num=signal_p.shape[0])
        signal_p,x,z = signal_filter(signal_p,t,20,50)
    num_samples = global_nsamples #numero de amostras p cada sinal a ser dividido
    num_signals = math.floor( signal_p.size / num_samples)# O numero total de sinais dividos
    spaced_signals = signal_p[:(num_samples*num_signals)]
    rest_signals = signal_p[(num_samples*num_signals):]
    nrest = len(rest_signals)
    l_signals = np.split(spaced_signals,num_signals)
    if nrest > num_samples/2:
        z = np.zeros((1,(num_samples - nrest)))
        rest_signals = np.append(rest_signals,z)
        l_signals.append(rest_signals)
    M_signals = np.stack(l_signals) # cria uma matriz dim = num_signals X num_samples
    return M_signals
    #return signal_p[(num_samples*num_signals):]

def reshapeSignal(signal,size): #funcao que usa um sinal do banco Redd e divide-o em sinais com espaçamentos de 1h
    num_samples = size #numero de amostras p cada sinal a ser dividido
    num_signals = math.floor( signal.size / num_samples)# O numero total de sinais dividos
    spaced_signals = signal[:(num_samples*num_signals)]
    rest_signals = signal[(num_samples*num_signals):]
    nrest = len(rest_signals)
    l_signals = np.split(spaced_signals,num_signals)
    if nrest > num_samples/2:
        z = np.zeros((1,(num_samples - nrest)))
        rest_signals = np.append(rest_signals,z)
        l_signals.append(rest_signals)
    M_signals = np.stack(l_signals) # cria uma matriz dim = num_signals X num_samples
    return M_signals

#funcao que cria um arquivo hdf5 com os sinais de um aparelho especifico

def save_DevSignals(Ms,file_path):
    f = h5py.File(dir_path+file_path+'.hdf5','w')
    f_dset =f.create_dataset('signals',(Ms.shape[0],Ms.shape[1]),dtype='f8')
    f_dset[:]=Ms[:]
    f.close()

def read_DevSignals(file_path):
    f = h5py.File(dir_path+file_path+'.hdf5','r')
    key = list(f)[0]

    Ms =f[key][:]
    f.close()
    return Ms


def dsetForOneDevice(signals_files,house_dir,file_path):

    abs_path = house_dir+file_path
    print(abs_path)
    M = np.zeros((1,global_nsamples),dtype = np.float64)
    for s in signals_files:
        M=np.append(M,reshapeFileData(s,filter_op=False),0)
    M = M[1:]
    save_DevSignals(M,abs_path)

def getSelectedSignals(house_dir,prefix,devs_index = None):
    devs_names = getDevNames(house_dir,devs_index)
    devs_names = [prefix+devs_names[i] for i in range(len(devs_names))]
    l_devs =[read_DevSignals(house_dir+'/'+devs_names[i]) for i in range(len(devs_names)) ]
    return l_devs,devs_names

#dsetForOneDevice(['/house_1/channel_2.dat'],'/House_1','/mains2_h1')
#######

def getOnDevs(v_pos,dev_names):
    vi = np.where(v_pos>0)[0]
    print(vi)
    names = [ dev_names[k] for k in vi  ]
    return names

def plot_Mdev(M_s,title,i_n):
    nd = M_s.shape[0]*M_s.shape[1]
    s = np.reshape(M_s,nd)
    t =  np.arange(nd)
    plt.figure(i_n)
    plt.title(title)
    plt.ylabel('Power(W)')
    plt.xlabel('t(hour)')
    plt.plot(t,s)

    #plt.show()

def readOneDevice_signals(hdf5_name):
    f = h5py.File(dir_path+hdf5_name, 'r')
    power_signals = f['power_signals'][:]
    #print(hdf5_name)
    #print(power_signals.shape)
    #print(power_signals[:10])
    f.close()
    return power_signals

def getDevNames(house_path,devs_index=None):
    f  = open(dir_path+house_path+'/labels_names.txt')
    devices_names = list(f)
    f.close()
    devices_names = [ dev[:-1] for dev in devices_names ]
    devices_names = np.array(devices_names)
    if len(devs_index)>0:
        devices_names = devices_names[devs_index]
    return devices_names

def getHouseSignals(house_path,devs_index):
    devices_names = getDevNames(house_path,devs_index)
    power_signals = [read_DevSignals(house_path+'/'+dev_name) for dev_name in devices_names]
    return power_signals,devices_names

def getRandomSigs(Ms,nsigs):
    li = []

    if nsigs< Ms.shape[0]:
        while len(li)<nsigs:
            ix = np.random.randint(Ms.shape[0])
            if ix not in li:
                li.append(ix)

    else:
        li.extend(np.arange(Ms.shape[0]))
        sz_r = nsigs - Ms.shape[0]
        ix = np.random.randint(Ms.shape[0],size=sz_r)
        li.extend(ix)


    vix=np.array(li)
    rdM = np.empty(Ms[vix].shape)
    rdM[:] = Ms[vix]
    return rdM

#####################
##___ Funcoes responsaveis por manipular os banco de dados unificado
####################
##funcao que pega nsigs da matriz Ms
def aumentDim(Ms,nsigs):
    if nsigs > Ms.shape[0]:
        rM = np.empty((1,Ms.shape[1]))
        repeat_times= math.floor(nsigs/Ms.shape[0])
        for i in range(repeat_times):
            rM=np.append(rM,Ms[:],0)
        if (nsigs-rM.shape[0]-1)>0:
            rest = getRandomSigs(Ms,nsigs - (rM.shape[0]-1))
            rM = np.append(rM,rest,0)

        rM = rM[1:]
        #np.random.shuffle(rM)
        return rM
    else:
        return getRandomSigs(Ms,nsigs)

def saveUnifiedDset(database_name,X,Y,cod_M):
    f = h5py.File(global_dsetPath+database_name, 'w')
    inputs_dset= f.create_dataset("inputs",(X.shape[0],X.shape[1]),dtype='f8')
    inputs_dset[:]=X[:]

    cod_M=np.array(cod_M,dtype = np.int32)
    print(cod_M)
    cod_labels_dset =f.create_dataset("devices_names",(1,cod_M.shape[0]),dtype='i8')
    cod_labels_dset[:] = cod_M[:]

    labels_dset =f.create_dataset("labels",(Y.shape[0],Y.shape[1]),dtype='int')
    labels_dset[:] = Y[:]

    f.close()

def readUnifiedDset(dset_file,house_dir):#num_dev deve ser um numero entre 1 e 5
    f = h5py.File(global_dsetPath+dset_file, 'r')
    #print(list(f))
    X = f['inputs'][:]
    Y = f['labels'][:]
    cod_M = f['devices_names'][:]
    dev_names = getDevNames(house_dir,cod_M)
    #print(cod_M)
    f.close()
    return X,Y,dev_names,cod_M

def getIndexes(ydev,file_index):
    i_test = read_DevSignals(file_index)
    i_test = i_test.astype(int)
    i_test = np.reshape(i_test,i_test.shape[1])
    all_i=np.arange(ydev.shape[0])
    mask = np.logical_not(np.isin(all_i,i_test))
    i_train = all_i[mask]
    return i_train,i_test

def newOneHotLabels(label):
    idx=np.where(label>0)[0]
    bases = 2**idx
    num_hot = bases.sum()-1
    return num_hot

def getLabelsOneHot(Mlabels):
    lhot=[]
    for i in range(Mlabels.shape[0]):
        comb=Mlabels[i]
        nhot=newOneHotLabels(comb)
        lhot.append(nhot)
    vhot=np.array(lhot)
    return vhot

def generateCombMatrix(n_devs):
    combM =[]
    for i in range(1,n_devs+1):
        m=combMatrix(i,n_devs)
        m=m.astype(int)
        #combM = np.append(combM,m,0)
        combM.append(m)

    #combM=combM[1:]
    combM = np.vstack(combM)
    return combM

def convertLabels(Y_onehot,ndevs):
    Y = []
    #print(Y_onehot.shape)
    #print(Y_onehot[10])
    #combM = generateCombMatrix(ndevs)
    for i in range(Y_onehot.shape[0]):
        Y.append(decodeOneHot(Y_onehot[i],ndevs))
    Y = np.vstack(Y)
    return Y

def decodeOneHot(num_hot,num_devs):
    num_hot+=1
    bin_str=np.binary_repr(num_hot,width=num_devs)

    lb = [int(bin_str[i]) for i in range(len(bin_str))]
    lb = np.array(lb)
    label = np.flip(lb,0)
    return label

def getNMixSigs(s_sigs,combM,sizes):

    n_devs = combM.shape[1]
    X = []
    Y = []
    #for sz in sizes:
        #print('num_s: ',sz)
    for i in range(combM.shape[0]):
        cb = combM[i]
        #print(cb)
        i_cb = np.where(cb>0)[0]
        len_labels = len(i_cb)-1
        devs_selected_comb=[]
        accM=[]
        for k in range(i_cb.shape[0]):
            devs_selected_comb.append(s_sigs[i_cb[k]])

        accM = genNSumsOfSignals(devs_selected_comb,500,sizes)
        #code_oneHot = newOneHotLabels(cb)
        #oneHotLabels= np.full((accM.shape[0],1),code_oneHot,dtype=int)
        #labels=np.zeros((accM.shape[0],n_devs))
        #labels[:]=cb
        N_labels_len = np.full((accM.shape[0],1),len_labels,dtype=int)
        X.append(accM)
        #Y.append(oneHotLabels)
        Y.append(N_labels_len)
    X = np.vstack(X)
    Y = np.vstack(Y)

    return X,Y

def buildSelectDset(dir_house,database_name,prefix,num_devs):
    l_devs,names=getSelectedSignals(dir_house,prefix,num_devs)
    sl_devs=[]

    n_devs = len(l_devs)


    mixes_dset=[]
    mixes_labels=[]
    X = np.empty((1,100))
    Y = np.empty((1,1),dtype=int)


    for i in range(n_devs):

        ms=getDevSliding(l_devs[i],100,10)
        #fms=filter_device(ms)
        sl_devs.append(ms)

        '''
        am=aumentDim(ms,500)
        plot_Mdev(am,names[i],1)
        plt.show()
        sl_devs.append(am)
        '''
    comb_one=combMatrix(1,n_devs)
    sizes=20000
    #gera dos dados para combinacoes com apenas um aparelho
    for i in range(comb_one.shape[0]):
        cb = comb_one[i]#combinacao do tipo [0,0,0,0,0]em que cada posicao indica se determinado aparelho esta ligado
        one_hot = newOneHotLabels(cb)# converte a representação anterior em um numero inteiro

        idev=np.where(cb>0)[0]
        M_one = sl_devs[idev[0]]
        M_one = aumentDim(M_one,sizes)

        label_n_size=1-1#label do tipo 0,...., N-1 no qual N é o numero de aparelhos na combinação
        #labels = np.zeros((M_one.shape[0],n_devs))
        #labels[:]=cb
        #oneHotLabels = np.full((M_one.shape[0],1),one_hot,dtype=int)
        N_size_labels = np.full((M_one.shape[0],1),label_n_size,dtype=int)
        X = np.append(X,M_one,0)
        Y = np.append(Y,N_size_labels,0)

    #gera os dados para combinações com mais de 1 aparelho

    for i in range(2,n_devs+1):#i representa o numero de dispositivos na mistura
        combM=combMatrix(i,n_devs)
        print(combM)
        print("Combinação #:",i)
        M_sigs, M_labels = getNMixSigs(sl_devs,combM,sizes)
        print('Tamanho')
        print(M_sigs.shape,M_labels.shape)
        X=np.append(X,M_sigs,0)
        Y=np.append(Y,M_labels,0)

    X = X[1:]
    Y = Y[1:]

    dataset = np.append(X,Y,1)
    np.random.shuffle(dataset)
    X = dataset[:,:-1]
    Y = dataset[:,-1:]
    max=np.amax(Y)
    print('maior valor',max)
#    print(Y[:20])

    print("Tamanho dataset: ",X.shape,Y.shape)
    saveUnifiedDset(database_name,X,Y,num_devs)

def builFeatureDset(raw_dset_name,dir_house,feat_dset_name):
    X,Y,dev_names,cod_M = readUnifiedDset(raw_dset_name,dir_house)
    #X=X[:500]
    #Y=X[:500]
    cod_M=cod_M[0]

    Xfeatures = getDset_features(X)
    saveUnifiedDset(feat_dset_name,Xfeatures,Y,cod_M)


########################3
##___ Funcoes auxiliares , que convertem representacoes dos rotulos utilitarios
#######################
#funcao que realiza a extracao de caracteristicas utilizando o tsfresh
def getFeaturesFromDset(M_signals):
    col = ['id','time','value']
    dataFrame = get_DFrame(M_signals,col)
    fc_param = {
    "median":None,
    "kurtosis":None,
    "skewness":None,
    "cid_ce":[{"normalize":False}],
    "number_cwt_peaks":[{"n":2}],
    "number_peaks":[{"n":2}],
    "linear_trend":[{"attr":'slope'},{"attr":'intercept'}],
    "fft_coefficient":[{"coeff":7,"attr":"abs"},{"coeff":7,"attr":"angle"}]
    }
    print("Extracting features")
    features= extract_features(dataFrame, column_id="id", column_sort="time",column_kind=None, column_value=None,default_fc_parameters=fc_param)
    keys = list(features)
    print(keys)
    dsetFeatures = np.zeros((M_signals.shape[0],len(keys)))
    dsetFeatures[:]=features[keys]
    return dsetFeatures

#TRansformações nas representações dos rotulos das classe, target(o valor da classe),canonical(representa 1 em cada posicao dos aparelhos indicando quais estao ligados)
def OneHot2Canonical(onehot_labels,code_Matrix):
    indexes = np.where(onehot_labels>0)[1]
    canonical_labels = code_Matrix[indexes]
    return canonical_labels

def OneHot2Target(M_labels):
    targets=np.zeros(M_labels.shape[0],dtype = np.int64)
    indexes = np.where(M_labels>0)[1]
    targets[:]=indexes[:]
    return targets+1

def Target2OneHot(targets):
    n_classes = int(np.amax(targets))
    targets -=1
    oneHotlabels = np.zeros((targets.shape[0],n_classes),dtype =np.float64)
    for i in range(targets.shape[0]):
        oneHotlabels[i,int(targets[i])] = 1
    return oneHotlabels

def get_DFrame(M_mix,l_columns):
    nd = M_mix.shape[0]*M_mix.shape[1]
    id = np.arange(M_mix.shape[0])
    tmp = np.ones((M_mix.shape[1],M_mix.shape[0]))
    IDs = (id*tmp).T
    IDs = np.reshape(IDs,(nd,1))

    t = np.arange(M_mix.shape[1])
    Times=np.empty(M_mix.shape,dtype = np.float64)
    Times[:] = t
    Times = np.reshape(Times,(nd,1))

    values = np.reshape(M_mix,(nd,1))
    df = np.append(IDs,Times,1)
    df = np.append(df,values,1)
    dataf = pd.DataFrame(data=df,columns = l_columns)
    return dataf

def showDevicesOn(M_result,dev_names):
    N=M_result.shape[0]
    l,c = np.where(M_result>0.5)
    dev_on = [[] for i in range(N)]
    for i in range(l.shape[0]):
        dev_on[l[i]].append(dev_names[c[i]])
    for k in range(len(dev_on)):
        s = 'Hour %d: '%(k)
        if len(dev_on[k])>0:
            s+=','.join(dev_on[k])+ ':are ON'
        else:
            s += 'No devices ON'
        print(s)

def showOneHotDevOn(onehot_labels,cM,dev_names):
    canonical_labels = OneHot2Canonical(onehot_labels,cM)
    showDevicesOn(canonical_labels,dev_names)


def filter_device(M):
    t = np.arange(0,M.shape[1])
    fM = np.zeros((1,M.shape[1]),dtype = np.float64)

    ts=[]
    fs=[]
    for i in range(M.shape[0]):
        s= M[i]
        fs,ts=findsignalStates(s,t,20,50)
        fs = fs.reshape(1,fs.shape[0])
        fM = np.append(fM,fs,0)
        fs=[]
        ts=[]
    return fM[1:]


def show_folds_results(house_dir,devs_num,ver):
    dev_names = getDevNames(house_dir,devs_num)
    for i in range(len(devs_num)):

        path = '/info_folds_'+ver+dev_names[i]
        ms = read_DevSignals(path)

        acc=ms[0]

        print('Accuracy Mean: ',np.mean(acc))
        plt.figure(1)
        plt.title('Model: '+ver+dev_names[i])
        plt.xlabel('fold_number')
        plt.ylabel('accuracy')
        plt.plot(acc)
        plt.show()



#builFeatureDset('/Dset1.hdf5','/House_1','/Fdset1_v2.hdf5')

#buildSelectDset('/House_1',"/Dset1_vLenLabels.hdf5", 'vn_',[0,1,7,9])
#generateNmixDataset('/House_1','/House1_fDset.hdf5',devs_numbers=[0,1,4,7,9])#bathroom_gfi,dishwasher,lighting,refrigerator,washer_dryer
#generateNmixDataset('/House_2','/House2Dset.hdf5',devs_numbers=[2,3,4,5])#kitchen_outlets,lighting,microwave,refrigerator
#generateNmixDataset('/House_3','/House3Dset.hdf5',devs_numbers=[0,2,3,6,7,9,11])#bathroom_gfi,dishwasher,electronics,lighting,microwave,refrigerator,washer_dryer
#generateNmixDataset('/House_4','/House4Dset.hdf5',devs_numbers=[3,4,5,9,10])#furance,kitchen_outlets,lighting,stove,washer_dryer
#generateNmixDataset('/House_5','/House5Dset.hdf5',devs_numbers=[3,5,6,7,11])#electric_heat,furance,kitchen_outlets,lighting,refrigerator
#generateNmixDataset('/House_6','/House6Dset.hdf5',devs_numbers=[0,3,6,7,8])#air_conditioning,electric_heat,outlets_unknown,lighting,refrigerator
#P,labels=getHouseSignals(sys.argv[1])

#show_folds_results('/House_1',[0,1,4,7,9],sys.argv[1])
#buildSelectDset('/House_1','/Dset1_v3.hdf5','vn_',[0,1,7,9])
#builFeatureDset('/5Dev_Dset.hdf5','/House_1','/5Ddev_Fdset1.hdf5')

def convert_excel(dir_house,prefix,num_devs):
    l_devs,names=getSelectedSignals(dir_house,prefix,num_devs)
    dir='/home/julio/Framework_Projeto/Data_sets/ANN_data'+dir_house
    #names = names[0]
    for i in range(1):
        dev = l_devs[i]
        dev = np.reshape(dev,(dev.shape[0]*dev.shape[1]))
        idx = np.arange(dev.shape[0])/3600


        df = pd.DataFrame(data=dev,index=idx,columns=['Power'])
        #print(series)
        writer = pd.ExcelWriter(dir+'/'+names[i]+'.xlsx', engine='xlsxwriter')
        df.to_excel(writer, sheet_name='Sheet1')
        writer.save()

        '''
        print(dev.shape)
        print(names[i])
        plot_Mdev(dev,names[i],i)
        #plt.plot(dev)
        plt.show()
        '''
def convert_results_excel(data_set_file,dir_house,model_ver,i_dev):
    X,Y,dev_names,cm = readUnifiedDset(data_set_file,dir_house)
    dev_names = dev_names[0]
    #i_dev = int(input("Escolha um indice para visualizar os resultados:"))
    model_name=dev_names[i_dev]
    dir='/home/julio/Framework_Projeto/Data_sets/ANN_data'+dir_house
    file='/info_folds_NB_'+model_ver+model_name
    info=read_DevSignals(file)
    print(info.shape)
    #idx = np.arange(info.shape[1])
    df = pd.DataFrame({'Accuracy':info[0]})

    writer = pd.ExcelWriter(dir+file+'.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()


#for i in range(5):
#    convert_results_excel('/Dset1.hdf5','/House_1','nb_fv1_model_',i)
#convert_excel('/House_1','vn_',[0,1,4,7,9])
