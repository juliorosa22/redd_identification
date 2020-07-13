import REDD_management as rm
from basicLib import *
from apiFilter import *
from scipy.interpolate import interp1d
dir_redd = '/home/julio/Framework_Projeto/Data_sets/REDD'

dir_low='/low_freq'
dir_high='/high_freq_comp'

## Este arquivo contem funcoes que manipulam ambos corrente e potencia , no entando utiliza algumas funcoes do arquivo REDD_management.py que possui somente manipulcao dos sinais de potencia


##Funcoes que convertem os arquivos .dat em .h5 para melhor manipulacao
def csv2hf5(csv_path,file_path):
    time,power=rm.getLowFreqData(csv_path)

    f = h5py.File(dir_redd+file_path+'.hdf5','w')
    f_power =f.create_dataset('power',power.shape,dtype=power.dtype)
    f_time=f.create_dataset('time',time.shape,dtype=time.dtype)


    f_power[:]=power[:]
    f_time[:]=time[:]
    print(f_time[:5],f_power[:5])
    f.close()

def csv2hf5_highfreq(csv_file,h5_file):
    f = h5py.File(dir_redd+h5_file+'.hdf5','w')
    inf,power=rm.getHighFreqData(csv_file)
    f_power =f.create_dataset('power',power.shape,dtype=power.dtype)
    f_info=f.create_dataset('info',inf.shape,dtype=inf.dtype)
    f_power[:]=power[:]
    f_info[:]=inf[:]
    f.close()


def saveSig(file_path,time_r,power_r):
    time=time_r
    power=power_r
    f = h5py.File(dir_redd+'/'+file_path+'.hdf5','w')
    f_power =f.create_dataset('power',power.shape,dtype=power.dtype)
    f_time=f.create_dataset('time',time.shape,dtype=time.dtype)


    f_power[:]=power[:]
    f_time[:]=time[:]

    f.close()

def readSig(file):
    f = h5py.File(dir_redd+file+'.hdf5','r')
    keys = list(f)

    power =f[keys[0]][:]
    time =f[keys[1]][:]
    time=time.astype(int)
    f.close()
    return (time,power)

def read_highFreq(h5_file):
    f = h5py.File(dir_redd+h5_file+'.hdf5', 'r')
    power=f['power'][:]
    info=f['info'][:]
    return info,power


#### __ Funcoes que manipulam os timestamps de cada aparelho
def saveOneDevLabel(file_name,time,labels):

    path=dir_redd+file_name
    print(path)
    f = h5py.File(path+'.hdf5','w')
    f_labels =f.create_dataset('labels',labels.shape,dtype=labels.dtype)
    f_time=f.create_dataset('time',time.shape,dtype=time.dtype)



    f_labels[:]=labels[:]
    f_time[:]=time[:]

    f.close()

def readOneDevLabel(file_name):
    f = h5py.File(dir_redd+file_name+'.hdf5','r')
    keys = list(f)
    labels  = f[keys[0]][:]
    time=f[keys[1]][:]

    f.close()
    return time,labels


def getDevicesNames(dir_path,prefix,devs_index=None):
    f  = open(dir_redd+dir_path+'labels_names.txt')
    devices_names = list(f)
    f.close()
    devices_names = [ dev[:-1]+prefix for dev in devices_names ]
    devices_names = np.array(devices_names)
    if len(devs_index)>0:
        devices_names = devices_names[devs_index]
    return devices_names

def getHouseSignals(house_path,prefix,devs_index):
    devices_names = getDevicesNames(house_path,prefix,devs_index)
    signals = [readSig(house_path+dev_name) for dev_name in devices_names]
    return signals,devices_names



def is_sorted(a):
    for i in range(a.shape[0]-1):
         if a[i+1] < a[i] :
               return False
    return True

def getSortedStructArray(times,sig):
    dtype_str =[('tstamp',int),('signal',float)]
    lstruct=[(times[i],sig[i]) for i in range(times.shape[0])]
    vstruct=np.array(lstruct,dtype=dtype_str)
    vsorted = np.sort(vstruct,order='tstamp')
    return vsorted


#____funcoes que processam os arquivos .hdf5 de cada aparelhos antes q o rotulamento aconteça

def generate_sorted_sigs(dir_path,devs_index,prefix):
    l_sigs,names=getHouseSignals(dir_path,'',devs_index)
    mains=l_sigs[:2]
    devs =l_sigs[2:]

    t_mains = mains[0][0]
    t_devs = devs[0][0]

    ts_main = t_mains[3:-13]

     # ts_main sao os timestamps que os mains tem em comum com os timestamps dos aparelhos
    vs = np.zeros(ts_main.shape,dtype= np.float64)

    for i in range(len(devs)):
        tstamp=devs[i][0]
        sig = devs[i][1]
        print(names[i+2])
        str_array=getSortedStructArray(tstamp,sig)
        t = str_array[:]['tstamp']
        s = str_array[:]['signal']

        file_name = names[i+2]+'_sorted'
        saveSig(dir_path+file_name,t,s)

#identifica em quais timestamps em comum com seu main um determinado aparelho esta ligado, e cria um arquivo com 0 e 1s indicando quando esta desligado e ligado
def idf_one_dev(dir_path,file_label,ths,devs_index):
    l_sigs,names=getHouseSignals(dir_path,'_sorted',devs_index)
    main=l_sigs[0]
    t_main = main[0]

    devs =l_sigs[1:]
    t_devs = devs[0][0]


    sig_dev = devs[0][1]

    bi=np.where(t_main==t_devs[0])[0][0]
    ei=np.where(t_main==t_devs[-1])[0][0]
    nt_main=t_main[bi:ei+1]
    nt_sig_main = main[1][bi:ei+1]
    nt_devs=np.zeros(nt_main.shape,dtype=np.int32)
    nt_devs[:]=nt_main[:]

    new_sig=np.zeros(nt_main.shape,dtype=np.float64)
    est_sig=np.zeros(nt_main.shape,dtype=np.float64)

    main_mask = np.isin(nt_main,t_devs)
    it_not=np.where(main_mask==False)[0]
    it_yes=np.where(main_mask==True)[0]

    dev_mask = np.isin(t_devs,nt_main)
    it_yes_sig=np.where(dev_mask==True)[0]
    n_devs=len(devs)

    Mlabels=np.zeros((n_devs,est_sig.shape[0]),dtype=np.int32)
    t=np.arange(est_sig.shape[0])
    states=np.zeros(nt_main.shape,dtype=np.float32)
    print(t.shape,est_sig.shape)
    #ths=[100,100,100,100,100,50,50,50]
    l_ron=[]
    labels=np.zeros(nt_main.shape,dtype=int)
    for i in range(n_devs):
        print(i)
        print(names[i+1])
        sig = devs[i][1]
        y=sig[it_yes_sig]
        fdev=interp1d(it_yes,y)
        ynew=fdev(it_not)


        new_sig[it_yes]=y[:]
        new_sig[it_not]=ynew[:]
        est_sig[:]=est_sig+new_sig

        l_ron.append(detect_onset(est_sig,threshold=ths))

        bi=l_ron[i][:,0]
        ei=l_ron[i][:,1]
        for k in range(bi.shape[0]):
            states[bi[k]:ei[k]]=new_sig[bi[k]:ei[k]]

        inz=np.where(states>10)[0]
        print(inz.shape)
        labels[inz]=1
        plt.figure(1)
        plt.plot(est_sig)
        plt.figure(2)
        plt.plot(states,'g-')
        lnz=np.where(labels>0)[0]
        print(lnz.shape)
        print('Erro: ',((est_sig - states)**2).mean())
        plt.show()

    saveOneDevLabel(dir_path+names[1]+file_label,nt_devs,labels)



def verify_times(dir_path,devs_index,prefix):
    l_sigs,names=getHouseSignals(dir_path,'_sorted',devs_index)
    mains=l_sigs[:1]
    devs =l_sigs[1:]
    print(names)
    t_mains = mains[0][0]

    print("Devs:")
    t_devs = devs[0][0]
    i_dev = np.isin(t_devs,t_mains)
    nt_devs=t_devs[i_dev]
    acc=np.zeros(nt_devs.shape[0])
    plt.figure(1)
    print(len(devs))
    for i in range(len(devs)):
        #sig_dev = devs[i][1][i_dev]
        #acc=acc+sig_dev
        sig_dev = devs[i][1]
        plt.plot(sig_dev)
        print(sig_dev.mean())
        print(names[i+1])
        plt.show()
    plt.legend(devs_index[1:])
    print(names[1:])


#verify_times('/low_freq/house_3/',[1,2,4,6,8,10,12,16,18],'')
### ___ Funcoes que processam os dados ajustados dos aparelhos juntamente com os dados de seu respectivo main

#funcao responsavel por detectar quais aparelhos estao conectados em um respectivo main, testa todas as combinaçoes
# dos sinais de aparelhos separados e compara com a leitura agregada de potencia da main, escolhe a combinaçao de aparelhos
# que possui o menor erro quadratico medio

def identifyMainDevices(dir_path,devs_index,main_number):
    l_sigs,names=getHouseSignals(dir_path,'_sorted',devs_index)
    mains=l_sigs[:2]
    t_mains = mains[1][0]
    devs =l_sigs[2:]
    t_devs = devs[0][0]


    n_devs = len(devs)
    combM = np.empty((1,n_devs))
    for i in range(7,n_devs+1):#i representa o numero de dispositivos na mistura
        combM=np.append(combM,combMatrix(i,n_devs),0)

    combM = combM[1:]
    combM=combM.astype(int)
    print('Numero de Possibilidades',combM.shape[0])

    mask_main = np.isin(t_mains,t_devs)
    it_m=np.where(mask_main==True)[0]
    ts_main = t_mains[it_m]

    mask_dev =  np.isin(t_devs,t_mains)
    it_d=np.where(mask_dev==True)[0]
    print("Quantidade de timestamps devs in main")
    print(ts_main.shape)


    sig_main = mains[main_number][1][it_m]

    acc_sig = np.zeros(sig_main.shape[0],dtype=np.float64)
    best_comb = np.zeros(n_devs)
    best_sig = np.zeros(sig_main.shape[0],dtype=np.float64)
    low_coef = 1e7

    for comb in combM:
        icb=np.where(comb>0)[0]
        for i in range(icb.shape[0]):
            acc_sig=acc_sig+devs[icb[i]][1][it_d]
        coef = ((sig_main - acc_sig)**2).mean()
        if low_coef > coef:
            low_coef = coef
            best_sig[:] = acc_sig[:]
            best_comb[:]=comb[:]
        acc_sig[:]=0

    print(low_coef)
    print(devs_index[2:])
    print(best_comb)
    plt.plot(sig_main)
    plt.plot(best_sig,'r-')
    plt.show()

    '''
    print(sig_main.shape)
    print(sig_dev.shape)
    plt.plot(sig_main)
    plt.plot(sig_dev,'g-')
    plt.show()
    '''



def identifyMainDevices2(dir_path,devs_main,devs_find):
    l_sigs,names=getHouseSignals(dir_path,'_sorted',devs_main)
    unknow_devs,names2=getHouseSignals(dir_path,'_sorted',devs_find)

    mains=l_sigs[0]
    t_mains = mains[0]

    devs =l_sigs[1:]
    t_devs = devs[0][0]


    n_devs = len(unknow_devs)
    combM = np.empty((1,n_devs))
    for i in range(1,n_devs+1):#i representa o numero de dispositivos na mistura
        combM=np.append(combM,combMatrix(i,n_devs),0)

    combM = combM[1:]
    combM=combM.astype(int)
    print('Numero de Possibilidades',combM.shape[0])


    mask_main = np.isin(t_mains,t_devs)
    it_m=np.where(mask_main==True)[0]
    ts_main = t_mains[it_m]

    mask_dev =  np.isin(t_devs,t_mains)
    it_d=np.where(mask_dev==True)[0]
    know_sigs=np.zeros(it_d.shape[0],dtype=np.float64)

    for i in range(len(devs)):
        s=devs[i][1][it_d]
        know_sigs=know_sigs+s

    sig_main = mains[1][it_m]

    acc_sig = np.zeros(sig_main.shape[0],dtype=np.float64)
    best_comb = np.zeros(n_devs)
    best_sig = np.zeros(sig_main.shape[0],dtype=np.float64)
    low_coef = 1e7


    coef = ((sig_main - know_sigs)**2).mean()
    if low_coef > coef:
        low_coef = coef
        best_sig[:] = acc_sig[:]
        best_comb[:]=np.zeros(n_devs,dtype=np.int32)
    plt.figure(1)
    plt.plot(know_sigs,'r-')
    plt.plot(sig_main,'g--')




    for comb in combM:
        icb=np.where(comb>0)[0]

        for i in range(icb.shape[0]):
            acc_sig=acc_sig+unknow_devs[icb[i]][1][it_d]
        acc_sig=acc_sig+know_sigs
        coef = ((sig_main - acc_sig)**2).mean()
        if low_coef > coef:
            low_coef = coef
            best_sig[:] = acc_sig[:]
            best_comb[:]=comb[:]
        acc_sig[:]=0


    print(low_coef)

    print(devs_find)
    print(best_comb)
    plt.figure(2)
    plt.title("Identificação dos aparelhos conectados")
    plt.xlabel('time')
    plt.ylabel('Power(W)')
    plt.plot(best_sig,'r--',label='Combinação de sinais estimada')
    plt.plot(sig_main,'b-',label='Sinal real Main 2')
    plt.legend()
    plt.show()

    '''
    print(sig_main.shape)
    print(sig_dev.shape)
    plt.plot(sig_main)
    plt.plot(sig_dev,'g-')
    plt.show()
    '''

## Funcao que gera um arquivo contendo uma matriz em q a linha é um determinado aparelho e a coluna sao os timestamps , cada valor v[i,j]=0 ou =1
#indicando que o aparelho i esta ligado ou nao no timestamp j

def save_MainData(file_name,data_main,ths_count_labels):
    #ts_low,labels,devs=readMainLabels(low_freq_labels_file)
    ts_low,labels,devs=data_main

    vs=labels.sum(0)


    inz=np.where(vs>0)[0]
    new_ts_low = ts_low[inz]
    Mlabels = labels[:,inz]
    Mlabels=Mlabels.T
    vhot_labels=rm.getLabelsOneHot(Mlabels)
    print(vhot_labels.shape)
    #retira os rotulos q possuem poucos exemplos
    unique_labels,counts=np.unique(vhot_labels,return_counts=True)
    print("Infos:")
    print(unique_labels.shape)
    print(unique_labels)
    print(counts)
    print(vhot_labels.shape)
    print("Removendo os exemplos com poucos rotulos")
    filter_mask=counts<ths_count_labels
    #print(filter_mask)
    exclude_labels=unique_labels[filter_mask]
    counts=counts[~filter_mask]
    #print(counts)
    unique_labels=unique_labels[~filter_mask]
    mask_hotlabels=np.full(vhot_labels.shape[0],False)

    for k in range(exclude_labels.shape[0]):
        mask_hotlabels=mask_hotlabels | (vhot_labels==exclude_labels[k])
    vhot_labels=vhot_labels[~mask_hotlabels]
    new_ts_low=new_ts_low[~mask_hotlabels]
    print(vhot_labels.shape)
    print(counts)
    print(counts.shape)
    print(unique_labels)




    coded_labels=np.zeros(vhot_labels.shape,dtype=np.int32)

    for i in range(unique_labels.shape[0]):
        n=np.where(vhot_labels==unique_labels[i])[0]
        coded_labels[n]=i



    f = h5py.File(dir_redd+file_name+'.hdf5','w')
    f_times = f.create_dataset('times',new_ts_low.shape,dtype=new_ts_low.dtype)
    f_labels_onehot =f.create_dataset('labels_onehot',unique_labels.shape,dtype=unique_labels.dtype)
    f_labels_coded=f.create_dataset('labels_coded',coded_labels.shape,dtype=coded_labels.dtype)
    f_devs=f.create_dataset('devs',devs.shape,dtype=devs.dtype)

    f_times[:]=new_ts_low[:]#os timestamps no qual existe pelo menos 1 aparelho ligado
    f_labels_onehot[:]=unique_labels[:]#usado na conversao de volta para onehotlabels
    f_labels_coded[:]=coded_labels[:]
    f_devs[:]=devs[:]#vetor com os indices dos aparelho no arquivo labels.txt

    f.close()

def read_MainData(file):
    f = h5py.File(dir_redd+file+'.hdf5', 'r')
    keys=list(f)
    times=f[keys[3]][:]
    coded_labels = f[keys[1]][:]
    onehot_labels = f[keys[2]][:]
    devs=f[keys[0]][:]
    return (times,coded_labels,onehot_labels,devs)


def buildMainLabelsRaw(dir,devs,file_name):
    names = getDevicesNames(dir,'_sorted_label',devs)
    labels=[]
    t=[]
    for i in range(1,len(names)):
        print(names[i])
        t,label=readOneDevLabel(dir+names[i])
        #print(label.shape)
        l=np.where(label>0)[0]
        print(l.shape)
        labels.append(label)
    Mlabels= np.vstack(labels)
    devs=np.array(devs,dtype=np.int32)
    #print(t.shape)
    #print(Mlabels.shape)
    data_main=(t,Mlabels,devs)
    save_MainData(file_name,data_main,0)



#Responsavel por selecionar as formas de onda de um sinal de alta frequencia para determinada combinaçao de aparelhos ligados
def get_signal_form(high_sig,ts_high,list_sig,list_label,label,n,next_low_ts):
    if n<ts_high.shape[0]:
        if (next_low_ts - ts_high[n]) > 0:
            list_sig.append(high_sig[n])
            list_label.append(label)
            get_signal_form(high_sig,ts_high,list_sig,list_label,label,n+1,next_low_ts)
        else:
            return
    else:
        return

def saveDsetHighFreq (file_name,inputs,labels):
    f = h5py.File(dir_redd+file_name+'.hdf5', 'w')
    dset_inputs=f.create_dataset('inputs',inputs.shape,dtype=inputs.dtype)
    dset_labels=f.create_dataset('labels',labels.shape,dtype=labels.dtype)
    dset_inputs[:]=inputs[:]
    dset_labels[:]=labels[:]
    f.close()

def creat_fft_dset(new_dset,dir_house,dset_name):
    infos_dset=read_MainData(dir_house+'main2_data_v2_labels')
    devs=infos_dset[3]
    mask_label=infos_dset[2]
    names = getDevicesNames(dir_house,'',devs)

    devs_names=names[1:]
    print(devs_names)
    print(mask_label)

    X,Y =readDsetHighFreq(dset_name)

    Y=Y.astype(int)
    #ir=np.random.randint(X.shape[0],size=(n))
    #ir=np.array([100,1500,2000,3000,5000,10000])
    N=X.shape[1]
    l_fft=[]

    for i in range(X.shape[0]):
        print(i)
        l_fft.append(get_fft_values(X[i],N))

    #l_fft=getDset_features(X)
    l_fft=np.vstack(l_fft)
    print(l_fft.shape)
    saveDsetHighFreq(new_dset,l_fft,Y)

def readDsetHighFreq(file_name):

    f = h5py.File(dir_redd+file_name, 'r')
    keys=list(f)
    sigs=f[keys[0]][:]
    labels=f[keys[1]][:]
    return sigs,labels

def labelForHighFreq(high_freq_signal_file,data_main_file,dset_name):

    data=read_MainData(data_main_file)

    ts_low=data[0]
    vlabels = data[1]
    print(ts_low.shape,vlabels.shape)

    inf,sig = read_highFreq(high_freq_signal_file)
    ts_high = inf[:,0]
    v_counts=inf[:,1]

    print(sig.shape)
    print(ts_high.shape)
    l_sig=[]
    l_label=[]


    for i in range(ts_low.shape[0]):
        print(i)
        label=vlabels[i]
        tpi=ts_low[i]
        n=np.where(ts_high<=tpi)[0][-1]
        l=[]

        #print(n)

        l_sig.append(sig[n])
        l_label.append(label)

        next_time=tpi+1
        #if i<(ts_low.shape[0]-1):
        #    next_time=ts_low[i+1]
        get_signal_form(sig,ts_high,l_sig,l_label,label,n+1,next_time)



    print(ts_low.shape)
    dset_sig=np.vstack(l_sig)
    dset_label=np.array(l_label)
    Y = np.reshape(dset_label,(dset_label.shape[0],1))
    dset=np.append(dset_sig,Y,1)
    np.random.shuffle(dset)
    X = dset[:,:-1]
    Y = dset[:,-1:]
    print(X.shape)
    Y=Y.astype(int)
    print(Y.shape)
    c,u=np.unique(Y,return_counts=True)
    print(c)
    print(u)
    print(data[2])
    saveDsetHighFreq(dset_name,X,Y)



def show_classes_in_dset(dir_house,dset_name):
    infos_dset=read_MainData(dir_house+'proc_main2_labels')
    devs=infos_dset[3]
    mask_label=infos_dset[2]
    names = getDevicesNames(dir_house,'',devs)
    devs_names=names[1:]
    print(devs_names)
    print(mask_label)

    X,Y =readDsetHighFreq(dset_name)
    Y=Y.astype(int)
    #ir=np.random.randint(X.shape[0],size=(n))
    #ir=np.array([100,1500,2000,3000,5000,10000])


    print('Shape dataset')
    print(X.shape,Y.shape)

    for i in range(mask_label.shape[0]):
        iv=np.where(Y==i)[0]
        n=3 if iv.shape[0]>3 else iv.shape[0]
        iv=iv[:n]
        for index in iv:
            sig=X[index]
            label=Y[index]
            one_hot=mask_label[label]
            raw_label=rm.decodeOneHot(one_hot,len(devs_names))
            tindex=raw_label>0
            print(devs_names[tindex])
            plt.plot(sig)
            plt.show()

## funcoes para banco de dados com potencia

def normSig(lsig,sz):
    sig=np.array(lsig,dtype=np.float64)
    n = sig.shape[0]

    if n<sz:
        nzeros = sz - n
        rsz = math.ceil(nzeros/2)
        lsz = math.floor(nzeros/2)
        sig = np.append(np.zeros(lsz),np.append(sig,np.zeros(rsz)))
    return sig


def powerDset(data_main_file,dir_path,dset_name):

    data=read_MainData(data_main_file)
    main_data,name=getHouseSignals(dir_path,'_sorted',[0])
    t_main=main_data[0][0]
    main_sig=main_data[0][1]
    plt.figure(1)
    plt.plot(main_sig)
    '''
    ts_low=data[0]
    vlabels = data[1]
    devs=data[3]
    names=getDevicesNames(dir_path,'',devs)
    print(names)

    mask=np.isin(t_main,ts_low)
    mask_index=np.where(mask==True)

    selected_main_sig=main_sig[mask_index]


    s_label=vlabels[0]
    frame_sig=[selected_main_sig[0]]

    l_frame=[]
    l_labels=[]

    for i in range(1,vlabels.shape[0]):
        print(i)
        if s_label != vlabels[i] or len(frame_sig)==60:
            l_frame.append(normSig(frame_sig,60))
            l_labels.append(s_label)
            s_label=vlabels[i]
            del frame_sig[:]
        frame_sig.append(selected_main_sig[i])

    X = np.vstack(l_frame)
    Y = np.array(l_labels,dtype=np.int32)
    Y = np.reshape(Y,(Y.shape[0],1))

    dset=np.append(X,Y,1)
    np.random.shuffle(dset)
    X = dset[:,:-1]
    Y = dset[:,-1:]
    Y = Y.astype(int)
    print(X.shape,Y.shape)
    saveDsetHighFreq(dset_name,X,Y)
    '''

def occurrenceDevInlabel(n_devs,mask_dev,y_pred,y_labels):

    y_coded=np.argmax(y_labels,1)
    ypred_coded=np.argmax(y_pred,1)#vetor q contem um numero inteiro representando a combinacao
    print(y_coded.shape)
    print(ypred_coded.shape)


    y_onehot_labels=np.zeros(y_coded.shape[0])
    ypred_onehot_labels=np.zeros(y_coded.shape[0])

    y_onehot_labels[:]=mask_dev[y_coded]
    ypred_onehot_labels[:]=mask_dev[ypred_coded]


    y_onehot_labels=y_onehot_labels.astype(int)
    ypred_onehot_labels=ypred_onehot_labels.astype(int)

    #print(onehot_labels)
    yraw_labels=rm.convertLabels(y_onehot_labels,n_devs)
    ypred_raw_labels=rm.convertLabels(ypred_onehot_labels,n_devs)
    pred_count=np.zeros(n_devs)
    for i in range(yraw_labels.shape[0]):
        ip = ypred_raw_labels[i]>0
        iy = yraw_labels[i]>0
        pred_count[ip&iy]+=1
    count = yraw_labels.sum(0)
    count=count.astype(int)
    pred_count=pred_count.astype(int)
    return (count,pred_count)


## realiza a contagem dos falsos positivos,falsos negativos e etc
def probAnalisys(n_devs,mask_dev,y_pred,y_labels,TFPN,PN):
    y_coded=y_labels.flatten()#np.argmax(y_labels,1)#ground truth
    ypred_coded=np.argmax(y_pred,1)#vetor q contem numeros inteiro representando a combinacao
    print(y_coded.shape)
    print(ypred_coded.shape)


    y_onehot_labels=np.zeros(y_coded.shape[0])
    ypred_onehot_labels=np.zeros(y_coded.shape[0])

    y_onehot_labels[:]=mask_dev[y_coded]
    ypred_onehot_labels[:]=mask_dev[ypred_coded]


    y_onehot_labels=y_onehot_labels.astype(int)
    ypred_onehot_labels=ypred_onehot_labels.astype(int)

    #print(onehot_labels)
    yraw_labels=rm.convertLabels(y_onehot_labels,n_devs)
    ypred_raw_labels=rm.convertLabels(ypred_onehot_labels,n_devs)
    # Modifica a matriz TFPN 4xNdevs, onde l0=tp(true positive),l1=fp(false positive),l2=tn(true negative), l3=fn(false negative)
    lines_up=np.zeros(n_devs,dtype=np.int32)
    cols_up=np.arange(n_devs,dtype=np.int32)
    cols_up = cols_up.tolist()
    for i in range(yraw_labels.shape[0]):
        bz=yraw_labels[i]>0
        p_eq=yraw_labels[i]==ypred_raw_labels[i]
        for j in range(yraw_labels.shape[1]):
            if p_eq[j]:
                lines_up[j]= 0 if bz[j] else 2 # primeiro caso qndo ambos predicao e esperado sao iguais, 0 se for iguais positivos ,2 caso negativos
            else:
                lines_up[j]= 3 if bz[j] else 1 # qndo sao diferentes, 1 se for falso positivo e 3 falso negativo


            lup=lines_up.tolist()

        TFPN[lup,cols_up]+=1

    P = yraw_labels.sum(0)
    N = np.zeros(yraw_labels.shape[1],dtype=np.int32)
    N[:]=yraw_labels.shape[0]
    N = N - P
    lpn=np.vstack([P,N])
    PN+=lpn[:]


def occurrenceDevInlabel_ver2(n_devs,mask_dev,y_pred,y_labels):

    y_coded=y_labels
    ypred_coded=y_pred

    y_onehot_labels=np.zeros(y_coded.shape[0])
    ypred_onehot_labels=np.zeros(y_coded.shape[0])

    y_onehot_labels[:]=mask_dev[y_coded]
    ypred_onehot_labels[:]=mask_dev[ypred_coded]


    y_onehot_labels=y_onehot_labels.astype(int)
    ypred_onehot_labels=ypred_onehot_labels.astype(int)

    #print(onehot_labels)
    yraw_labels=rm.convertLabels(y_onehot_labels,n_devs)
    ypred_raw_labels=rm.convertLabels(ypred_onehot_labels,n_devs)
    pred_count=np.zeros(n_devs)
    for i in range(yraw_labels.shape[0]):
        ip = ypred_raw_labels[i]>0
        iy = yraw_labels[i]>0
        pred_count[ip&iy]+=1
    count = yraw_labels.sum(0)
    count=count.astype(int)
    pred_count=pred_count.astype(int)
    return (count,pred_count)

def showPowerSignals(dir_path,list):
    main_data,name=getHouseSignals(dir_path,'_sorted',list)
    t_main=main_data[0][0]
    main_sig=main_data[0][1]
    plt.figure(1)
    plt.ylabel('Power(W)')
    plt.xlabel('timestamp')
    plt.title("Main House_3")
    plt.plot(main_sig)
    dev_names = name[1:]
    new_names=[]
    plt.figure(2)
    plt.title("Aparelhos")
    for i in range(1,len(name)):
        sig = main_data[i][1]
        nm=name[i]
        nm=nm.split('_')
        new_names.append(nm[0])
        plt.plot(sig)
    new_name=['Aparelhos eletronicos','forno','secador de roupas','microondas','iluminacao','detector de fumaca','Circuito do banheiro','Itens de cozinha']
    plt.legend(new_name)
    plt.ylabel('Power(W)')
    plt.xlabel('timestamp')
    plt.show()

##____ FUncao que ordena os timestamps e as potencias
#generate_sorted_sigs('/low_freq/house_3/',[0,1,16,18],'_sorted')

#verify_times('/low_freq/house_3/',[0,1,3,11,14,17,20],'')


##___ FUncoes que fazem o preprocessamento dos dados antes de rotular a corrente
#identifyMainDevices('/low_freq/house_3/',[0,1,2,3,4,5,6,8,9,10,12,13,15,16,18,19,21],1)
#identifyMainDevices2('/low_freq/house_3/',[0,5,7,9,13,15,19,21],[3,11,14,17,20])
#identifyMainDevices2('/low_freq/house_3/',[1,2,4,6,8,10,12,16,18],[3,11,14,17,20])
#idf_one_dev('/low_freq/house_3/','_label',0,[1,4])

#buildMainLabelsRaw('/low_freq/house_3/',[0,3,5,7,9,11,13,14,15,17,19,20,21],'/data_main1_labels')

###__rotula a corrente e cria um banco com as transformadas de fourier
#labelForHighFreq('/high_freq_comp/house_3/current_2','/low_freq/house_3/data_main2_labels','/full_main2_dset')

#___ manipulacao da potencia
#powerDset('/low_freq/house_3/data_main2_labels','/low_freq/house_3/','/full_main2_power_dset')
#showPowerSignals('/low_freq/house_3/',[0,5,9,13,15,16,17,19,20])





'''
___ Informacoes referentes a cada aparelho conectado em cada main
## Aparelhos utilizados nos labels para o treinamento
#  main1: [0,3,5,7,9,11,13,14,15,17,19,20,21]
#main 2: [1,2,4,6,8,10,12,16,18]


#Numeraçao dos aparelhos ligados em cada painel principal
[0,1,2,4,5,6,8,9,10,12,13,15,16,18,19,21]
[2,4,5,6,8,9,10,12,13,15,16,18,19,21]
[outlets_unknown1,lighting1,electronics1,refrigerator,dishwasher1,furance1,lighting2,washer_dryer1,washer_dryer2,microwave1,lighting4,lighting5,bathroom_gfi1,kitchen_outlets2]
main 1 = [0. 0. 1. 0. 0. 1. 0. 0. 1. 1. 0. 0. 1. 1.]
main 2 = [1. 1. 0. 1. 1. 0. 1. 1. 0. 0. 1. 1. 0. 0.]
devices on in main 1
[0,3,5,7,9,11,13,14,15,17,19,20,21]
devices on in main 2
[1,2,4,6,8,10,12,16,18]
main1:[5,9,]
main2:[2,4,6,8,10,]

'''
#values for edge detection
# 3:1.002
# 5:100  
# 7:4.5
# 9:50
# 11:10
# 13:10
# 14:4
# 15:100
# 17:3
# 19:100
# 20:5
# 21:100

