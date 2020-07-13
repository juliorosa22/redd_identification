from basicLib import *
from scipy.interpolate import interp1d
import util
import matplotlib.pyplot as plt

lowfreq_dir=curr_work_dir+'/REDD/low_freq/low_freq/'
h5_lowfreq_dir = curr_work_dir+'/REDD_h5/low_freq/'

highfreq_dir=curr_work_dir+'/REDD/high_freq_comp/high_freq/'
h5_highfreq_dir = curr_work_dir+'/REDD_h5/high_freq/'
dset_dir=curr_work_dir+'/REDD_h5/datasets/'

##___ Funcoes que manipulam potencias do diretorio lowfreq___

# lê um arquivo channel_i.dat e retorna os vetores com timestamp e power
def lowGetData(channel_file):
    plist=[]
    tlist=[]
    with open(channel_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            t=int(row[0].split('.')[0])
            s=float(row[1])
            #print(s,t)
            tlist.append(t)
            plist.append(s)
    pv=np.array(plist,dtype=np.float32)
    tv=np.array(tlist,dtype=np.int32)
    return pv,tv

def lowSortStructArray(times,sig):
    dtype_str =[('timestamp',int),('power',float)]
    lstruct=[(times[i],sig[i]) for i in range(times.shape[0])]
    vstruct=np.array(lstruct,dtype=dtype_str)
    vsorted = np.sort(vstruct,order='timestamp')
    return vsorted

## Transforma os arquivos .dat localizados em /REDD/low_freq/lowfreq/house_i/channel_j.dat
## em channel_j.h5 e salva em /REDD_h5/low_freq/house_i
def lowChangeHouseFilesToH5Files(house_dir):
    new_dir=h5_lowfreq_dir+house_dir
    devs_name=lowGetDevicesNames(lowfreq_dir+house_dir,[])
    if not os.path.exists(new_dir):
        try:
            os.makedirs(new_dir,0o777)
        except OSError:
            print('Creation of: '+new_dir+' failed.')
            return  
    for i in range(devs_name.shape[0]):
        print('Changing file:'+devs_name[i])
        pv,tv = lowGetData(lowfreq_dir+house_dir+'channel_'+str(i+1)+'.dat')
        data_sorted = lowSortStructArray(tv,pv)
        data = [data_sorted[:]['timestamp'],data_sorted[:]['power']]
        labels = ['timestamp','power']
        saveH5File(new_dir+'channel_'+str(i+1)+'.h5',labels,data)


# Faz a leitura do arquivo labels.dat de uma pasta house_i e retorna uma lista de strings com os nomes dos
# aparelhos de uma casa
def lowGetDevicesNames(dir_path,devs_index=None):
    f  = open(dir_path+'labels.dat')
    devices_names = list(f)
    f.close()
    devices_names = [ dev[:-1] for dev in devices_names ]
    devices_names = np.array(devices_names)
    if len(devs_index)>0:
        devices_names = devices_names[devs_index]
    return devices_names


def lowGetHouseSignals(dir_path):
    channel_names=lowGetDevicesNames(lowfreq_dir+dir_path,[])
    #data_devs = [ readH5File(dir_path+'channel_'+str(i+1)+'.h5')[1] for i in devs_index]
    data_devs = [ readH5File(h5_lowfreq_dir+dir_path+'channel_'+str(i+1)+'.h5')[1] for i in range(len(channel_names))]
    return data_devs


def lowIdentifyMainDevices(main_data,devs):
    # if len(devs_index)<8:
    #     print('Erro: devs_index must have more than 7 indexes of devices.')
    #     return
    # l_sigs = lowGetHouseSignals(h5_lowfreq_dir+dir_path,devs_index)
    # names = lowGetDevicesNames(lowfreq_dir+dir_path,devs_index)
    # main_data=l_sigs[:2]
    # devs =l_sigs[2:]
    t_main_data = main_data[1]
    t_devs = devs[0][1]

    mask_main = np.isin(t_main_data,t_devs)
    ts_main = t_main_data[mask_main]
    mask_dev =  np.isin(t_devs,t_main_data)
    #print("Quantidade de timestamps devs in main")
    #print(ts_main.shape)
    n_devs = len(devs)
    combM = np.empty((1,n_devs))
    for i in range(6,n_devs+1):#i representa o numero de dispositivos na mistura
        combM=np.append(combM,util.combMatrix(i,n_devs),0)
    combM = combM[1:]
    combM=combM.astype(int)
    print('Numero de Possibilidades',combM.shape[0])

    sig_main = main_data[0][mask_main]
    acc_sig = np.zeros(sig_main.shape[0],dtype=np.float64)
    best_comb = np.zeros(n_devs)
    best_sig = np.zeros(sig_main.shape[0],dtype=np.float64)
    low_coef = 1e7
    
    for comb in combM:
        icb=np.where(comb>0)[0]
        for i in range(icb.shape[0]):
            acc_sig=acc_sig+devs[icb[i]][0][mask_dev]
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
    
    main 1: [0. 1. 0. 1. 0. 0. 1. 0. 0. 1. 1. 0. 0. 1. 1.]-> [3,5,9,13,15,19,21]
    main2: [1. 1. 1. 0. 1. 1. 0. 1. 1. 0. 0. 1. 1. 0. 0.]-> [2,4,6,8,10,12,16,18]
    range dos sindices de dispositivos: [2,21]
    canais impares:main1
    canais pares: main2
    '''



def lowIdentifyOnStatesDevice(dir_path,dev_index,save_cond=False):
    l_sigs = lowGetHouseSignals(dir_path)
    names = lowGetDevicesNames(lowfreq_dir+dir_path,[])
    
    main=l_sigs[0]
    dev =l_sigs[dev_index]

    t_main = main[1]
    t_dev = dev[1]
    
    bi=np.where(t_main==t_dev[0])[0][0]
    ei=np.where(t_main==t_dev[-1])[0][0]

    nt_main=t_main[bi:ei+1]
    
    nt_dev=np.zeros(nt_main.shape,dtype=np.int32)
    nt_dev[:]=nt_main[:]
    
    new_sig=np.zeros(nt_main.shape,dtype=np.float64)
    cp_sig=np.zeros(nt_main.shape,dtype=np.float64)

    main_mask = np.isin(nt_main,t_dev)
    it_not=np.where(main_mask==False)[0]
    it_yes=np.where(main_mask==True)[0]

    dev_mask = np.isin(t_dev,nt_main)
  
    states=np.zeros(nt_main.shape,dtype=np.float32)  
    labels=np.zeros(nt_main.shape,dtype=int)
    
    sig = dev[0]
    y=sig[dev_mask]
    fdev=interp1d(it_yes,y)
    ynew=fdev(it_not)

    new_sig[it_yes]=y[:]
    new_sig[it_not]=ynew[:]
    cp_sig [:]=new_sig[:]

    print(names[dev_index])
    # plt.figure(1)
    # plt.title(names[dev_index])
    # plt.plot(cp_sig)
    # plt.show()

    ths = float(input('Enter a threshold to detect On states in power signal: '))
    edges_index=util.detect_onset(new_sig,threshold=ths)
    
    bi=edges_index[:,0]
    ei=edges_index[:,1]
    for k in range(bi.shape[0]):
        states[bi[k]:ei[k]]=new_sig[bi[k]:ei[k]]

    inz= states>0
    labels[inz]=1
    
    # plt.figure(1)
    # plt.plot(cp_sig)
    plt.figure(2)
    plt.plot(states,'r-')
    
    print('Mean Square Error: ',((cp_sig - states)**2).mean())
    total=np.sum(inz)
    print('On states:')
    print(total)
    plt.show()
    new_dir = h5_lowfreq_dir+dir_path+'timestamp_labels/'
    if not os.path.exists(new_dir):
        try:
            os.makedirs(new_dir,0o777)
        except OSError:
            print('Creation of: '+new_dir+' failed.')
            return  
    if save_cond:        
        saveH5File(new_dir+'tslabel_channel_'+str(dev_index+1)+'.h5',['label','timestamp'],[labels,nt_dev])
    return labels,nt_dev


def lowGenerateSwitchStatesDevInMain(dir_path,devs_index):
    for i in devs_index:
        lowIdentifyOnStatesDevice(dir_path,i,True)        
    

def lowCodifyLabelsMain(main_data,ths_count):
    #Descricao: Funcao que é responsavel por codificar os rotulos das misturas
    #           com base no numero maximo de combinações encontradas
    #Parametros: 
    # -- main_data : é uma tupla (timestamp,labels), onde labels é uma matriz em q cada linha representa o estado
    #                 on (1) ou off(0) em um determinado timestamp de um determinado aparalho no respectivo main
    #-- ths_count : é um número inteiro que limita a quantidade de misturas que possui um determinado rotulo
    #               caso a ocorrencia de uma mistura com rotulo i aparecer menos que ths_count esta será removida

    raw_labels,ts_low=main_data
    
    vs=raw_labels.sum(0)

    inz=np.where(vs>0)[0]
    new_ts_low = ts_low[inz]
    Mraw_labels = raw_labels[:,inz]
    print('Non zero states')
    print(Mraw_labels.shape)
    Mraw_labels=Mraw_labels.T
    int_labels=util.labelGetIntLabels(Mraw_labels)
    
    #retira os rotulos q possuem poucos exemplos
    unique_labels,counts=np.unique(int_labels,return_counts=True)
    # print("Infos:")
    # print(unique_labels.shape)
    # print(unique_labels)
    # print(counts)
    # print(int_labels.shape)
    # print("Removendo os exemplos com poucos rotulos")
    filter_mask=counts<ths_count
    #print(filter_mask)
    exclude_labels=unique_labels[filter_mask]
    counts=counts[~filter_mask]
    #print(counts)
    unique_labels=unique_labels[~filter_mask]
    rm_int_label_mask=np.full(int_labels.shape[0],False)

    for k in range(exclude_labels.shape[0]):
        rm_int_label_mask=rm_int_label_mask | (int_labels==exclude_labels[k])
    int_labels=int_labels[~rm_int_label_mask]
    new_ts_low=new_ts_low[~rm_int_label_mask]
 
    labels_codrange=np.zeros(int_labels.shape,dtype=np.int32)

    for i in range(unique_labels.shape[0]):
        n=int_labels==unique_labels[i]
        #n=np.where(int_labels==unique_labels[i])[0]
        labels_codrange[n]=i

    return (['labels','times','int_labels_mask'],[labels_codrange,new_ts_low,unique_labels])


def lowLabelMainTimestamps(dir_path,main_index,devs_index,min_samples):
    
    names= lowGetDevicesNames(lowfreq_dir+dir_path,[])
    labels=[]
    labels_dir=h5_lowfreq_dir+dir_path+'timestamp_labels/'
    for i in devs_index:
        #dev = data[devs_index[i]]
        print('Device: '+names[i])
        data=readH5File(labels_dir+'tslabel_channel_'+str(i+1)+'.h5')[1]
        l=data[0]
        t = data[1]
        
        labels.append(l)
    
    labels = np.vstack(labels)
    main_data = (labels,t)
    names_dset,main_fill = lowCodifyLabelsMain(main_data,min_samples) 
    names_dset.append('devices_index')
    main_fill.append(np.array(devs_index,dtype=np.int32))
    print('Total labels:')
    print(main_fill[2].shape)
     # coded_labels = main_fill[0]
     # newts_low = main_fill[1]
     # mask_labels = main_fill[2]
    
    saveH5File(h5_lowfreq_dir+dir_path+'timestamp_labels/'+'main'+str(main_index+1)+'_labels.h5',names_dset,main_fill)


def highGetData(file):
    lv = np.empty((1,275),dtype=np.float64)
    linf=np.empty((1,2),dtype=np.int32)
    #c=0
    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            #print(c)
            inf=[float(row[0]),float(row[1]) ]
            inf = np.array([inf],dtype=np.float64)
            linf= np.append(linf,inf,0)
            lv=np.append(lv,np.array([row[2:]],dtype=np.float64),0)

            #c=c+1

    return linf[1:],lv[1:]


def highChangeCurrentToH5File(house_dir,file):
    new_dir=h5_highfreq_dir+house_dir
    if not os.path.exists(new_dir):
        try:
            os.makedirs(new_dir,0o777)
        except OSError:
            print('Creation of: '+new_dir+' failed.')
            return  
    inf,current=highGetData(highfreq_dir+house_dir+file)
    saveH5File(new_dir+(file.split('.'))[0]+'.h5',['info','wave_current'],[inf,current])


#Responsavel por selecionar as formas de onda de um sinal de alta frequencia para determinada combinaçao de aparelhos ligados
def dset_get_signal_form(high_sig,ts_high,list_sig,list_label,label,n,next_low_ts):
    if n<ts_high.shape[0]:
        if (next_low_ts - ts_high[n]) > 0:
            list_sig.append(high_sig[n])
            list_label.append(label)
            dset_get_signal_form(high_sig,ts_high,list_sig,list_label,label,n+1,next_low_ts)
        else:
            return
    else:
        return


def dsetBuildDataset(house_dir,highfreq_file,labels_main_file,dbase_file_name):

    #main_data: 0 - devices_index,
    #main_data: 1 - int_labels_mask,
    #main_data: 2 - labels,
    #main_data: 3 - times,
    tstamp_dir=h5_lowfreq_dir+house_dir+'timestamp_labels/'
    main_data=readH5File(tstamp_dir+labels_main_file)[1]
    devs_index,mask_labels,main_labels,ts_low = main_data
    print(ts_low.shape,main_labels.shape)


    highfreq_data=readH5File(h5_highfreq_dir+house_dir+highfreq_file)[1]
    inf,sig = highfreq_data
    ts_high = inf[:,0]

    v_counts=inf[:,1]

    print(sig.shape)
    print(ts_high.shape)
    l_sig=[]
    l_label=[]
    
    new_dir = dset_dir+house_dir

    if not os.path.exists(new_dir):
        try:
            os.makedirs(new_dir,0o777)
        except OSError:
            print('Creation of: '+new_dir+' failed.')
            return  

    for i in range(ts_low.shape[0]):
        print(i)
        label=main_labels[i]
        tpi=ts_low[i]
        n=np.where(ts_high<=tpi)[0][-1]
        l=[]
        #print(n)
        l_sig.append(sig[n])
        l_label.append(label)
        next_time=tpi+1
        #if i<(ts_low.shape[0]-1):
        #    next_time=ts_low[i+1]
        dset_get_signal_form(sig,ts_high,l_sig,l_label,label,n+1,next_time)



    #print(ts_low.shape)
    dset_sig=np.vstack(l_sig)
    dset_label=np.array(l_label)
    Y = np.reshape(dset_label,(dset_label.shape[0],1))
    dset=np.append(dset_sig,Y,1)
    np.random.shuffle(dset)
    X = dset[:,:-1]
    Y = dset[:,-1:]
    #print(X.shape)
    Y=Y.astype(int)
    #print(Y.shape)
    #c,u=np.unique(Y,return_counts=True)
    #print(c)
    #print(u)
    
    saveH5File(dset_dir+house_dir+dbase_file_name,['inputs','labels'],[X,Y])
    







#lowLabelMainTimestamps('house_3/',0,[3,5,7,9,11,13,14,15,17,19,20,21],0)
#lowLabelMainTimestamps('house_3/',1,[2,4,6,8,10,12,16,18],0)
#dsetBuildDataset('house_3/','current_1.h5','main1_labels.h5','dset_main1.h5')
#dsetBuildDataset('house_3/','current_2.h5','main2_labels.h5','dset_main2.h5')
