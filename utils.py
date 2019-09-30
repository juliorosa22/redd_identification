from basicLib import *
import pandas as pd


dir_path = '/home/julio/Framework_Projeto/Data_sets/ANN_data'
global_dsetPath =dir_path+'/Unified_Databases'




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
