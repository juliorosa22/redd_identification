from kerasBasics import *
from IPython.display import SVG
import h5py as h5



def plot_hist(history,label):
    plt.figure(1)
    #plt.plot(history[0])
    plt.plot(history[1])
    plt.title(label+': Model accuracy')
    plt.ylabel('Accuracy(%)')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()
    # Plot training & validation loss values
    plt.figure(2)
    #plt.plot(history[2])
    plt.plot(history[3])
    plt.title(label+': Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()

def showLogModel(log_file):
    abs_path = dir_models+'logs/'+log_file
    name,metrics=readHistLog(abs_path)
    plot_hist(metrics,name)

def getExcelResults(file_h5,file_path):

    df = pd.DataFrame({'Accuracy_training':file_h5['train_acc'][:]})
    df2 = pd.DataFrame({'Loss_training':file_h5['train_loss'][:]})
    df3 = pd.DataFrame({'Accuracy_val':file_h5['val_acc'][:]})
    df4 = pd.DataFrame({'Loss_val':file_h5['val_loss'][:]})


    writer = pd.ExcelWriter(file_path+'.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='acc_training')
    df2.to_excel(writer, sheet_name='loss_training')
    df3.to_excel(writer, sheet_name='acc_validation')
    df4.to_excel(writer, sheet_name='loss_validation')

    writer.save()

def checkModelAcc(model_file,xtest,ytest,plot_flag=False):
    abs_path = dir_models+model_file
    model = load_model(abs_path)
    #print(model.summary())
    acc=model.evaluate(x=xtest, y=ytest, batch_size=500, verbose=1)
    print(model_file[:-4]+" model accuracy ")
    print(acc)
    '''
    if plot_flag:
        plot_model(model,to_file=dir_models+'graphic1.png',show_shapes=True)
    '''
    return acc

#funcao que checa o desempenho de um modelo model_ver em identificar, mostrando resultados por aparelhos
def getInfoModel(dset_name,n_classes,last_index,model_dir,model_ver,labels_main=None):
    flag_labels_file=False
    dir_base='/low_freq/house_3/'
    n_devs=0
    if labels_main!=None:
        flag_labels_file=True
        infos_dset=mgr.read_MainData(labels_main)
        devs=infos_dset[3]
        mask_label=infos_dset[2]
        print(mask_label.shape)
        n_devs=devs.shape[0]-1
        print(devs)
        names = mgr.getDevicesNames(dir_base,'',devs)#pega os nomes dos aparelhos contidos no arquivo labels.txt
        print(names[1:])

    bacc=0
    bloss=0

    X = HDF5Matrix(datapath=dir_redd+dset_name, dataset='inputs', start=0, end=last_index)
    Y = HDF5Matrix(datapath=dir_redd+dset_name, dataset='labels', start=0, end=last_index)

    X=X[:]
    Y=Y[:]
    result_counts=np.zeros(n_devs,dtype=np.int32)
    pred_counts=np.zeros(n_devs,dtype=np.int32)


    indexes=np.arange(last_index)


    nfolds=10 #numero de folds para a validacao cruzada
    ix_folds = np.split(indexes,nfolds)
    acc_folds = []
    loss_folds=[]
    func = None
    for k in range(nfolds):
        print('Fold number :',k)
        model_name = model_dir+str(k)+'_'+model_ver
        '''
        print(model_name)
        idx_test = ix_folds[k]
        idx_test=np.sort(idx_test)
        idx_test=idx_test.tolist()
        xtest=X[idx_test]
        ytest=Y[idx_test]
        acc=model.evaluate(xtest,ytest,batch_size=500,verbose=2)
        '''
        model=load_model(dir_models+model_name+'.hdf5')
        #print(model.summary())
        test_gen=None
        i_test = ix_folds[k]
        if func != None:
            test_gen=dataGenerator(X,Y,n_classes,i_test,func_process=func,batch_size=1000)
        else:
            test_gen=dataGenerator(X,Y,n_classes,i_test,batch_size=1000)
        acc = model.evaluate_generator(test_gen,verbose=2)
        if acc[1] > bacc:
            bacc = acc[1]
            bloss = acc[0]

        ypred=model.predict_generator(test_gen)
        ytest = Y[i_test]
        ytest = to_categorical(ytest,num_classes=n_classes)
        c,pred_c=mgr.occurrenceDevInlabel(n_devs,mask_label,ypred,ytest)

        pred_counts=pred_counts+pred_c
        result_counts=result_counts+c

    print(model.summary())
    print("Numero de vezes que cada aparelho aparece:")
    print(result_counts)
    print("Numero acertos:")
    print(pred_counts)
    print("Numero de erros:")
    print(result_counts-pred_counts)
    print("porcentagem de acerto")
    perc=pred_counts/result_counts
    print(perc)
    print("Porcentagem de erro:")
    print((result_counts-pred_counts)/result_counts)
    print(bloss,bacc)


def getProbEstatistics(dset_name,n_classes,last_index,model_dir,model_ver,labels_main):
    flag_labels_file=False
    dir_base='/low_freq/house_3/'
    n_devs=0
    infos_dset=mgr.read_MainData(labels_main)
    devs=infos_dset[3]
    mask_label=infos_dset[2]
    print(mask_label)
    n_devs=devs.shape[0]-1
    print(devs)
    names = mgr.getDevicesNames(dir_base,'',devs)#pega os nomes dos aparelhos contidos no arquivo labels.txt
    print(names[1:])


    X = HDF5Matrix(datapath=dir_redd+dset_name, dataset='inputs', start=0, end=last_index)
    Y = HDF5Matrix(datapath=dir_redd+dset_name, dataset='labels', start=0, end=last_index)

    X=X[:]
    Y=Y[:]

    TFPN=np.zeros((4,n_devs),dtype=np.int32)
    PN = np.zeros((2,n_devs),dtype=np.int32)

    indexes=np.arange(last_index)

    nfolds=10 #numero de folds para a validacao cruzada
    ix_folds = np.split(indexes,nfolds)
    acc_folds = []
    loss_folds=[]
    func = None
    for k in range(nfolds):
        print('Fold number :',k)
        model_name = model_dir+str(k)+'_'+model_ver
        model=load_model(dir_models+model_name+'.hdf5')

        test_gen=None
        i_test = ix_folds[k]
        if func != None:
            test_gen=dataGenerator(X,Y,n_classes,i_test,func_process=func,batch_size=500)
        else:
            test_gen=dataGenerator(X,Y,n_classes,i_test,batch_size=500)

        ypred=model.predict_generator(test_gen)
        ytest = Y[i_test]
        #ytest = to_categorical(ytest,num_classes=n_classes)
        mgr.probAnalisys(n_devs,mask_label,ypred,ytest,TFPN,PN)


    dev_names=names[1:]
    scores_toexecel(model_ver,TFPN,PN,dev_names)

def getLogs(dir_logs,model_ver):
    nfolds = 10

    for i in range(nfolds):
        log_file =dir_models+dir_logs+'log_'+str(i)+'_'+model_ver+'.hdf5'
        print(log_file)
        name,metrics=readHistLog(log_file)
        metrics = np.vstack(metrics)



        plt.figure(1)
        plt.title('Dense Main2-modelo numero:'+str(i+1))
        plt.xlabel('Épocas')
        plt.ylabel('Acurácia(%)')
        plt.plot(100*metrics[0],label='Treinamento')
        plt.plot(100*metrics[1],label='Validação')
        plt.grid(color='k',linestyle='-')
        plt.legend()
        plt.figure(2)

        plt.title('Dense Main2-modelo numero:'+str(i+1))
        plt.ylabel('Função de Perda')
        plt.xlabel('Épocas')
        plt.plot(metrics[2],label='Treinamento')
        plt.plot(metrics[3],label='Validação')
        plt.grid(color='k',linestyle='-')
        plt.legend()
        plt.show()

        #print(metrics.shape)

def testKModel(dset_name,dir_type,model_ver,n_classes,last_index,k,func=None):

    X = HDF5Matrix(datapath=dir_redd+dset_name, dataset='inputs', start=0, end=last_index)
    Y = HDF5Matrix(datapath=dir_redd+dset_name, dataset='labels', start=0, end=last_index)

    X=X[:]
    Y=Y[:]

    indexes=np.arange(last_index)
    print('Dataset size X:(%d,%d)| Y:(%d,)'%(X.shape[0],X.shape[1],Y.shape[0]))

    nfolds=10 #numero de folds para a validacao cruzada
    ix_folds = np.split(indexes,nfolds)
    acc_folds = []
    loss_folds=[]

    #print('Fold number :',k)
    model_name = str(k)+'_'+model_ver
    i_test = ix_folds[k]
    print("Test Daset size: ",i_test.shape[0])
    #i_test=i_test.to_list()
    test_gen=None
    if func != None:
        test_gen=dataGenerator(X,Y,n_classes,i_test,func_process=func,batch_size=500)
    else:
        test_gen=dataGenerator(X,Y,n_classes,i_test,batch_size=500)
    print("Aqui")
    model = load_model(dir_models+dir_type+model_name+'.hdf5')
    #print(model.summary())
    acc=model.evaluate_generator(test_gen,verbose=1)

    #x=get_fft_values(xtest,275)


    return acc

'''
acc=[]
loss=[]

for i in range(10):
    print("Fold number:"+str(i))
    acc1 = testKModel('/full_main2_dset.hdf5','','lstm_main2_model',64,853040,i,None)
    #acc1 = testKModel('/full_main1_dset.hdf5','','conv_main1_model',187,858420,i)
    #acc1 = testKModel('/full_main2_dset.hdf5','','lstm_main2_model',64,853040,i,None)
    #acc1 = testKModel('/full_main1_dset.hdf5','/House_3/conv_models/','conv_main1_model',187,858420,i,getBatchFourier)
    #acc1 =testKModel('/full_main1_dset.hdf5','/House_3/dense_models/','dense_main1_model',187,858420,i,getBatchFourier)

    #loss.append(acc1[0])
    acc.append(acc1[1])

    #dense_loss.append(acc2[0])
    #dense_acc.append(acc2[1])
accv=np.array(acc)
print(accv)
print("Max:",accv.max())
print("Mean:",accv.mean())
print("STD:",accv.std())
'''

'''
model_ver = '0_dense_main1_model.hdf5'
#model_ver = '0_conv_main1_model.hdf5'
file = dir_models+'House_3/dense_models/'+model_ver
model = load_model(file)
print(model.summary())
#plot_model(model,to_file='/home/julio/conv_model1.png',expand_nested=True)
plot_model(model, to_file=model_ver+'.png', show_shapes=True, show_layer_names=False)
'''
#getProbEstatistics('/full_main2_dset.hdf5',64,853040,'','conv_main2_model',labels_main='/low_freq/house_3/data_main2_labels')
getProbEstatistics('/full_main1_dset.hdf5',187,858420,'','conv_main1_model',labels_main='/low_freq/house_3/data_main1_labels')

#test main 1 187,858420
#test main2 64,853040
