from sklearn.svm import LinearSVC
from joblib import dump, load
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from dataGenerator import *
import time
import h5py as h5
svm_dir = 'House_3/SVM/'

def checkResults(data_set_file,dir_house,model_ver):
    X,Y,dev_names,cm = readUnifiedDset(data_set_file,dir_house)
    dev_names = dev_names[0]
    print('Aparelhos presentes no banco de dados: ',dev_names)

    Y = convertLabels(Y,5)
    print(Y.shape)
    i_dev = int(input("Escolha um indice para testar um SVM  para um aparelho:"))

    Ydev = Y[:,i_dev]
    print(Ydev.shape)
    Ydev = np.reshape(Ydev,(Ydev.shape[0],1))

    num_classes =Ydev.shape[1]
    Ydev = Ydev.astype(int)
    Dataset =np.append(X,Ydev,1)
    folds = np.split(Dataset,10)
    acc_folds = np.zeros(10,dtype = np.float64)
    loss_folds = np.zeros(10,dtype = np.float64)
    model_name = dev_names[i_dev]
    cv=10
    for k in range(cv):
        file_path = dir_models+str(k)+'_'+model_ver+model_name+'.joblib'
        test = folds[k]
        xtest=test[:,:-num_classes]
        ytest = test[:,-num_classes:]
        model = load(file_path)
        score=model.score(xtest,ytest)
        acc_folds[k]=score
        loss_folds[k]=score
    info_test = np.vstack([acc_folds,loss_folds])
    plt.plot(info_test[0])
    plt.plot(info_test[1])
    plt.show()
    print('Resultados:')
    print(info_test)
    save_DevSignals(info_test,'/info_folds_SVM_'+model_ver+model_name)

def trainSvm(dset_name,model_ver,last_index):

    X = HDF5Matrix(datapath=dir_redd+dset_name, dataset='inputs', start=0, end=last_index)
    Y = HDF5Matrix(datapath=dir_redd+dset_name, dataset='labels', start=0, end=last_index)
    X=X[:]
    X = getBatchFourier(X)
    Y=Y[:]

    indexes=np.arange(last_index)
    print('Dataset size X:(%d,%d)| Y:(%d,)'%(X.shape[0],X.shape[1],Y.shape[0]))

    nfolds=10 #numero de folds para a validacao cruzada
    ix_folds = np.split(indexes,nfolds)
    acc_folds = []
    loss_folds=[]

    for k in range(nfolds):
        print('Fold number :',k)
        model_name = str(k)+'_'+model_ver#+dev_names[i_dev]

        idx_train = [ix_folds[i] for i in range(nfolds) if i!= k ]

        idx_train = [ix_folds[i] for i in range(nfolds) if i!= k ]
        idx_train=np.array(idx_train)
        idx_train=idx_train.flatten()
        idx_train=np.sort(idx_train)

        xtrain=X[idx_train]
        ytrain=Y[idx_train]
        ytrain = ytrain.flatten()
        idx_test = ix_folds[k]
        xtest = X[idx_test]
        ytest = Y[idx_test]
        ytest = ytest.flatten()
        print('Training svm to :',model_name)

        #Using linear SVM
        svm_model = LinearSVC(dual=False, tol=0.0001, C=1.0, multi_class='ovr')
        #svm_model=SVC(kernel='linear',gamma='auto')
        time1 = time.time()
        svm_model.fit(xtrain,ytrain)
        time2 = time.time()
        print('Timing : {:.3f} ms'.format( (time2-time1)*1000.0))
        '''
        test_scores = svm_model.score(xtest,ytest)
        print("Resultado do SVM")
        print(test_scores)
        dump(svm_model,dir_models+model_name+'.joblib')
        '''


def testSvm(dset_name,labels_main,model_ver,last_index):
    '''
    dir_base='/low_freq/house_3/'
    infos_dset=mgr.read_MainData(labels_main)
    devs=infos_dset[3]
    mask_label=infos_dset[2]
    print(mask_label.shape)
    n_devs=devs.shape[0]-1
    print(devs)
    names = mgr.getDevicesNames(dir_base,'',devs)#pega os nomes dos aparelhos contidos no arquivo labels.txt
    print(names[1:])
    '''
    X = HDF5Matrix(datapath=dir_redd+dset_name, dataset='inputs', start=0, end=last_index)
    Y = HDF5Matrix(datapath=dir_redd+dset_name, dataset='labels', start=0, end=last_index)
    X=X[:]
    X = getBatchFourier(X)
    Y=Y[:]



    indexes=np.arange(last_index)
    print('Dataset size X:(%d,%d)| Y:(%d,)'%(X.shape[0],X.shape[1],Y.shape[0]))

    nfolds=10 #numero de folds para a validacao cruzada
    ix_folds = np.split(indexes,nfolds)
    acc_folds = []
    loss_folds=[]


    #result_counts=np.zeros(n_devs,dtype=np.int32)
    #pred_counts=np.zeros(n_devs,dtype=np.int32)

    for k in range(nfolds):
        print('Fold number :',k)
        model_name = str(k)+'_'+model_ver#+dev_names[i_dev]

        idx_test = ix_folds[k]
        xtest = X[idx_test]
        ytest = Y[idx_test]
        ytest = ytest.flatten()
        print('Training svm to :',model_name)

        #Using linear SVM
        svm_model = load(dir_models+svm_dir+model_name+'.joblib')
        xa = np.random.randint(10,size=(1,137))
        xa=xa.astype(float)
        time1 = time.time()
        pred=svm_model.predict(xa)
        time2 = time.time()
        print('Timing : {:.3f} ms'.format( (time2-time1)*1000.0))
        '''
        test_scores = svm_model.score(xtest,ytest)
        print("Resultado do SVM")
        print(test_scores)
        acc_folds.append(test_scores)
        '''
        #c,pred_c=mgr.occurrenceDevInlabel_ver2(n_devs,mask_label,ypred,ytest)

        #pred_counts=pred_counts+pred_c
        #result_counts=result_counts+c
    '''
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
    '''
    #acc = np.array(acc_folds)
    #f=h5py.File(dir_models+'acc_'+model_ver+'.hdf5','w')
    #acc_test=f.create_dataset('labels',acc.shape,dtype=acc.dtype)
    #acc_test[:]=acc[:]
    #f.close()
#cross validation

trainSvm('/full_main1_dset.hdf5','svm_main1_model',858420)
#trainSvm('/full_main2_dset.hdf5','svm_main2_model',853040)
#testSvm('/full_main1_dset.hdf5','svm_main1_model',858420)
#testSvm('/full_main1_dset.hdf5','/low_freq/house_3/data_main1_labels','svm_main1_model',858420)


'''
file_path = dir_models+sys.argv[1]
model = load(file_path)
s=model.score(xtest,ytest)
print(s)
'''
