import keras
from keras.models import Sequential, load_model,Model
from keras.layers import Conv1D, Reshape,Dense, LSTM,GlobalMaxPooling1D,MaxPooling1D,GlobalAveragePooling1D, Flatten, Input,Dropout,BatchNormalization,Embedding
#from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import plot_model,to_categorical,HDF5Matrix,Sequence
from keras import optimizers,regularizers
from keras import metrics
import manage_REDD as mgr
from basicLib import *

from keras.callbacks import EarlyStopping

dir_models = '/home/julio/Framework_Projeto/Models/KerasModels/'

dir_redd = '/home/julio/Framework_Projeto/Data_sets/REDD'

dir_base=''

def convert_excel(data,file_name):
    idx = np.arange(data.shape[1])

    val = np.vstack([data[1],data[3]])

    train=np.vstack([data[0],data[2]])


    df_val = pd.DataFrame(data=val.T,index=idx,columns=['Accuracy','Loss'])
    df_train=pd.DataFrame(data=train.T,index=idx,columns=['Accuracy','Loss'])


    writer = pd.ExcelWriter(file_name+'.xlsx', engine='xlsxwriter')
    df_val.to_excel(writer, sheet_name='val_data')
    df_train.to_excel(writer, sheet_name='train_data')
    writer.save()

def getArrayHistLog(history):
    train_acc = np.array(history.history['acc'],dtype = np.float64)
    val_acc = np.array(history.history['val_acc'],dtype = np.float64)
    train_loss =np.array(history.history['loss'],dtype = np.float64)
    val_loss = np.array(history.history['val_loss'],dtype = np.float64)
    return [train_acc,val_acc,train_loss,val_loss]

def saveHistLog(file_path,lmetrics):
    train_acc = lmetrics[0]
    val_acc = lmetrics[1]
    train_loss = lmetrics[2]
    val_loss = lmetrics[3]
    f = h5py.File(dir_models+'log_'+file_path,'w')
    tacc_dset = f.create_dataset("train_acc",train_acc.shape,dtype = 'f8')
    vacc_dset = f.create_dataset("val_acc",val_acc.shape,dtype = 'f8')
    tloss_dset = f.create_dataset("train_loss",train_loss.shape,dtype = 'f8')
    vloss_dset=f.create_dataset("val_loss",val_loss.shape,dtype = 'f8')

    tacc_dset[:] = train_acc[:]
    vacc_dset[:] = val_acc[:]
    tloss_dset[:] = train_loss[:]
    vloss_dset[:]= val_loss[:]
    f.close()

def readHistLog(file_path):
    file_path=dir_models+file_path
    f = h5py.File(file_path, 'r')
    train_acc =f['train_acc'][:150]
    val_acc=f['val_acc'][:150]
    train_loss=f['train_loss'][:150]
    val_loss=f['val_loss'][:150]
    f.close()
    name = file_path.split('/')[-1]
    metrics = [train_acc,val_acc,train_loss,val_loss]
    return name,metrics

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
def getInfoModel(dset_name,n_classes,last_index,model_ver,labels_main=None):
    flag_labels_file=False
    if labels_main!=None:
        #abre o arquivo que contem informações sobre a quantidade de combinações e os respectivos aparelhos que cada rotulo significa
        flag_labels_file=True
        infos_dset=mgr.read_MainData(labels_main)
        devs=infos_dset[3]
        mask_label=infos_dset[2]
        names = mgr.getDevicesNames(dir_base,'',devs)#pega os nomes dos aparelhos contidos no arquivo labels.txt
        print(names[1:])


    #abre apenas a base de dados para verificar a acuracia do modelo
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

    for k in range(nfolds):
        print('Fold number :',k)
        model_name = str(k)+'_'+model_ver#+dev_names[i_dev]

        idx_test = ix_folds[k]
        idx_test=np.sort(idx_test)
        idx_test=idx_test.tolist()
        xtest=X[idx_test]
        ytest=Y[idx_test]

        model=load_model(dir_models+model_name+'.hdf5')
        acc=model.evaluate(xtest,ytest,batch_size=500,verbose=2)
        print('model_accuracy: ',acc)
        if flag_labels_file:
            ypred=model.predict(xtest,verbose=1,batch_size=500)
            c,pred_c=mgr.occurrenceDevInlabel(n_devs,mask_label,ypred,ytest)

            pred_counts=pred_counts+pred_c
            result_counts=result_counts+c

    if flag_labels_file:
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
