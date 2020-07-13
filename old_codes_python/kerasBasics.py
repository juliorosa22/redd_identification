import keras
from keras.models import Sequential, load_model,Model
from keras.layers import Conv1D, Reshape,Dense, LSTM,GlobalMaxPooling1D,MaxPooling1D,GlobalAveragePooling1D, Flatten, Input,Dropout,BatchNormalization,Embedding
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import plot_model,to_categorical,HDF5Matrix,Sequence
from keras import optimizers,regularizers
from keras import metrics
import manage_REDD as mgr
from basicLib import *
from dataGenerator import *

from keras.callbacks import EarlyStopping,ModelCheckpoint

dir_models = '/home/julio/Framework_Projeto/Models/KerasModels/'

dir_redd = '/home/julio/Framework_Projeto/Data_sets/REDD'

def scores_toexecel(file_name,TFPN,PN,names):

    P=PN[0]
    N=PN[1]

    TP = TFPN[0]
    FP = TFPN[1]
    TN = TFPN[2]
    FN = TFPN[3]
    recall=TP/(TP+FN)
    precision= TP/(TP+FP)
    acc=(TP+TN)/(P+N)

    f1_score=2*((precision*recall)/(precision+recall))

    data_d = np.vstack([recall,precision,acc,f1_score])
    data_tfpn=np.append(TFPN,PN,0)

    tfpn_pd=pd.DataFrame(data=data_tfpn.T,index=names,columns=['true_positive','false_positive','true_negative','false_negative','total_positive','total_negative'])
    scores_pd=pd.DataFrame(data=data_d.T,index=names,columns=['recall','precision','acc','f1_score'])
    writer = pd.ExcelWriter(file_name+'_stats.xlsx', engine='xlsxwriter')

    tfpn_pd.to_excel(writer, sheet_name='Individual P_N')

    scores_pd.to_excel(writer, sheet_name='scores')

    writer.save()

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
    f = h5py.File(file_path, 'r')
    train_acc =f['train_acc'][:150]
    val_acc=f['val_acc'][:150]
    train_loss=f['train_loss'][:150]
    val_loss=f['val_loss'][:150]
    f.close()
    name = file_path.split('/')[-1]
    metrics = [train_acc,val_acc,train_loss,val_loss]
    return name,metrics
