from kerasBasics import *
from scipy.fftpack import fft

def getIndexData(l_indexes,useVal=False):
    val_idx=[l_indexes[0][:],l_indexes[1][:]]
    val_idx = np.concatenate(val_idx)
    lval_idx = val_idx.tolist()
    train_idx=[l_indexes[i][:] for i in range(2,len(l_indexes)) ]
    train_idx = np.concatenate(train_idx)
    ltrain_idx=train_idx.tolist()
    return (ltrain_idx,lval_idx)

def basic_func(x):
    return 1*x

def get_fft_values(y_values,N):
    #f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_values_ = fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return fft_values

def getBatchFourier(X):
    N=X.shape[1]
    l_fft=[]

    for i in range(X.shape[0]):
        l_fft.append(get_fft_values(X[i],N))

    #l_fft=getDset_features(X)
    l_fft=np.vstack(l_fft)
    return l_fft

## Generator class utilizado para que a base de dados de treinamento nao precise ser carregada totalmente na RAM, permitindo que o treinamento fique um pouco mais rapido
class SignalSequence(Sequence):

    def __init__(self,x,y,n_classes,list_index,func_process= basic_func,batch_size=32):
        self.x=x
        self.y=y
        self.n_classes=n_classes
        self.batch_size=batch_size
        self.func_process = func_process

        n_batches = int(np.ceil(len(list_index)/ float(self.batch_size)))


        copy_list = list_index[:]
        slice = len(copy_list) - (n_batches-1)*self.batch_size
        norm_list = copy_list[:-slice]
        rest_list=copy_list[-slice:]

        nlist_array = np.array(norm_list).astype(int)
        rest_array = np.array(rest_list).astype(int)
        final_list = np.split(nlist_array,(n_batches-1))
        final_list.append(rest_array)

        self.list_index=final_list
        #print(n_batches,len(self.list_index[-1]),len(self.list_index[-2]))




    def __len__(self):
        return len(self.list_index)

    def __getitem__(self, idx):

        x = self.x[(self.list_index[idx]).tolist()]
        y = self.y[(self.list_index[idx]).tolist()]

        y = to_categorical(y,num_classes=self.n_classes)
        x = self.func_process(x)##realiza alguma operação sobre as entradas

        return (x,y)
