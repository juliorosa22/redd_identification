from basicLib import *
from scipy.special import expit as sigmoid_func
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
global_nsamples = 3600


def sigmoid(z):
    return sigmoid_func(z)


def grad_sigmoid(x):
    return  np.multiply(x,(1-x))

def reg_L2(l_weights,lmbda):
    r=0
    if lmbda>0.0:
        for w in l_weights:
            w_square = np.multiply(w,w)
            w_square[:,0]=0
            vw=np.sum(w_square,1)
            r+=np.sum(vw,0)
    return ((lmbda/2)*r)

def cross_Entropy(outNeurons,Y,l_weights,lmbda):

    A = outNeurons
    n= Y.shape[0]#obtem o numero de entradas de treinamento
    J = (np.multiply((-1)*Y,np.log(A)) - np.multiply((1-Y),np.log((1-A))) )
    J = np.sum(J,1)
    J = np.sum(J,0)
    reg = reg_L2(l_weights,lmbda)
    return ((1/n)*(J+reg))

def truncated_normal(mean, sd, low, upp):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

class NeuralNetwork:

    def __init__(self,sizes,cost_func,activation_func,grad_func):#sizes é uma lista contendo a quantidade de neuronio em cada camada da rede, incluindo a camada de entrada'''
        self.nLayers = len(sizes) #'''quantidade de camadas incluindo a camada de entrada'''
        self.nNeurons = sizes
        self.activation_func = activation_func
        self.cost_func = cost_func
        self.grad_func = grad_func
        l = 1.0/math.sqrt(self.nNeurons[0])
        #g = truncated_normal(0,1,-l,l)
        #self.netWeights = [g.rvs( (self.nNeurons[i],self.nNeurons[i-1]+1)) for i in range(1,self.nLayers) ]
        self.netWeights = [ np.random.normal(0,1.0/math.sqrt(self.nNeurons[i-1]),(self.nNeurons[i],self.nNeurons[i-1]+1))  for i in range(1,self.nLayers) ]
        self.netA=[[] for i in range(self.nLayers)]

    @staticmethod
    def genericFeedForward(X,l_weights,ac_func):
        layers = [[] for i in range(len(l_weights)+1)]
        layers[0] = X
        nLayers = len(layers)
        for i in range(1,nLayers):
            bias=np.ones((layers[i-1].shape[0],1))
            inp_bias=np.concatenate((bias,layers[i-1]),1 )
            z=np.dot(inp_bias, l_weights[i-1].T)
            a =ac_func(z)
            layers[i]=a
        return layers

    @staticmethod
    def genericBackpropagation(E,layers,l_weights,grad_func,lmbda):
        nLayers = len(layers)
        n = E.shape[0]
        l_deltas=[ [] for i in range(nLayers) ]
        l_grads=[[] for i in range(nLayers)]
        l_deltas[-1]=E
        #faz a propagação do erro para as camadas ocultas
        for i in reversed(range(1,nLayers -1)):
            prod=np.dot(l_deltas[i+1],l_weights[i])
            prod = prod[:,1:]
            grad_z=grad_func(layers[i])
            dl= np.multiply(prod,grad_z)
            l_deltas[i]=dl
        #calcula as derivadas da funcao de custo em relação aos pesos da rede
        for i in range(1,nLayers):
            bias = np.ones((layers[i-1].shape[0],1))
            a = np.concatenate((bias,layers[i-1]),1 )
            grad=np.dot(l_deltas[i].T,a)
            w_unbiased = np.array(l_weights[i-1])
            w_unbiased[:,0]=0
            l_grads[i] = (1/n)*(grad+((lmbda)*w_unbiased))

        return l_grads[1:]


    @staticmethod
    def compNumericalGrad(X,Y,l_weights,ac_func,cost_func,lmbda):
        nWeights = len(l_weights)
        l_compNumGrads=[np.zeros(l_weights[n].shape,dtype = np.float64) for n in range(nWeights)]
        eps = 1e-5
        #print(eps)
        for n in range(nWeights):
            l_waux = l_weights[:]#copia a lista com as matrizes de pesos da rede
            for i in range(l_weights[n].shape[0]):
                for j in range(l_weights[n].shape[1]):
                    w_aux = l_waux[n][i,j]

                    l_waux[n][i,j] = w_aux-eps
                    layers1 = NeuralNetwork.genericFeedForward(X,l_waux,ac_func)
                    loss1 = cost_func(layers1[-1],Y,l_weights,lmbda)

                    l_waux[n][i,j] = w_aux+eps
                    layers2 = NeuralNetwork.genericFeedForward(X,l_waux,ac_func)
                    loss2 = cost_func(layers2[-1],Y,l_weights,lmbda)

                    l_compNumGrads[n][i,j] = (loss2 - loss1)/(2*eps)
                    l_waux[n][i,j] = w_aux
        return l_compNumGrads

    @staticmethod
    def checkNNGradients(X,Y,l_weights,ac_func,grad_func,cost_func,lmbda):

        layers=NeuralNetwork.genericFeedForward(X,l_weights,ac_func)
        A_out = layers[-1]
        E = A_out - Y
        l_grads = NeuralNetwork.genericBackpropagation(E,layers,l_weights,grad_func,lmbda)
        numGrads=NeuralNetwork.compNumericalGrad(X,Y,l_weights,ac_func,cost_func,lmbda)

        v_numGrads = np.reshape(numGrads[0],(1,numGrads[0].shape[0]*numGrads[0].shape[1]))
        v_Grads = np.reshape(l_grads[0],(1,l_grads[0].shape[0]*l_grads[0].shape[1]))

        for i in range(1,len(numGrads)):
            vnum = np.reshape(numGrads[i],(1,numGrads[i].shape[0]*numGrads[i].shape[1]))
            vgrad  =np.reshape(l_grads[i],(1,l_grads[i].shape[0]*l_grads[i].shape[1]))
            v_numGrads=np.concatenate((v_numGrads,vnum),1)
            v_Grads=np.concatenate((v_Grads,vgrad),1)
        print('Tamanhos')
        print(v_Grads.shape)
        print(v_numGrads.shape)
        ac = np.linalg.norm(v_numGrads - v_Grads)/((np.linalg.norm(v_numGrads)+np.linalg.norm(v_Grads)))
        #np.linalg.norm(v_numGrads+v_Grads)
        #max(np.linalg.norm(v_numGrads),np.linalg.norm(v_Grads))
        #np.linalg.norm(v_numGrads)+np.linalg.norm(v_Grads)
        return ac


    def checkGradients_training(self,layers,X,Y,l_grads,lmbda):
        E = layers[-1]-Y
        numGrads = NeuralNetwork.compNumericalGrad(X,Y,self.netWeights,self.activation_func,self.cost_func,lmbda)
        v_numGrads = np.reshape(numGrads[0],(1,numGrads[0].shape[0]*numGrads[0].shape[1]))
        v_Grads = np.reshape(l_grads[0],(1,l_grads[0].shape[0]*l_grads[0].shape[1]))
        for i in range(1,len(numGrads)):
            vnum = np.reshape(numGrads[i],(1,numGrads[i].shape[0]*numGrads[i].shape[1]))
            vgrad  =np.reshape(l_grads[i],(1,l_grads[i].shape[0]*l_grads[i].shape[1]))
            v_numGrads=np.concatenate((v_numGrads,vnum),1)
            v_Grads=np.concatenate((v_Grads,vgrad),1)

        ac = np.linalg.norm(v_numGrads - v_Grads)/(np.linalg.norm(v_numGrads)+np.linalg.norm(v_Grads))
        return ac


    @staticmethod
    def saveModel_status(file_name,l_weights,costs,acc):
        n =len(l_weights)
        print(file_name)
        f =h5py.File(file_name, 'w')
        wDsets = [[] for i in range(n)]
        for i in range(n):
            name = 'weight_'+str(i)
            print(name)
            wDsets[i]=f.create_dataset(name,l_weights[i].shape,dtype='f8')
            wDsets[i][:] = l_weights[i][:]
        cost_dset = f.create_dataset('costs_dset',costs.shape,dtype='f8')
        cost_dset[:] = costs[:]
        acc_dset = f.create_dataset('acc_dset',acc.shape,dtype='f8')
        acc_dset[:] = acc[:]

        f.close()

    @staticmethod
    def readModel(file):
        f = h5py.File(file,'r')
        dset_names = list(f.keys())
        n = len(dset_names)
        #print(dset_names)
        acc = f[dset_names[0]][:]
        costs = f[dset_names[1]][:]
        W=[]
        for i in range(2,n):
            W.append(f[dset_names[i]][:])
            #print(W[i-2].shape)
        return W,acc,costs


    def resetWeights(self):
        #l = 1.0/math.sqrt(self.nNeurons[0])
        self.netWeights = [ np.random.normal(0,1.0/math.sqrt(self.nNeurons[i-1]),(self.nNeurons[i],self.nNeurons[i-1]+1))  for i in range(1,self.nLayers) ]

    def checkAcc(self,y_out,y_val,bound):
        print('In checkAcc')
        print(y_out.shape)
        print(y_val.shape)

        check_out=np.greater(y_out,bound)
        check_label = np.greater(y_val,bound)
        check = np.equal(check_out,check_label)
        acc = 100*np.sum(check)/y_val.shape[0]
        return acc

    def BGD(self,X,Y,l_rate,n_iterations,lmbda):
        print('Tamanho das amostras %d'%(X.shape[0]))
        C =[]
        t=[]
        print('Batch Gradient Descent trainning')
        for i in range(n_iterations):
            layers=NeuralNetwork.genericFeedForward(X,self.netWeights,self.activation_func)
            y_out =  layers[-1]
            E = y_out - Y
            grads = NeuralNetwork.genericBackpropagation(E,layers,self.netWeights,self.grad_func,lmbda)
            c=self.cost_func(y_out,Y,self.netWeights,lmbda)
            print("iter: %d cost: %f"%(i,c))
            for j in range(len(self.netWeights)):
                w = self.netWeights[j]
                self.netWeights[j] =  w - (l_rate)*grads[j]
            C.append(c)
        C =np.array(C,dtype = np.float64)
        #t = np.arange(C.shape[0])
        return C


    def MGD(self,X,Y,l_rate,epochs,lmbda,m):#m é o tamanho de cada mini batch , deve ser um numero inteiro tal que a divisao de X  seja inteira e igual
        nBatches = math.floor(X.shape[0]/m)
        x_batches = np.split(X,nBatches)
        y_batches = np.split(Y,nBatches)
        print('Mini Batch Gradient Descent trainning')
        print('number of minibatches: {}'.format(nBatches))
        C = []
        for ep in range(epochs):#numero de epocas
            c=[]
            for i in range(nBatches):#itera sobre todos os minibatches
                x_miniB = x_batches[i]
                y_miniB = y_batches[i]
                layers = NeuralNetwork.genericFeedForward(x_miniB,self.netWeights,self.activation_func)
                cost =self.cost_func(layers[-1],y_miniB,self.netWeights,lmbda)
                #print(cost)
                c.append(cost)
                E = layers[-1] - y_miniB
                grads=NeuralNetwork.genericBackpropagation(E,layers,self.netWeights,self.grad_func,lmbda)
                tx= (1/(1+0.5*ep) )*l_rate

                for j in range(len(self.netWeights)):
                    w = self.netWeights[j]
                    self.netWeights[j] = w - (tx)*grads[j]

            c =np.array(c,dtype = np.float64)
            mean =np.mean(c)
            print('Epoch[%d],Mean Cost:%f'%(ep,mean))
            #print('Mean Cost %f'%(mean))
            C.append(mean)
        C = np.array(C)
        return C

    def k_foldCrossVal(self,k,dset,l_rate,n_trainIter,lmbda,m):
        folds = np.split(dset,k)
        costs=[]
        acc=[]
        lw=[]
        perc=0
        for i in range(k):
            print('Fold %d'%(i))
            fold_train  = [folds[j] for j in range(k) if j!=i]
            dset_train = np.vstack(fold_train)
            x_train = dset_train[ : ,:-(self.netWeights[-1].shape[0])]
            y_train = dset_train[:,-(self.netWeights[-1].shape[0]):]
            c = self.MGD(x_train,y_train,l_rate,n_trainIter,lmbda,m)

            #c = self.BGD(x_train,y_train,l_rate,n_trainIter,lmbda)
            costs.append(c)
            l_c = c[-1]
            #realiza a validação
            dset_val = folds[i]
            x_val = dset_val[ : ,:-(self.netWeights[-1].shape[0])]
            y_val = dset_val[:,-(self.netWeights[-1].shape[0]):]
            layers=NeuralNetwork.genericFeedForward(x_val,self.netWeights,self.activation_func)
            out  = layers[-1]
            mean_perc = self.checkAcc(out,y_val,0.5)
            print('Media no fold : %f'%(mean_perc))
            acc.append(mean_perc)
            if mean_perc>perc:
                lw=[self.netWeights[i][:] for i in range(len(self.netWeights))]
            #reseta os pesos
            self.resetWeights()

        costs = np.array(costs,dtype = np.float64)
        acc = np.array(acc)
        return lw,acc,costs



        '''


    def holdout(self,perc_hold,X,Y,l_rate,n_trainIter,lmbda,miniB_size):
        n = math.floor(X.shape[0]*perc_hold)
        print(n)
        x_val =  X[:n]
        y_val = Y[:n]
        x_train = X[n:]
        y_train = Y[n:]
        print(y_train.shape[0])
        self.SGD(x_train,y_train,l_rate,n_trainIter,lmbda)
        layers=NeuralNetwork.genericFeedForward(x_val,self.netWeights,self.activation_func)
        out  = layers[-1]
        index_out = np.argmax(out,1)
        index_labels = np.argmax(y_val,1)
        t = np.equal(index_out,index_labels)
        print(t)
        mean=np.sum(t)/x_val.shape[0]
        print(mean)
    '''
