from kerasBasics import *
from basicLib import *
from REDD_management import *
import os.path as path
import manage_REDD as mgr
from dataGenerator import *

def trainDenseModel(model_name,train_gen,val_gen,n_epochs):

    print('Training model : ',model_name)


    #op=optimizers.SGD(lr=1.0, momentum=0.05, decay=0.0, nesterov=False)
    #op=optimizers.RMSprop(lr=0.7, rho=0.9, epsilon=None, decay=0.0)
    #op=optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
    #op=optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    #op=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    op=optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    es = EarlyStopping(monitor='val_loss',patience=100,verbose=1)

    inpL = Input(shape=(xtrain.shape[1],))
    hiddL = Dense(128,use_bias=True,activation='relu')(inpL)
    hiddL = Dropout(rate=0.2)(hiddL)
    hiddL = Dense(64,use_bias=True,activation='relu')(hiddL)
    hiddL = Dropout(rate=0.2)(hiddL)
    hiddL = Dense(32,use_bias=True,activation='relu')(hiddL)
    hiddL = Dropout(rate=0.2)(hiddL)

    outL = Dense(train_gen.n_classes,use_bias=True,activation='softmax')(hiddL)
    model = Model(inputs=inpL, outputs=outL)

    model.compile(loss='categorical_crossentropy',optimizer=op,metrics=['accuracy'])
    print(model.summary())

    #hist = model.fit(xtrain, ytrain,epochs=n_epochs, validation_split= 0.25,verbose=2 ,batch_size=500,callbacks=[es])
    hist=model.fit_generator(generator=train_gen,validation_data=val_gen, epochs=n_epochs, verbose=2, callbacks=[es])
    saveHistLog(model_name+'.hdf5',getArrayHistLog(hist))
    print("Treinamento completo")
    model.save(dir_models+model_name+'.hdf5')
    return model

def trainConvModel(model_name,train_gen,val_gen,n_epochs):
    print('Training model : ',model_name)
    #op=optimizers.SGD(lr=1.0, momentum=0.05, decay=0.0, nesterov=False)
    #op=optimizers.RMSprop(lr=0.7, rho=0.9, epsilon=None, decay=0.0)
    #op=optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
    #op=optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    #op=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    opoptimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    es = EarlyStopping(monitor='val_loss',patience=100,verbose=1)



    inpL = Input(shape=(275,))
    hiddL = Reshape((275,1))(inpL)

    hiddL = Conv1D(32, kernel_size=3, strides = 1, activation='relu')(hiddL)
    hiddL = Conv1D(32, kernel_size=3, strides = 1, activation='relu' )(hiddL)
    hiddL = MaxPooling1D(pool_size=3,strides=1)(hiddL)
    hiddL = BatchNormalization()(hiddL)

    hiddL= Conv1D(64, kernel_size=3, strides = 1, activation='relu' )(hiddL)
    hiddL= Conv1D(64, kernel_size=3, strides = 1, activation='relu' )(hiddL)
    hiddL = MaxPooling1D(pool_size=3,strides=1)(hiddL)
    hiddL = BatchNormalization()(hiddL)

    hiddL= Conv1D(64, kernel_size=3, strides = 1, activation='relu' )(hiddL)
    hiddL= Conv1D(64, kernel_size=3, strides = 1, activation='relu' )(hiddL)
    hiddL = GlobalMaxPooling1D()(hiddL)
    hiddL = BatchNormalization()(hiddL)

    #hiddL = Flatten()(hiddL)
    hiddL=Dense(256, activation='relu')(hiddL)
    hiddL = BatchNormalization()(hiddL)
    hiddL=Dense(128, activation='relu')(hiddL)

    outL = Dense(train_gen.n_classes,activation='softmax')(hiddL)

    model = Model(inputs=inpL, outputs=outL)

    model.compile(loss='categorical_crossentropy',optimizer=op,metrics=['accuracy'])
    print(model.summary())
    #mudar os argumentos caso nao for utilizar generator para pegar as amostras do batch
    #hist = model.fit(xtrain, ytrain,epochs=n_epochs, validation_split= 0.25,verbose=2 ,batch_size=500,callbacks=[es])
    hist=model.fit_generator(generator=train_gen,validation_data=val_gen, epochs=n_epochs, verbose=2, callbacks=[es])
    saveHistLog(model_name+'.hdf5',getArrayHistLog(hist))
    print("Treinamento completo")
    model.save(dir_models+model_name+'.hdf5')
    return model

def trainLSTMModel(model_name,train_gen,val_gen,n_epochs):
    print('Training model  : ',model_name)



    #op=optimizers.SGD(lr=1.0, momentum=0.05, decay=0.0, nesterov=False)
    #op=optimizers.RMSprop(lr=0.7, rho=0.9, epsilon=None, decay=0.0)
    #op=optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
    #op=optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    #op=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    op=optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    es = EarlyStopping(monitor='val_loss',patience=100,verbose=1)


    inpL = Input(shape=(275,))
    hiddL = Reshape((275,1,))(inpL)


    hiddL = LSTM(32,activation='tanh',recurrent_activation='hard_sigmoid')(hiddL)
    hiddL = Reshape((32,1,))(hiddL)
    hiddL = LSTM(32,activation='tanh',recurrent_activation='hard_sigmoid')(hiddL)
    hiddL = Reshape((32,1,))(hiddL)
    hiddL =LSTM(32,activation='tanh',recurrent_activation='hard_sigmoid')(hiddL)
    outL = Dense(train_gen.n_classes,use_bias=True,activation='softmax')(hiddL)

    model = Model(inputs=inpL, outputs=outL)


    model.compile(loss='categorical_crossentropy',optimizer=op,metrics=['accuracy'])
    print(model.summary())


    #hist = model.fit(xtrain, ytrain,epochs=n_epochs, validation_split= 0.25,verbose=1 ,batch_size=500,callbacks=[es])
    hist=model.fit_generator(generator=train_gen,validation_data=val_gen, epochs=n_epochs, callbacks=[es],verbose=1)
    saveHistLog(model_name+'.hdf5',getArrayHistLog(hist))
    print("Treinamento completo")
    model.save(dir_models+model_name+'.hdf5')
    return model



def buildModel(dset_name,dir_base,n_classes,last_index,model_ver):


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

    for k in range(nfolds):
        print('Fold number :',k)
        model_name = str(k)+'_'+model_ver#+dev_names[i_dev]

        idx_train = [ix_folds[i] for i in range(nfolds) if i!= k ]
        '''
        #qndo nao for usar generator para pegar os dados da base
        idx_train = [ix_folds[i] for i in range(nfolds) if i!= k ]
        idx_train=np.array(idx_train)
        idx_train=np.reshape(idx_train,(idx_train.shape[0]*idx_train.shape[1]))
        idx_train=np.sort(idx_train)
        idx_test = ix_folds[k]
        xtrain=X[idx_train]
        ytrain=Y[idx_train]
        ytrain=to_categorical(ytrain,num_classes=n_classes)
        '''
        i_train,i_val=getIndexData(idx_train)

        train_generator = dataGenerator(X,Y,n_classes,i_train,batch_size=500)
        val_generator = dataGenerator(X,Y,n_classes,i_val,batch_size=500)

        #print('Teste size:(%d,%d)| Y:(%d,%d)'%(xtest.shape[0],xtest.shape[1],ytest.shape[0],ytest.shape[1]))
        #Descomentar para escolher um dos tipos de rede
        model=trainLSTMModel(model_name,train_generator,val_generator,150)
        #model=trainModel(model_name,train_generator,val_generator,1000)
        #model=trainConvModel(model_name,train_generator,val_generator,300)







#buildModel('/full_main1_dset.hdf5','/low_freq/house_3/','lstm_main1_model')

buildModel('/full_main1_dset.hdf5','/low_freq/house_3/',187,858420,'test_main1_model')
#buildModel('/full_main2_dset.hdf5','/low_freq/house_3/',64,853040,'test_main2_model')



#dset_main1
#last_index =858420
#n_classes=187

#dset_main2
#last_index =853040
#n_classes=64
