from kerasBasics import *
import SignalSequence as ss


dset_dir = curr_work_dir+'/REDD_h5/datasets/'

##Arquivo q lida com o treinamento e validação cruzada dos modelos COnvet,lstm e MLP(dense)

def trainDenseModel(model_name,train_gen,val_gen,n_epochs):

    print('Training model : ',model_name)
    print('Number of epochs: ',n_epochs)

    #op=optimizers.SGD(lr=1.0, momentum=0.05, decay=0.0, nesterov=False)
    #op=optimizers.RMSprop(lr=0.7, rho=0.9, epsilon=None, decay=0.0)
    #op=optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
    #op=optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    #op=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    op=optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    es = EarlyStopping(monitor='val_loss',patience=50,verbose=1)
    csv_logger = CSVLogger(model_dir+'logs/'+model_name+'.csv')
    inpL = keras.Input(shape=(137,))
    hiddL = Dense(128,use_bias=True,activation='relu')(inpL)
    hiddL = Dropout(rate=0.2)(hiddL)
    hiddL = Dense(64,use_bias=True,activation='relu')(hiddL)
    hiddL = Dropout(rate=0.2)(hiddL)
    hiddL = Dense(32,use_bias=True,activation='relu')(hiddL)
    hiddL = Dropout(rate=0.2)(hiddL)

    outL = Dense(train_gen.n_classes,use_bias=True,activation='softmax')(hiddL)
    model = keras.Model(inputs=inpL, outputs=outL)

    model.compile(loss='categorical_crossentropy',optimizer=op,metrics=['accuracy'])
    print(model.summary())

    hist=model.fit(x=train_gen,validation_data=val_gen, epochs=n_epochs, verbose=2, callbacks=[csv_logger,es])
    print(hist)
    # saveHistLog(model_name+'.h5',getArrayHistLog(hist))
    # print("Treinamento completo")
   
    return model

def trainConvModel(model_name,train_gen,val_gen,n_epochs):
    print('Training model : ',model_name)
    #op=optimizers.SGD(lr=1.0, momentum=0.05, decay=0.0, nesterov=False)
    #op=optimizers.RMSprop(lr=0.7, rho=0.9, epsilon=None, decay=0.0)
    #op=optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
    #op=optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    #op=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    op=optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    es = EarlyStopping(monitor='val_loss',patience=100,verbose=1)
    csv_logger = CSVLogger(model_dir+'logs/'+model_name+'.csv')


    inpL = keras.Input(shape=(275,))
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

    model = keras.Model(inputs=inpL, outputs=outL)

    model.compile(loss='categorical_crossentropy',optimizer=op,metrics=['accuracy'])
    print(model.summary())
    #mudar os argumentos caso nao for utilizar generator para pegar as amostras do batch
    #hist = model.fit(xtrain, ytrain,epochs=n_epochs, validation_split= 0.25,verbose=2 ,batch_size=500,callbacks=[es])
    hist=model.fit(x=train_gen,validation_data=val_gen, epochs=n_epochs, verbose=2, callbacks=[csv_logger,es])
    # saveHistLog(model_name+'.h5',getArrayHistLog(hist))
    # print("Treinamento completo")
    
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
    csv_logger = CSVLogger(model_dir+'logs/'+model_name+'.csv')
    check = ModelCheckpoint(model_name+'weights.{epoch:02d}-{val_loss:.2f}.h5',monitor='val_loss')

    inpL = keras.Input(shape=(275,))
    hiddL = Reshape((275,1,))(inpL)


    hiddL = LSTM(32,activation='tanh',recurrent_activation='hard_sigmoid')(hiddL)
    hiddL = Reshape((32,1,))(hiddL)
    hiddL = LSTM(32,activation='tanh',recurrent_activation='hard_sigmoid')(hiddL)
    hiddL = Reshape((32,1,))(hiddL)
    hiddL =LSTM(32,activation='tanh',recurrent_activation='hard_sigmoid')(hiddL)
    outL = Dense(train_gen.n_classes,use_bias=True,activation='softmax')(hiddL)

    model = keras.Model(inputs=inpL, outputs=outL)


    model.compile(loss='categorical_crossentropy',optimizer=op,metrics=['accuracy'])
    print(model.summary())


    #hist = model.fit(xtrain, ytrain,epochs=n_epochs, validation_split= 0.25,verbose=1 ,batch_size=500,callbacks=[es])
    hist=model.fit(x=train_gen,validation_data=val_gen, epochs=n_epochs, callbacks=[es,check,CSVLogger],verbose=2)
    
    return model


def buildModel(house_dir,dset_file,model_ver,n_classes,nfolds):
    new_dir = model_dir+house_dir

    if not os.path.exists(new_dir):
        try:
            os.makedirs(new_dir,0o777)
        except OSError:
            print('Creation of: '+new_dir+' failed.')
            return  

    
    X=h5.File(dset_dir+house_dir+dset_file,'r')['inputs']
    Y=h5.File(dset_dir+house_dir+dset_file,'r')['labels']
    print('Dataset size:')
    print(X.shape)
    
    indexes=np.arange(X.shape[0])
    #necessario 
    
    r = X.shape[0]%nfolds
    ix_folds=np.split(indexes[:-r],nfolds) if r>0 else np.split(indexes[:],nfolds)
    if r>0:
        ix_folds[-1] = np.append(ix_folds[-1],indexes[-r:])
    
    
    acc_folds = []
    loss_folds=[]

    for k in range(nfolds):
        print('Fold number :',k)
        model_name = model_ver+'_'+str(k)

        idx_train = [ix_folds[i] for i in range(nfolds) if i!= k ]
        
        i_train,i_val=ss.getIndexData(idx_train)
        #no caso do modelo MLP é preciso aplicar a funcao getBatchFourier, caso contrario o argumento pode ser None
        train_generator = ss.SignalSequence(X,Y,n_classes,i_train,batch_size=500,func_process=ss.getBatchFourier)
        val_generator = ss.SignalSequence(X,Y,n_classes,i_val,batch_size=500,func_process=ss.getBatchFourier)

        
        #Descomentar para escolher um dos tipos de rede
        #model=trainLSTMModel(model_name,train_generator,val_generator,50)
        model=trainDenseModel(model_name,train_generator,val_generator,5)
        #model=trainConvModel(model_name,train_generator,val_generator,300)
        
        model.save(model_dir+house_dir+model_name+'.h5')

        ##Para realizar o teste do modelo no fold de test
        '''
        idx_test = ix_folds[k]
        x_test=X[idx_test]
        #x_test = ss.getBatchFourier(x_test)#se for o modelo dense
        y_test = Y[idx_test]
        loss,acc=model.evaluate(x_test,to_categorical(y_test,num_classes=n_classes),verbose=2)
        print('Fold test results:')
        print('Test Loss: ',loss)
        print('Test Accuracy: ',acc)
        '''

#buildModel('house_3/','dset_main1.h5','dense',187,10)
buildModel('house_3/','dset_main2.h5','dense',64,10)