# -*- coding: utf-8 -*-
"""
Implementation code for proposed model in "Semi-supervised Violin Finger Generation Using Variational Autoencoders" by Vincent K.M. Cheung, Hsuan-Kai Kao, and Li Su in Proc. of the 22nd Int. Society for Music Information Retrieval Conf., Online, 2021.

"""


import dataset_prepare as vfp
from custom_layers import Sampling, KLDivergenceLayer

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import random as python_random
from tensorflow.keras import regularizers

import copy
import matplotlib.pyplot as plt
from tensorflow.keras.utils import custom_object_scope



gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


#%% Presets
vae_enc_units = 64 #number of hidden units in RNN (need to x2 if BLSTM) 
vae_dec_units = 128 #number of hidden units in RNN (need to x2 if BLSTM) 
latent_dim = 16 #number of units in latent space 

rnn_units = 128   #number of hidden units in classifier RNN (need to x2 if BLSTM) 
cnn_units = 16  #number of units in CNN (16 heads in inception) 

n_dim1 = 64 #64 number of units in dense layer of embedder
n_dim2 = n_dim1 #number of incoming units in classifier and encoder
n_blstm = 256

training_corpus_name = 'vf_dataset_window32_gap16.pickle'
testing_corpus_name = 'vf_dataset_window32_gap32.pickle'

training_full = vfp.load_data(training_corpus_name)
training_corpus = {k: v for k, v in training_full.items() if 'vio2_' in k} # only use vio2_
testing_full = vfp.load_data(testing_corpus_name)
testing_corpus = {k: v for k, v in testing_full.items() if 'vio2_' in k} # also only use vio2_


epochs = 350
batch_size = 32

validation_split = 0.05
monitor = 'val_loss' 
patience = 10 #--20
restore_best_weights = True


#### Input dimensions
embedding_size_pitch = 16 
embedding_size_start = 32
embedding_size_duration = 8
embedding_size_pure_pitch = 8 #pitch classes
embedding_size_pure_octave = 4 #octaves

n_pitch = 46+1 #midi numbers 55 to 100 + number 54 for invalid notes
n_start = 96 #only 56 in dataset
n_duration = 32 #only 23 in dataset
n_pure_pitch = 21+1 #include enharmonics separately (to hopefully get key information also)
n_pure_octave = 5 

n_string = 4
n_position = 12
n_finger = 5
n_spf = 240+1 #4*12*5+1 but 133 in dataset, extra to handle new data


pretrain_needed = 0

if (pretrain_needed == 1):
    #%% Joint (string,position,finger) dictionary 
    spf = np.zeros(240)
    ct = 0 
    for xs in range(1,5):
        for xp in range(1,13):
            for xf in range(5):         
                spf[ct] = 1000*xs + 10*xp + xf
                ct +=1            
    spf = np.int32(spf)
    spf_unique = np.concatenate(([10,11,12,13,14],spf,[0]))
    spf_list = np.concatenate(([0,0,0,0,0],range(1,241),[0]))
    spf_dict = {spf_unique[k] : spf_list[k] for k in range(len(spf_unique))}
    spf_inv_unique = np.concatenate(([0], spf))
    spf_inv_dict = {k : spf_inv_unique[k] for k in range(len(spf_inv_unique))}

    # トレーニングパラメータとベストモデルの追跡用
    best_accuracy = 0
    best_embedder = None
    best_encoder = None
    best_blstm = None
    best_decoder = None
    best_classifier = None
    best_M2l = None

    #%% Divide data into test and training sets
    def make_data(training_corpus, training_key_list, validation_split=0):

        Xtrain, Ytrain = vfp.split_data(training_corpus, training_key_list)
        Xrest = Xtrain['test']
        Xtrain = Xtrain['train']
        Ytrain = Ytrain['train']
        songlen = Xtrain['length']
            
        #### extra code to convert start and duration to categorical using dictionary
        start_unique = np.array(np.unique(np.concatenate((Xtrain['start'],Xrest['start'])))*256 , dtype='int')
        start_dict = {start_unique[k] : k for k in range(len(start_unique))}
        
        duration_unique = np.array(np.unique(np.concatenate((Xtrain['duration'],Xrest['duration'])))*256 , dtype='int')
        duration_dict = {duration_unique[k] : k for k in range(len(duration_unique))}
        
        spft = (Ytrain['string']+1)*1000 + (Ytrain['position']+1)*10 + Ytrain['finger']
        
        #### Split training and validation data
        # validation indices
        validation_index = list(range(len(spft)-np.int32(np.ceil(len(spft)*validation_split)),len(spft))) #last % for validation 
        print(validation_index)
        training_index = np.setdiff1d(range(len(spft)),validation_index)

        midiinfo = Xtrain['pitch'][validation_index,:] - 54
        
        # training data
        training_data = [Xtrain['pitch'][training_index,:] - 54, #minus 54 because 0 is invalid
                        np.vectorize(start_dict.__getitem__)(np.array(Xtrain['start'][training_index,:]*256, dtype='int')),
                        np.vectorize(duration_dict.__getitem__)(np.array(Xtrain['duration'][training_index,:]*256, dtype='int')),
                        Xtrain['pure_pitch'][training_index,:], 
                        Xtrain['pure_octave'][training_index,:] -3, #G3 is lowest on violin
                        ]
        training_vae_labels = [keras.utils.to_categorical(Xtrain['pitch'][training_index,:] - 54, n_pitch), 
                        keras.utils.to_categorical(np.vectorize(start_dict.__getitem__)(np.array(Xtrain['start'][training_index,:]*256, dtype='int')), n_start),
                        keras.utils.to_categorical(np.vectorize(duration_dict.__getitem__)(np.array(Xtrain['duration'][training_index,:]*256, dtype='int')), n_duration), 
                        keras.utils.to_categorical(Xtrain['pure_pitch'][training_index,:], n_pure_pitch),
                        keras.utils.to_categorical(Xtrain['pure_octave'][training_index,:] - 3, n_pure_octave), 
                        ]
        training_classifier_labels = keras.utils.to_categorical(np.vectorize(spf_dict.__getitem__)(spft[training_index,:]), n_spf)

        
        # validation data
        if validation_split !=0:
            validation_data =  [Xtrain['pitch'][validation_index,:] - 54, #minus 54 because 0 is invalid
                            np.vectorize(start_dict.__getitem__)(np.array(Xtrain['start'][validation_index,:]*256, dtype='int')),
                            np.vectorize(duration_dict.__getitem__)(np.array(Xtrain['duration'][validation_index,:]*256, dtype='int')),
                            Xtrain['pure_pitch'][validation_index,:], 
                            Xtrain['pure_octave'][validation_index,:] -3, #G3 is lowest on violin
                            ]
            validation_vae_labels = [keras.utils.to_categorical(Xtrain['pitch'][validation_index,:] - 54, n_pitch), 
                            keras.utils.to_categorical(np.vectorize(start_dict.__getitem__)(np.array(Xtrain['start'][validation_index,:]*256, dtype='int')), n_start), 
                            keras.utils.to_categorical(np.vectorize(duration_dict.__getitem__)(np.array(Xtrain['duration'][validation_index,:]*256, dtype='int')), n_duration),  
                            keras.utils.to_categorical(Xtrain['pure_pitch'][validation_index,:], n_pure_pitch),
                            keras.utils.to_categorical(Xtrain['pure_octave'][validation_index,:] - 3, n_pure_octave), 
                            ]
            validation_classifier_labels = keras.utils.to_categorical(np.vectorize(spf_dict.__getitem__)(spft[validation_index,:]), n_spf)
        else: validation_data = validation_vae_labels = validation_classifier_labels = None

        class output: None    
        output = output()
        output.training_data = training_data
        output.training_vae_labels = training_vae_labels
        output.training_classifier_labels = training_classifier_labels
        output.validation_data = validation_data
        output.validation_vae_labels = validation_vae_labels
        output.validation_classifier_labels = validation_classifier_labels
        output.start_dict = start_dict
        output.duration_dict = duration_dict
        output.Ytrain = Ytrain
        output.spf = spft
        output.songlen = songlen
        output.midiinfo = midiinfo
        return output


    #default : 14
    size = 14
    #%% For loop
    fix_unlabelled = 1 #!!!!!! use the same 6 unlabelled songs
    #for training_size in  [13,1,2,3,4,5,6,7]: #number of labelled songs to train on  #!!!!!!!
    for training_size in  [13]: #no unlabled training
        for tekl in range(size): #!!!!!! #which song to leave out for testing
                
            print('\nTesting song '+str(tekl)+' on Training size of '+str(training_size))
            pkl = np.setdiff1d(range(0,14),tekl) #index of remaining songs
            
            np.random.seed(tekl)
            python_random.seed(tekl)
            tf.random.set_seed(tekl)
            
            permpkl = np.random.permutation(pkl) #randomise remaining songs
            tkl = permpkl[:training_size] #take 4 songs as labelled data = 28.57% of dataset, 3 = 23.08%

            if fix_unlabelled == 1 and training_size > 0 and training_size < 8: #make sure same unlabelled songs for different size of labelled songs on same testing song
                ukl = np.random.permutation(np.setdiff1d(pkl,permpkl[:7]))[:6] #index of unlabelled songs, randomly take first n songs
            else:
                ukl = np.setdiff1d(pkl,tkl)

                    
            testing_key_list = np.array(list(testing_corpus.keys()))[tekl]
            labelled_key_list = np.array(list(training_corpus.keys()))[tkl]
            unlabelled_key_list = np.array(list(training_corpus.keys()))[ukl]

            print('testing_key_list:')   
            print(testing_key_list)   
            print('labelled_key_list:')   
            print(labelled_key_list)
            print('unlabelled_key_list:')   
            print(unlabelled_key_list)
            

            #%% Input                   
            seq_len = int(training_corpus_name[17:19]) #length of sequence
            l1l2 = regularizers.l1_l2(l1=1e-5, l2=1e-4)
            l2 = regularizers.l2(1e-4)
                
            #%% Embedder        
            # Inputs
            in_pitch = keras.Input(shape=(seq_len,), name='pitch')  
            in_start = keras.Input(shape=(seq_len,), name='start')
            in_duration = keras.Input(shape=(seq_len,), name='duration')
            
            # Embedding
            x1 = layers.Embedding(n_pitch, embedding_size_pitch, mask_zero=False, embeddings_regularizer=l1l2)(in_pitch)
            x2 = layers.Embedding(n_start, embedding_size_start, mask_zero=False, embeddings_regularizer=l1l2)(in_start)
            x3 = layers.Embedding(n_duration, embedding_size_duration, mask_zero=False, embeddings_regularizer=l1l2)(in_duration)
            x = layers.Concatenate()([x1,x2,x3])        
            x = layers.Dense(n_dim1, kernel_regularizer=l1l2, bias_regularizer=l2, activity_regularizer=l2)(x)        
            x = layers.PReLU()(x)
            out_emb = layers.LayerNormalization()(x)         

            # Model
            emb_in = [in_pitch, in_start, in_duration]
            emb_out = [out_emb]
            embedder = keras.Model(emb_in,emb_out, name='embedder')
            
            #%% Encoder                
            # Input
            in_enc = keras.Input(shape=(seq_len, n_dim2, ), name = 'encoder in')

            # Layers
            z = layers.Bidirectional(layers.LSTM(vae_enc_units, return_sequences=True, kernel_regularizer=l1l2, bias_regularizer=l2, activity_regularizer=l2, recurrent_regularizer=l1l2))(in_enc)        
            z_mean = layers.Dense(latent_dim, kernel_regularizer=l1l2, bias_regularizer=l2, activity_regularizer=l2)(z)
            z_log_var = layers.Dense(latent_dim, kernel_regularizer=l1l2, bias_regularizer=l2, activity_regularizer=l2)(z)
            z_mean, z_log_var = KLDivergenceLayer()([z_mean, z_log_var])
            out_enc = Sampling()([z_mean, z_log_var])
            
            # Model
            enc_in = [in_enc]
            enc_out = [out_enc]
            encoder = keras.Model(enc_in,enc_out, name='encoder')
            
            #%% BLSTM               
            # Input
            in_blstm = keras.Input(shape=(seq_len, n_dim2, ), name='blstm in')
            
            # Layers
            x = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=True, kernel_regularizer=l1l2, bias_regularizer=l2, activity_regularizer=l2, recurrent_regularizer=l1l2))(in_blstm)        

            # Model
            blstm_in = [in_blstm]
            blstm_out = [x]
            blstm = keras.Model(blstm_in,blstm_out, name='blstm')

            #%% Classifier                
            # Input
            in_cla = keras.Input(shape=(seq_len, n_blstm, ), name='classifier in')

            #%%% logit output for gumbel softmax
            x = layers.Dense(n_spf)(in_cla)
            out_cla = layers.Activation('softmax')(x)

            # Model
            cla_in = [in_cla]
            cla_out = [out_cla]
            classifier = keras.Model(cla_in,cla_out, name='classifier')   

            #%% Decoder        
            # Inputs
            in_dec_z = keras.Input(shape=(seq_len, latent_dim, ), name = 'decoder z input')
            in_dec_spf = keras.Input(shape=(seq_len, n_spf, ), name = 'decoder spf input')
            
            # Layers
            x = layers.Concatenate()([in_dec_z,in_dec_spf])               
            x = layers.Bidirectional(layers.LSTM(vae_dec_units, return_sequences=True, kernel_regularizer=l1l2, bias_regularizer=l2, activity_regularizer=l2, recurrent_regularizer=l1l2))(x)        
            
            # Outputs
            out_pitch = layers.Dense(n_pitch, activation = 'softmax', name = 'out_pitch')(x)
            out_start = layers.Dense(n_start, activation = 'softmax', name = 'out_start')(x) 
            out_duration = layers.Dense(n_duration, activation = 'softmax', name = 'out_duration')(x) 
            
            # Model
            dec_in = [in_dec_z,in_dec_spf]
            dec_out = [out_pitch, out_start, out_duration]
            decoder = keras.Model(dec_in,dec_out, name="decoder")
            
            #%% Make model for labelled data        
            # Inputs - labelled
            in_lpitch = keras.Input(shape=(seq_len,), name='lpitch')  
            in_lstart = keras.Input(shape=(seq_len,), name='lstart')
            in_lduration = keras.Input(shape=(seq_len,), name='lduration')

            emb_inl = [in_lpitch, in_lstart, in_lduration]      
            
            embedded_l = embedder(emb_inl)
            blstm_l = blstm(embedded_l)
            encoded_l = encoder(embedded_l)
            classified_l = classifier(blstm_l)
            decoded_l = decoder([encoded_l, classified_l])

            M2l = keras.Model(emb_inl, [decoded_l, classified_l], name='labelled')

            #%% Train model      
            checknum = M2l.get_weights()[0][0][0]/M2l.get_weights()[0][0][1]
            print('checknum: '+str(checknum))
                
            cal_earlystopping = keras.callbacks.EarlyStopping(
            monitor=monitor, 
            patience=patience, 
            verbose=2,
            restore_best_weights=restore_best_weights) #early stopping
        
            initial_learning_rate = 0.01
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate,
                decay_steps=350,
                decay_rate=0.96,
                staircase=True)
            opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=0.001)
            
            
            print("\nBegin training...")


            #### Labelled data only
            if len(labelled_key_list) > 0:
                print('only labelled data! - supervised learning')

                La = make_data(training_corpus, labelled_key_list, validation_split) 
                
                print('fitting model...')        
                M2l.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', 'accuracy', 'accuracy', 'accuracy']) 
                validation_data = (
                    La.validation_data[0:3], 
                    [La.validation_vae_labels[0:3], La.validation_classifier_labels]
                )
                
                history = M2l.fit(
                    x = La.training_data[0:3],
                    y = [La.training_vae_labels[0:3], La.training_classifier_labels],
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(La.validation_data[0:3], [La.validation_vae_labels[0:3], La.validation_classifier_labels]), 
                    callbacks=[cal_earlystopping]
                    )
                M2 = M2l     
            

            # バリデーション精度を取得（最後のエポックのval_classifier_accuracy）
            validation_accuracy = history.history['val_classifier_accuracy'][-11]

            # 最も高いバリデーション精度を持つモデルを保存
            if validation_accuracy > best_accuracy:
                best_accuracy = validation_accuracy
                best_embedder = copy.deepcopy(embedder)
                best_blstm = copy.deepcopy(blstm)
                best_encoder = copy.deepcopy(encoder)
                best_decoder = copy.deepcopy(decoder)
                best_classifier = copy.deepcopy(classifier)
                best_M2l = copy.deepcopy(M2l)

            print("End training")
        
    #%% clear and delete
            keras.backend.clear_session()
            print('session cleared')      
            
            del(embedder, encoder, blstm, decoder, classifier, M2l )
            print('embedder,encoder,blstm,decoder,classifier, M2l, deleted')
            try:
                del(M2)
                print('deleted M2')
            except:
                print('no M2u to delete')
            try:
                del(La)
                print('deleted La')
            except:
                print('no La to delete')
            
    #%%Save model        
    model_name = 'bestmodel'

    savemodel = True #True if save is needed
    if savemodel:
        print("\nSaving models...")
        best_embedder.save(model_name + '_emb.h5')
        best_encoder.save(model_name + '_enc.h5')
        best_blstm.save(model_name + '_bls.h5')
        best_decoder.save(model_name + '_dec.h5')
        best_classifier.save(model_name + '_cla.h5')
        #np.save(model_name+'.npy',history.history)
        print("Saved\n")
        exit()

###transfer learning
#dataset prepare
training_corpus_name = 'trainingdataset32n3.pickle'
testing_corpus_name = 'testingdataset32n3.pickle'

training_full = vfp.load_data(training_corpus_name)
training_corpus = {k: v for k, v in training_full.items()}
testing_full = vfp.load_data(testing_corpus_name)
testing_corpus = {k: v for k, v in testing_full.items()}

n_string = 4
n_position = 26
n_finger = 6
n_spf = 1248+1 #4*12*5+1 but 133 in dataset, extra to handle new data
#%% Joint (string,position,finger) dictionary 
tspf = np.zeros(1248)
ct = 0 
for xs in range(1,5):
    for xp in range(1,27):
        for xf in range(6):       
            for xe in range(2):
                if (xf == 5):
                    tspf[ct] = 10000*xs + 100*xp + 90 + xe
                else:
                    tspf[ct] = 10000*xs + 100*xp + 10*xf + xe
                ct +=1            
tspf = np.int32(tspf)
tspf_unique = np.concatenate(([10,11,12,13,14],tspf,[0]))
tspf_list = np.concatenate(([0,0,0,0,0],range(1,1249),[0]))
tspf_dict = {tspf_unique[k] : tspf_list[k] for k in range(len(tspf_unique))}
tspf_inv_unique = np.concatenate(([0], tspf))
tspf_inv_dict = {k : tspf_inv_unique[k] for k in range(len(tspf_inv_unique))}

all_results = []
movement_results = []  # 移動距離と移動回数の結果を保存

#%% Divide data into test and training sets
def t_make_data(training_corpus, training_key_list, validation_split=0):

    Xtrain, Ytrain = vfp.t_split_data(training_corpus, training_key_list)
    Xrest = Xtrain['test']
    Xtrain = Xtrain['train']
    Ytrain = Ytrain['train']
    songlen = Xtrain['length']
        
    #### extra code to convert start and duration to categorical using dictionary
    start_unique = np.array(np.unique(np.concatenate((Xtrain['start'],Xrest['start'])))*256 , dtype='int')
    start_dict = {start_unique[k] : k for k in range(len(start_unique))}
    
    duration_unique = np.array(np.unique(np.concatenate((Xtrain['duration'],Xrest['duration'])))*256 , dtype='int')
    duration_dict = {duration_unique[k] : k for k in range(len(duration_unique))}
    
    spft = (Ytrain['string']+1)*10000 + (Ytrain['position'])*100 + Ytrain['finger']*10 + Ytrain['expansion']
    
    #### Split training and validation data
    # validation indices
    validation_index = list(range(len(spft)-np.int32(np.ceil(len(spft)*validation_split)),len(spft))) #last % for validation 
    print(validation_index)
    training_index = np.setdiff1d(range(len(spft)),validation_index)

    midiinfo = Xtrain['pitch'][training_index,:] - 35
    
    # training data
    training_data = [Xtrain['pitch'][training_index,:] - 35, #minus 54 because 0 is invalid
                      np.vectorize(start_dict.__getitem__)(np.array(Xtrain['start'][training_index,:]*256, dtype='int')),
                      np.vectorize(duration_dict.__getitem__)(np.array(Xtrain['duration'][training_index,:]*256, dtype='int')),
                      ]
    training_vae_labels = [keras.utils.to_categorical(Xtrain['pitch'][training_index,:] - 35, n_pitch), 
                      keras.utils.to_categorical(np.vectorize(start_dict.__getitem__)(np.array(Xtrain['start'][training_index,:]*256, dtype='int')), n_start),
                      keras.utils.to_categorical(np.vectorize(duration_dict.__getitem__)(np.array(Xtrain['duration'][training_index,:]*256, dtype='int')), n_duration), 
                      ]
    print("spft values:", spft[training_index, :])
    print("tspf_dict keys:", list(tspf_dict.keys())[:10])  # 最初の10個を表示

    training_classifier_labels = keras.utils.to_categorical(np.vectorize(tspf_dict.__getitem__)(spft[training_index,:]), n_spf)

    
    # validation data
    if validation_split !=0:
        validation_data =  [Xtrain['pitch'][validation_index,:] - 35, #minus 54 because 0 is invalid
                          np.vectorize(start_dict.__getitem__)(np.array(Xtrain['start'][validation_index,:]*256, dtype='int')),
                          np.vectorize(duration_dict.__getitem__)(np.array(Xtrain['duration'][validation_index,:]*256, dtype='int')),
                          ]
        validation_vae_labels = [keras.utils.to_categorical(Xtrain['pitch'][validation_index,:] - 35, n_pitch), 
                          keras.utils.to_categorical(np.vectorize(start_dict.__getitem__)(np.array(Xtrain['start'][validation_index,:]*256, dtype='int')), n_start), 
                          keras.utils.to_categorical(np.vectorize(duration_dict.__getitem__)(np.array(Xtrain['duration'][validation_index,:]*256, dtype='int')), n_duration),  
                          ]
        validation_classifier_labels = keras.utils.to_categorical(np.vectorize(tspf_dict.__getitem__)(spft[validation_index,:]), n_spf)
    else: validation_data = validation_vae_labels = validation_classifier_labels = None

    class output: None    
    output = output()
    output.training_data = training_data
    output.training_vae_labels = training_vae_labels
    output.training_classifier_labels = training_classifier_labels
    output.validation_data = validation_data
    output.validation_vae_labels = validation_vae_labels
    output.validation_classifier_labels = validation_classifier_labels
    output.start_dict = start_dict
    output.duration_dict = duration_dict
    output.Ytrain = Ytrain
    output.spf = spft
    output.songlen = songlen
    output.midiinfo = midiinfo
    return output

#train model
# default 15
t_size = 18
fix_unlabelled = 1 #!!!!!! use the same 6 unlabelled songs
for training_size in  [17]: #no unlabled training
    for tekl in range(t_size): #!!!!!! #which song to leave out for testing
            
        print('\nTesting song '+str(tekl)+' on Training size of '+str(training_size))
        pkl = np.setdiff1d(range(0,18),tekl) #index of remaining songs
        
        np.random.seed(tekl)
        python_random.seed(tekl)
        tf.random.set_seed(tekl)
        
        permpkl = np.random.permutation(pkl) #randomise remaining songs
        tkl = permpkl[:training_size] #take 4 songs as labelled data = 28.57% of dataset, 3 = 23.08%

        if fix_unlabelled == 1 and training_size > 0 and training_size < 8: #make sure same unlabelled songs for different size of labelled songs on same testing song
            ukl = np.random.permutation(np.setdiff1d(pkl,permpkl[:7]))[:6] #index of unlabelled songs, randomly take first n songs
        else:
            ukl = np.setdiff1d(pkl,tkl)

                
        testing_key_list = np.array(list(testing_corpus.keys()))[tekl]
        labelled_key_list = np.array(list(training_corpus.keys()))[tkl]

        print('testing_key_list:')   
        print(testing_key_list)   
        print('labelled_key_list:')   
        print(labelled_key_list)

        #%% Input                   
        seq_len = 32 #length of sequence
        l1l2 = regularizers.l1_l2(l1=1e-5, l2=1e-4)
        l2 = regularizers.l2(1e-4)
    
        #%% Embedder        
        # Inputs
        in_pitch = keras.Input(shape=(seq_len,), name='pitch')  
        in_start = keras.Input(shape=(seq_len,), name='start')
        in_duration = keras.Input(shape=(seq_len,), name='duration')
        
        # Embedding
        x1 = layers.Embedding(n_pitch, embedding_size_pitch, mask_zero=False, embeddings_regularizer=l1l2)(in_pitch)
        x2 = layers.Embedding(n_start, embedding_size_start, mask_zero=False, embeddings_regularizer=l1l2)(in_start)
        x3 = layers.Embedding(n_duration, embedding_size_duration, mask_zero=False, embeddings_regularizer=l1l2)(in_duration)
        x = layers.Concatenate()([x1,x2,x3])        
        x = layers.Dense(n_dim1, kernel_regularizer=l1l2, bias_regularizer=l2, activity_regularizer=l2)(x)        
        x = layers.PReLU()(x)
        out_emb = layers.LayerNormalization()(x)         

        # Model
        emb_in = [in_pitch, in_start, in_duration]
        emb_out = [out_emb]
        embedder = keras.Model(emb_in,emb_out, name='embedder')
        
        #%% Encoder                
        # Input
        in_enc = keras.Input(shape=(seq_len, n_dim2, ), name = 'encoder in')

        # Layers
        z = layers.Bidirectional(layers.LSTM(vae_enc_units, return_sequences=True, kernel_regularizer=l1l2, bias_regularizer=l2, activity_regularizer=l2, recurrent_regularizer=l1l2))(in_enc)        
        z_mean = layers.Dense(latent_dim, kernel_regularizer=l1l2, bias_regularizer=l2, activity_regularizer=l2)(z)
        z_log_var = layers.Dense(latent_dim, kernel_regularizer=l1l2, bias_regularizer=l2, activity_regularizer=l2)(z)
        z_mean, z_log_var = KLDivergenceLayer()([z_mean, z_log_var])
        out_enc = Sampling()([z_mean, z_log_var])
        
        # Model
        enc_in = [in_enc]
        enc_out = [out_enc]
        encoder = keras.Model(enc_in,enc_out, name='encoder')
        
        #%% BLSTM               
        # Input
        in_blstm = keras.Input(shape=(seq_len, n_dim2, ), name='blstm in')
        
        # Layers
        x = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=True, kernel_regularizer=l1l2, bias_regularizer=l2, activity_regularizer=l2, recurrent_regularizer=l1l2))(in_blstm)        

        # Model
        blstm_in = [in_blstm]
        blstm_out = [x]
        blstm = keras.Model(blstm_in,blstm_out, name='blstm')
        
        #%% Classifier                
        # Input
        in_cla = keras.Input(shape=(seq_len, n_blstm, ), name='classifier in')

        #%%% logit output for gumbel softmax
        x = layers.Dense(n_spf)(in_cla)
        out_cla = layers.Activation('softmax')(x)

        # Model
        cla_in = [in_cla]
        cla_out = [out_cla]
        classifier = keras.Model(cla_in,cla_out, name='classifier')   
        
        
        #%% Decoder        
        # Inputs
        in_dec_z = keras.Input(shape=(seq_len, latent_dim, ), name = 'decoder z input')
        in_dec_spf = keras.Input(shape=(seq_len, n_spf, ), name = 'decoder spf input')
        
        # Layers
        x = layers.Concatenate()([in_dec_z,in_dec_spf])               
        x = layers.Bidirectional(layers.LSTM(vae_dec_units, return_sequences=True, kernel_regularizer=l1l2, bias_regularizer=l2, activity_regularizer=l2, recurrent_regularizer=l1l2))(x)        
        
        # Outputs
        out_pitch = layers.Dense(n_pitch, activation = 'softmax', name = 'out_pitch')(x)
        out_start = layers.Dense(n_start, activation = 'softmax', name = 'out_start')(x) 
        out_duration = layers.Dense(n_duration, activation = 'softmax', name = 'out_duration')(x) 
        
        # Model
        dec_in = [in_dec_z,in_dec_spf]
        dec_out = [out_pitch, out_start, out_duration]
        decoder = keras.Model(dec_in,dec_out, name="decoder")
        
        #%% Make model for labelled data  ※これだと元モデルのloadができていないので要改良      
        # Inputs - labelled
        in_lpitch = keras.Input(shape=(seq_len,), name='lpitch')  
        in_lstart = keras.Input(shape=(seq_len,), name='lstart')
        in_lduration = keras.Input(shape=(seq_len,), name='lduration')

        emb_inl = [in_lpitch, in_lstart, in_lduration]      
        
        embedded_l = embedder(emb_inl)
        blstm_l = blstm(embedded_l)
        encoded_l = encoder(embedded_l)
        classified_l = classifier(blstm_l)
        decoded_l = decoder([encoded_l, classified_l]) #入力が1249次元に対応していない（241次元）ため作り直し

        tM2l = keras.Model(emb_inl, [decoded_l, classified_l], name='labelled')
        #tM2l = keras.Model(emb_inl, classified_l, name='labelled')
        
        #%% Train model
        checknum = tM2l.get_weights()[0][0][0]/tM2l.get_weights()[0][0][1]
        print('checknum: '+str(checknum))

        cal_earlystopping = keras.callbacks.EarlyStopping(
        monitor=monitor, 
        patience=patience, 
        verbose=2,
        restore_best_weights=restore_best_weights) #early stopping

        initial_learning_rate = 0.01
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=350,
            decay_rate=0.96,
            staircase=True)
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=0.001)

        print("\nBegin training...")
        

        #### Labelled data only
        if len(labelled_key_list) > 0:
            print('only labelled data! - supervised learning')

            La = t_make_data(training_corpus, labelled_key_list, validation_split) 
            
            print('fitting model...')

            tM2l.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', 'accuracy', 'accuracy', 'accuracy'])
            validation_data = (
                La.validation_data[0:3], 
                [La.validation_vae_labels[0:3], La.validation_classifier_labels]
                #La.validation_classifier_labels
            )

            history = tM2l.fit(
                x = La.training_data[0:3],
                y = [La.training_vae_labels[0:3], La.training_classifier_labels],
                #y = La.training_classifier_labels,
                batch_size=batch_size,
                epochs=epochs,
                verbose=2,
                validation_data=(La.validation_data[0:3], [La.validation_vae_labels[0:3], La.validation_classifier_labels]), 
                #validation_data=(La.validation_data[0:3], La.validation_classifier_labels), 
                callbacks=[cal_earlystopping]
                )
            tM2 = tM2l

        print("End training")
        #
        Ts = t_make_data(testing_corpus, testing_key_list, validation_split =0)

        # Embederを使用してデータを変換
        embedded_data = embedder.predict(Ts.training_data[0:3])  # embedder

        #BLSTM
        in_cla_result = blstm.predict(embedded_data)

        # エンコードされたデータで分類器の予測を実行
        out_cla_result = classifier.predict(in_cla_result)

        # 予測結果とラベルの比較（argmaxでラベルを取得）
        y_pred = np.argmax(out_cla_result, axis=-1)
        y_true = np.argmax(Ts.training_classifier_labels, axis=-1)

        # validation_dataの pitch データを取得して、予測結果と比較
        midi = Ts.midiinfo
        array = midi.flatten()
        validation_pitch = np.where(array == 0, array + 10, array)

        print(y_true)
        print(validation_pitch)

        # データセット用のMIDI変換関数
        string_midi_base = {1: 22, 2: 15, 3: 8, 4: 1} # -54されている
        position_midi_offsets = {1: 1, 2: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 10: 8, 11: 9, 12: 10, 13: 11, 14: 12, 15: 13, 16: 14, 18: 15, 19: 16, 20: 17, 21: 18, 22: 19, 24: 20, 25: 21}
        finger_midi_offsets = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 9: -1}
        expansion_midi_offsets = {0: 0, 1: 1}

        # MIDIノートを生成する関数
        def generate_midi_from_spf(string, position, finger, expansion):
            #(string, position, finger)からMIDIノートを生成する
            base_midi = string_midi_base.get(string, 0)  # 弦に基づくMIDIベース値
            pos_offset = position_midi_offsets.get(position, 0)  # ポジションに基づくオフセット
            finger_offset = finger_midi_offsets.get(finger, 0)  # 指に基づくオフセット
            expansion_offset = expansion_midi_offsets.get(expansion, 0)  # 指に基づくオフセット
            # fingerが0の場合はbase_midiを直接返し、それ以外の場合はオフセットを加える
                # 親指 (finger == 9) の場合、特定の例外処理を適用
            if finger == 9:
                # 親指の場合、0 から -6 の範囲で可能なMIDIノートを返す
                midi_values = [base_midi + pos_offset + offset + expansion_offset for offset in range(-6, 1)]
                return midi_values  # リストで返す
            elif finger == 0:
                return [base_midi]
            else:
                # 通常の指の場合、単一のMIDI値を返す
                return [base_midi + pos_offset + finger_offset + expansion_offset]
        
        def get_spf_combination(index):
            #240個のspf配列から (string, position, finger) を取得する
            value = tspf[index-1]  # インデックスから整数を取得
            string = value // 10000  # 弦 (string)
            position = (value % 10000) // 100  # ポジション (position)
            finger = (value % 100) // 10  # 指 (finger)
            expansion = value % 10
            return string, position, finger, expansion

        # 予測と正解の (string, position, finger) を取得
        y_pred_spf = [get_spf_combination(idx) for idx in y_pred.flatten()]
        y_true_spf = [get_spf_combination(idx) for idx in y_true.flatten()]

        # 各 (string, position, finger) の組み合わせに基づくMIDIノートベクトルを生成
        y_pred_midi = [generate_midi_from_spf(*tspf) for tspf in y_pred_spf]
        y_true_midi = [generate_midi_from_spf(*tspf) for tspf in y_true_spf]

        print(y_true_spf)
        print(y_true_midi)
        
        # MIDIノートベクトルをnumpy配列に変換
        #y_pred_midi_array = np.array(y_pred_midi)
        #y_true_midi_array = np.array(y_true_midi)
        """
        def calculate_accuracy_with_tolerance(pred_midi, true_midi):
            correct_predictions = np.sum(np.abs(pred_midi - true_midi) <= 0)
            total_predictions = len(true_midi)
            accuracy = correct_predictions / total_predictions
            return accuracy
        """
        def calculate_accuracy_with_tolerance(pred_midi, true_midi):
            """
            誤差許容範囲を考慮した精度を計算。
            
            - pred_midi: 各予測に対して許容されるMIDI値のリストのリスト
            - true_midi: 真のMIDI値の配列
            """
            correct_predictions = 0

            for pred, true in zip(pred_midi, true_midi):
                if true in pred:  # 真のMIDI値が許容される予測値リスト内にある場合
                    correct_predictions += 1

            total_predictions = len(true_midi)
            accuracy = correct_predictions / total_predictions
            return accuracy
        
        # 各許容範囲での精度を計算
        accuracy_exact = calculate_accuracy_with_tolerance(y_pred_midi, validation_pitch) #array削除
        accuracy_exact_true = calculate_accuracy_with_tolerance(y_true_midi, validation_pitch)

        # 結果の表示
        print(f"Validation Pitch Accuracy (Exact Match): {accuracy_exact * 100:.2f}%")
        print(f"Label Pitch Accuracy (Exact Match): {accuracy_exact * 100:.2f}%")

        all_results.append(
            {
                "test_case": tekl,  # テストケースの名前
                "accuracy_exact": accuracy_exact,  # MIDI精度 (Pred)
                "accuracy_exact_true": accuracy_exact_true,  # MIDI精度 (True)
            }
        )


        #pitch_differences = y_true_midi_array - validation_pitch
        # 各MIDIリストから最適な値を選択（validation_pitch に最も近い値）
        optimized_y_true_midi = [
            min(midi_values, key=lambda x: abs(x - val_pitch))
            for midi_values, val_pitch in zip(y_true_midi, validation_pitch)
        ]

        # Pitch differences を計算
        pitch_differences = np.array(optimized_y_true_midi) - np.array(validation_pitch)

        # Plot the histogram of pitch differences
        plt.figure(figsize=(10, 6))
        plt.hist(pitch_differences, bins=20, edgecolor='black')
        plt.xlabel("Difference between Label and Validation Pitch (MIDI)")
        plt.ylabel("Frequency")
        plt.title("Distribution of Pitch Differences")

        # Save the plot as a PNG file for easy viewing in Visual Studio
        output_path = f"review/scratch_true({str(tekl)}).png"
        plt.savefig(output_path)

        #pitch_differences = y_pred_midi_array - validation_pitch
        # 各MIDIリストから最適な値を選択（validation_pitch に最も近い値）
        optimized_y_pred_midi = [
            min(midi_values, key=lambda x: abs(x - val_pitch))
            for midi_values, val_pitch in zip(y_pred_midi, validation_pitch)
        ]

        # Pitch differences を計算
        pitch_differences = np.array(optimized_y_pred_midi) - np.array(validation_pitch)

        # Plot the histogram of pitch differences
        plt.figure(figsize=(10, 6))
        plt.hist(pitch_differences, bins=20, edgecolor='black')
        plt.xlabel("Difference between Predicted and Validation Pitch (MIDI)")
        plt.ylabel("Frequency")
        plt.title("Distribution of Pitch Differences")

        # Save the plot as a PNG file for easy viewing in Visual Studio
        output_path = f"review/scratch_pred({str(tekl)}).png"
        plt.savefig(output_path)

        # 差の絶対値が5以上のインデックスを取得
        large_difference_indices = np.where(np.abs(pitch_differences) >= 1)[0]

        # Attempting to save the large pitch differences as a text file without using pandas

        # Construct the output content with relevant information
        output_content = "Index, Predicted_MIDI, Validation_Pitch, Difference\n"
        #for idx in large_difference_indices:
        #    output_content += f"{idx}, {y_pred_midi_array[idx]}, {validation_pitch[idx]}, {pitch_differences[idx]}\n"

        # Save the content to a .txt file for easy viewing in Visual Studio Code
        output_path = f"review/scratch_txt({str(tekl)}).txt"
        with open(output_path, "w") as file:
            file.write(output_content)

        #distance
        y_pred_position = [x[1] for x in y_pred_spf]
        y_true_position = [x[1] for x in y_true_spf]

        def calculate_total_movement_distance_and_count(positions):
            
            #各ポジション間の手の移動距離と移動回数を計算します。
            
            #Parameters:
            #positions (list of int): 曲全体のポジションのシーケンス。
            
            #Returns:
            #tuple: (総移動距離, 移動回数) のタプル。
            
            # 各ポジションでの弦の長さを計算するための辞書
            n_values = [1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25]
            position_lengths = {n: 33 * (1 / 1.059) ** n for n in n_values}

            # 移動距離と移動回数を初期化
            total_movement_distance = 0
            movement_count = 0
            previous_position_length = None

            for position in positions:
                # 現在のポジションの弦の長さを取得
                if position in position_lengths:
                    current_position_length = position_lengths[position]
                    
                    # 初回以降に移動距離と移動回数をカウント
                    if previous_position_length is not None:
                        movement_distance = abs(previous_position_length - current_position_length)
                        if movement_distance > 0:  # 移動があった場合のみカウント
                            total_movement_distance += movement_distance
                            movement_count += 1
                    
                    # 現在のポジション長を次の比較用に更新
                    previous_position_length = current_position_length

            return total_movement_distance, movement_count
        
        def calculate_movement_per_bar_with_transition(start, positions):
            """
            小節ごとに移動距離と回数を計算し、区切り位置間の移動を次の小節に含める。

            Parameters:
            - start: ndarray, 小節の区切りを表すタイムスタンプ。
            - positions: ndarray, 各タイムステップでのポジション。

            Returns:
            - bar_distances: 各小節の総移動距離のリスト。
            - bar_counts: 各小節の移動回数のリスト。
            """
            # 小節の境界を検出
            bar_boundaries = np.where(np.diff(start) < 0)[0] + 1
            bar_boundaries = np.concatenate(([0], bar_boundaries, [len(start)]))  # 境界に先頭と末尾を追加

            bar_distances = []
            bar_counts = []

            # 各小節について計算
            for i in range(len(bar_boundaries) - 1):
                start_idx = bar_boundaries[i]
                end_idx = bar_boundaries[i + 1]

                # 次の小節への移動を考慮
                if end_idx < len(positions):
                    end_idx += 1  # 次の小節開始位置の移動を含める

                bar_positions = positions[start_idx:end_idx]

                # 移動距離と回数を計算
                total_distance, movement_count = calculate_total_movement_distance_and_count(bar_positions)
                bar_distances.append(total_distance)
                bar_counts.append(movement_count)

            return bar_distances, bar_counts


        
        pred_distance, pred_movements = calculate_total_movement_distance_and_count(y_pred_position)
        print(f"Total hand movement distance(pred): {pred_distance}")
        print(f"Total hand movement count: {pred_movements}")
        true_distance, true_movements = calculate_total_movement_distance_and_count(y_true_position)
        print(f"Total hand movement distance(true): {true_distance}")
        print(f"Total hand movement count: {true_movements}")

        movement_results.append(
            {
                "test_case": tekl,  # テストケースの名前
                "distance_true": true_distance,  # 総移動距離（True）
                "distance_pred": pred_distance,  # 総移動距離（Pred）
                "count_true": true_movements,  # 総移動回数（True）
                "count_pred": pred_movements,  # 総移動回数（Pred）
            }
        )


        # 小節ごとの計算
        true_bar_distances, true_bar_counts = calculate_movement_per_bar_with_transition(validation_pitch, y_true_position)
        pred_bar_distances, pred_bar_counts = calculate_movement_per_bar_with_transition(validation_pitch, y_pred_position)

        # 移動距離の折れ線グラフ
        plt.figure(figsize=(12, 6))
        plt.plot(true_bar_distances, label="True Distance", marker='o')
        plt.plot(pred_bar_distances, label="Predicted Distance", marker='o')
        plt.title("Hand Movement Distance per Bar")
        plt.xlabel("Bar Number")
        plt.ylabel("Distance")
        plt.legend()
        plt.grid(True)
        output_path = f"review/scratch_dist({str(tekl)}).png"
        plt.savefig(output_path)

        # 移動回数の折れ線グラフ
        plt.figure(figsize=(12, 6))
        plt.plot(true_bar_counts, label="True Movements", marker='o')
        plt.plot(pred_bar_counts, label="Predicted Movements", marker='o')
        plt.title("Hand Movement Count per Bar")
        plt.xlabel("Bar Number")
        plt.ylabel("Movement Count")
        plt.legend()
        output_path = f"review/scratch_ct({str(tekl)}).png"
        plt.savefig(output_path)

    
#%% clear and delete
        keras.backend.clear_session()
        print('session cleared')      
        
        del(classifier, decoder, tM2l )
        print('embedder,encoder,blstm,decoder,classifier, M2l, deleted')
        try:
            del(tM2)
            print('deleted M2')
        except:
            print('no M2u to delete')
        try:
            del(La)
            print('deleted La')
        except:
            print('no La to delete')

#評価の統計
# 精度データを抽出
accuracy_pred = [result["accuracy_exact"] for result in all_results]

# 箱ひげ図をプロット
plt.figure(figsize=(10, 6))
plt.boxplot(
    accuracy_pred,
    patch_artist=True,
    boxprops=dict(facecolor="lightblue", color="blue"),
    medianprops=dict(color="red"),
    vert=True,  # 縦方向に描画
    labels=["Predicted Accuracy"]  # ラベル
)
plt.title("Boxplot of Predicted Accuracy")
plt.ylabel("Accuracy")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("review/scratch_accuracy_boxplot.png")

# グラフ生成用データの抽出
test_cases = [result["test_case"] for result in all_results]
accuracy_pred = [result["accuracy_exact"] for result in all_results]
accuracy_true = [result["accuracy_exact_true"] for result in all_results]

bar_width = 0.35
index = np.arange(len(test_cases))

plt.figure(figsize=(12, 6))
plt.bar(index, accuracy_true, bar_width, label='True Accuracy', color='blue')
plt.bar(index + bar_width, accuracy_pred, bar_width, label='Predicted Accuracy', color='orange')

plt.xlabel('Test Case')
plt.ylabel('Accuracy')
plt.title('MIDI Accuracy Comparison')
plt.xticks(index + bar_width / 2, test_cases, rotation=45, ha='right')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("review/scratch_midi_accuracy_bar.png")

# データ抽出
test_cases = [result["test_case"] for result in movement_results]
distance_true = [result["distance_true"] for result in movement_results]
distance_pred = [result["distance_pred"] for result in movement_results]
count_true = [result["count_true"] for result in movement_results]
count_pred = [result["count_pred"] for result in movement_results]

# 棒グラフの設定
bar_width = 0.35
index = np.arange(len(test_cases))

# 1. 移動距離の棒グラフ
plt.figure(figsize=(12, 6))
plt.bar(index, distance_true, bar_width, label='True Distance', color='blue')
plt.bar(index + bar_width, distance_pred, bar_width, label='Predicted Distance', color='orange')

plt.xlabel('Test Case')
plt.ylabel('Total Movement Distance')
plt.title('Total Movement Distance Comparison')
plt.xticks(index + bar_width / 2, test_cases, rotation=45, ha='right')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("review/scratch_movement_distance_bar.png")

# 2. 移動回数の棒グラフ
plt.figure(figsize=(12, 6))
plt.bar(index, count_true, bar_width, label='True Movement Count', color='blue')
plt.bar(index + bar_width, count_pred, bar_width, label='Predicted Movement Count', color='orange')

plt.xlabel('Test Case')
plt.ylabel('Total Movement Count')
plt.title('Total Movement Count Comparison')
plt.xticks(index + bar_width / 2, test_cases, rotation=45, ha='right')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("review/scratch_movement_count_bar.png")


# 1. 移動距離の箱ひげ図
plt.figure(figsize=(10, 6))
plt.boxplot(
    [distance_true, distance_pred],
    sym='',
    labels=["True Distance", "Predicted Distance"],
    patch_artist=True,
    boxprops=dict(facecolor="lightblue", color="blue"),
    medianprops=dict(color="red"),
)
plt.title("Boxplot of Total Movement Distance")
plt.ylabel("Distance")
plt.grid(True)
plt.tight_layout()
plt.savefig("review/scratch_movement_distance_boxplot.png")

# 2. 移動回数の箱ひげ図
plt.figure(figsize=(10, 6))
plt.boxplot(
    [count_true, count_pred],
    sym='',
    labels=["True Movement Count", "Predicted Movement Count"],
    patch_artist=True,
    boxprops=dict(facecolor="lightgreen", color="green"),
    medianprops=dict(color="red"),
)
plt.title("Boxplot of Total Movement Count")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.savefig("review/scratch_movement_count_boxplot.png")

# サンプルデータ
accuracy_pred = [result["accuracy_exact"] for result in all_results]  # MIDI一致率

# 距離の比率 (distance_pred / distance_true) を計算
distance_ratios = [
    result["distance_pred"] / result["distance_true"]
    for result in movement_results
    if result["distance_true"] != 0  # 0 の場合を除外
]

# 回数の比率 (count_pred / count_true) を計算
count_ratios = [
    result["count_pred"] / result["count_true"]
    for result in movement_results
    if result["count_true"] != 0  # 0 の場合を除外
]

# 統計量を計算する関数
def calculate_statistics(data):
    """
    データの統計量を計算
    """
    return {
        "mean": np.mean(data),
        "median": np.median(data),
        "q1": np.percentile(data, 25),
        "q3": np.percentile(data, 75),
        "max": np.max(data),
        "min": np.min(data)
    }

# 統計量の計算
accuracy_stats = calculate_statistics(accuracy_pred)
distance_ratio_stats = calculate_statistics(distance_ratios)
count_ratio_stats = calculate_statistics(count_ratios)

# 結果をテキストファイルに保存
output_path = "review/scratch_statistics_summary.txt"
with open(output_path, "w") as f:
    f.write("=== MIDI 一致率の統計量 ===\n")
    f.write(f"平均値: {accuracy_stats['mean']:.4f}\n")
    f.write(f"中央値: {accuracy_stats['median']:.4f}\n")
    f.write(f"第1四分位数: {accuracy_stats['q1']:.4f}\n")
    f.write(f"第3四分位数: {accuracy_stats['q3']:.4f}\n")
    f.write(f"最大値: {accuracy_stats['max']:.4f}\n")
    f.write(f"最小値: {accuracy_stats['min']:.4f}\n")
    f.write("\n")

    f.write("=== 距離の比率 (Pred / True) の統計量 ===\n")
    f.write(f"平均値: {distance_ratio_stats['mean']:.4f}\n")
    f.write(f"中央値: {distance_ratio_stats['median']:.4f}\n")
    f.write(f"第1四分位数: {distance_ratio_stats['q1']:.4f}\n")
    f.write(f"第3四分位数: {distance_ratio_stats['q3']:.4f}\n")
    f.write(f"最大値: {distance_ratio_stats['max']:.4f}\n")
    f.write(f"最小値: {distance_ratio_stats['min']:.4f}\n")
    f.write("\n")

    f.write("=== 回数の比率 (Pred / True) の統計量 ===\n")
    f.write(f"平均値: {count_ratio_stats['mean']:.4f}\n")
    f.write(f"中央値: {count_ratio_stats['median']:.4f}\n")
    f.write(f"第1四分位数: {count_ratio_stats['q1']:.4f}\n")
    f.write(f"第3四分位数: {count_ratio_stats['q3']:.4f}\n")
    f.write(f"最大値: {count_ratio_stats['max']:.4f}\n")
    f.write(f"最小値: {count_ratio_stats['min']:.4f}\n")

print(f"統計情報を {output_path} に保存しました。")