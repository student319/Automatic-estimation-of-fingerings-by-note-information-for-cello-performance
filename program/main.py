# -*- coding: utf-8 -*-
"""
This code references the implementation code for proposed model in "Semi-supervised Violin Finger Generation Using Variational Autoencoders" by Vincent K.M. Cheung, Hsuan-Kai Kao, and Li Su in Proc. of the 22nd Int. Society for Music Information Retrieval Conf., Online, 2021.

"""


import dataset_prepare as vfp
from custom_layers import Sampling, KLDivergenceLayer

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import random as python_random
from tensorflow.keras import regularizers
from tensorflow.keras.utils import custom_object_scope
import copy

#GPU Settings
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

pretrain_needed = 1

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

    # Tracking the parameters to save the best model
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
            

            # Checking the validation accuracy for best epoch
            validation_accuracy = history.history['val_classifier_accuracy'][-11]

            # Save the best model
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
n_spf = 1248+1

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

#load best model
with custom_object_scope({'KLDivergenceLayer': KLDivergenceLayer, 'Sampling': Sampling}):
    best_encoder = keras.models.load_model("bestmodel_enc.h5")
best_blstm = keras.models.load_model("bestmodel_bls.h5")
best_embedder = keras.models.load_model("bestmodel_emb.h5")


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
    start_list = np.array([start_dict.get(int(val * 256), -1) for val in Xtrain['start'].flatten()])
    start_list = start_list.reshape(Xtrain['start'].shape)
    
    # training data
    training_data = [Xtrain['pitch'][training_index,:] - 35, #minus 54 because 0 is invalid
                      np.vectorize(start_dict.__getitem__)(np.array(Xtrain['start'][training_index,:]*256, dtype='int')),
                      np.vectorize(duration_dict.__getitem__)(np.array(Xtrain['duration'][training_index,:]*256, dtype='int')),
                      ]
    training_vae_labels = [keras.utils.to_categorical(Xtrain['pitch'][training_index,:] - 35, n_pitch), 
                      keras.utils.to_categorical(np.vectorize(start_dict.__getitem__)(np.array(Xtrain['start'][training_index,:]*256, dtype='int')), n_start),
                      keras.utils.to_categorical(np.vectorize(duration_dict.__getitem__)(np.array(Xtrain['duration'][training_index,:]*256, dtype='int')), n_duration), 
                      ]
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
    output.start = start_list
    return output

#train model
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
        """    
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
        """
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
        
        #freeze the previous training layers
        for layer in best_embedder.layers:
            layer.trainable = False
        for layer in best_blstm.layers:
            layer.trainable = False
        for layer in best_encoder.layers:
            layer.trainable = False

        #%% Make model for labelled data    
        # Inputs - labelled
        in_lpitch = keras.Input(shape=(seq_len,), name='lpitch')  
        in_lstart = keras.Input(shape=(seq_len,), name='lstart')
        in_lduration = keras.Input(shape=(seq_len,), name='lduration')

        emb_inl = [in_lpitch, in_lstart, in_lduration]      
        
        embedded_l = best_embedder(emb_inl)
        blstm_l = best_blstm(embedded_l)
        encoded_l = best_encoder(embedded_l)
        classified_l = classifier(blstm_l)
        decoded_l = decoder([encoded_l, classified_l])

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

        #%%Save model              
        model_name = 'TransferLearningModel'+str(tekl)
        
        if savemodel:
            print("\nSaving models...")
            embedder.save(model_name + '_emb.h5')
            encoder.save(model_name + '_enc.h5')
            decoder.save(model_name + '_dec.h5')
            classifier.save(model_name + '_cla.h5')
            #np.save(model_name+'.npy',history.history)
            print("Saved\n")

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