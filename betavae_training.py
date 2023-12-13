import os
import pandas
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from betavae_model import BVAE 
from visualization import plot_label_clusters_3d

#import dataset
df = pandas.read_csv('ML_MED_Dataset_train_preprocessed_scale_complete_label.csv')
X_train = df.iloc[:,1:46]
print(X_train.head())
X_train = X_train.to_numpy()
y_train = df.iloc[:,46:47]
print(y_train.head())
y_train = y_train.to_numpy() 
print(X_train.shape)
    
#set early stop     
early_stop = keras.callbacks.EarlyStopping(
    monitor="loss",
    min_delta=0.03,
    patience=10,
    verbose=0,
    mode="min",
    baseline=None,
    restore_best_weights=True,
    start_from_epoch=15
)

#create model and assign hyperparameters
bvae = BVAE(  input_dim=45,
            output_dim=45,
            latent_dim=3,
            encoder_activation_function="tanh",
            encoder_output_activation_function="tanh",
            decoder_activation_function="tanh",
            decoder_reconstruction_activation_function="tanh",
            decoder_output_activation_function="tanh",
            decoder_deconstruction_layers_num=5,
            beta=21)

bvae.compile(optimizer=keras.optimizers.SGD(learning_rate=0.001, momentum=0.1) )
bvae.fit(X_train, epochs=120, batch_size=32, callbacks=None )

#classes clusters visualization
plot_label_clusters_3d(bvae, X_train, y_train)


