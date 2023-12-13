import os
import pandas
import numpy as np
import tensorflow as tf
import keras
import keras_tuner
from betavae_model import BVAE 

#import dataset
df = pandas.read_csv('ML_MED_Dataset_train_preprocessed_scale_complete_label.csv')
X_train = df.iloc[:,1:46]
print(X_train.head())
X_train = X_train.to_numpy()
y_train = df.iloc[:,46:47]
print(y_train.head())
y_train = y_train.to_numpy() 
print(X_train.shape)

#hyperparameter tuning 
def build_model(hp):
    input_dim=45
    output_dim=45
    latent_dim=hp.Int("latent_dim", min_value=2, max_value=8, step=1)
    encoder_activation_function=hp.Choice("encoder_activation_function",["relu","tanh","sigmoid"])
    encoder_output_activation_function=hp.Choice("encoder_output_activation_function",["relu","tanh","linear","sigmoid"])
    decoder_activation_function=hp.Choice("decoder_activation_function",["relu","tanh","sigmoid"])
    decoder_reconstruction_activation_function=hp.Choice("decoder_reconstruction_activation_function",["relu","tanh","sigmoid"])
    decoder_output_activation_function=hp.Choice("decoder_output_activation_function",["relu","tanh","sigmoid","linear"])
    decoder_deconstruction_layers_num=hp.Int("decoder_deconstruction_layers_num",min_value=1, max_value=30, step=1)
    learning_rate = hp.Choice("learning_rate", [0.001,0.01,0.1,0.2] )
    momentum = hp.Choice("momentum", [0.001,0.01,0.1,0.2] )
    optimizer = hp.Choice("optimizer",["adam","sgd"])
    beta = hp.Int("beta",min_value=1 ,max_value=30, step=1)

    model = BVAE(  
            input_dim=input_dim,
            output_dim=output_dim,
            latent_dim=latent_dim,
            encoder_activation_function=encoder_activation_function,
            encoder_output_activation_function=encoder_output_activation_function,
            decoder_activation_function=decoder_activation_function,
            decoder_reconstruction_activation_function=decoder_reconstruction_activation_function,
            decoder_output_activation_function=decoder_output_activation_function,
            decoder_deconstruction_layers_num=decoder_deconstruction_layers_num,
            beta=beta
    )
    if (optimizer=="sgd"):
        model.compile(optimizer=keras.optimizers.SGD( learning_rate=learning_rate, momentum=momentum), loss = keras.losses.MeanSquaredError())
    if (optimizer=="adam") :
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss = keras.losses.MeanSquaredError())
    return model

tuner = keras_tuner.GridSearch(
    hypermodel=build_model,
    objective="loss",
    executions_per_trial=2,
    overwrite=True,
    directory="Vae_search",
    project_name="Vae_augmentation",
)

tuner.search(X_train, epochs=50)

# Get the top  hyperparameters.
best_hps = tuner.get_best_hyperparameters(10)
# Build the model with the best hp.
print(best_hps[0])
model = build_model(best_hps[0])
# Save the model.
model.save("tuned_noweight_model_bvae.keras")
