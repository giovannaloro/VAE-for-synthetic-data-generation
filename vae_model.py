import numpy as np
import tensorflow as tf
import keras
from keras import layers


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    

class Encoder(layers.Layer):
    "Encode block"

    def __init__(self,input_dim,latent_dim,activation_function,output_activation_function):
        super().__init__()
        self.input_layer = layers.Dense(input_dim, activation=activation_function)
        self.compression_layers = []
        self.mean_layer = layers.Dense(latent_dim, activation=output_activation_function)
        self.log_var_layer = layers.Dense(latent_dim, activation=output_activation_function)
        self.sample_layer = Sampling()
        #Build compression layers dynamically
        for i in range(input_dim -1, latent_dim, -1):
            self.compression_layers.append(layers.Dense(i, activation=activation_function))

    def call(self,inputs):
        x = self.input_layer(inputs)
        for layer in self.compression_layers:
            x = layer(x)
        z_mean = self.mean_layer(x)
        z_log_var = self.log_var_layer(x)
        z = self.sample_layer([z_mean, z_log_var])
        return z_mean, z_log_var, z
    
    def get_config(self):
        return {"input_layer": self.input_layer, 
                "compression_layers": self.compression_layers,
                "mean_layer": self.mean_layer,
                "log_var_layer": self.log_var_layer,
                "sample_layer": self.sample_layer
        }


class Decoder(layers.Layer):
    "Decode block"

    def __init__(self,output_dim,latent_dim,activation_function,reconstruction_activation_function,output_activation_function,reconstruction_layers_num):
        super().__init__()
        self.input_layer = layers.Dense(latent_dim, activation=activation_function)
        self.decompression_layers = []
        self.reconstruction_layers = []
        self.output_layer = layers.Dense(output_dim, activation=output_activation_function) 
        #Build decompression layers dynamically
        for i in range(latent_dim , output_dim +1):
            self.decompression_layers.append(layers.Dense(i, activation=activation_function))
        #Build reconstruction layers dunamically
        for i in range(reconstruction_layers_num):
            self.reconstruction_layers.append(layers.Dense(output_dim, activation=reconstruction_activation_function))

    def call(self,inputs):
        x = self.input_layer(inputs)
        for layer in self.decompression_layers:
            x = layer(x)
        for layer in self.reconstruction_layers:
            x = layer(x)
        output = self.output_layer(x)
        return output
    
    def get_config(self):
        return {"input_layer": self.input_layer, 
                "decompression_layers": self.decompression_layers,
                "reconstruction_layers": self.reconstruction_layers,
                "output_layer": self.output_layer,
        }


#vae model definition
class VAE(keras.Model):
    def __init__(
            self,
            input_dim,
            output_dim,
            latent_dim,
            encoder_activation_function,
            encoder_output_activation_function,
            decoder_activation_function,
            decoder_reconstruction_activation_function,
            decoder_output_activation_function,
            decoder_deconstruction_layers_num,
              **kwargs):
        super().__init__(**kwargs)
        self.encoder = Encoder(input_dim=input_dim,latent_dim=latent_dim,activation_function=encoder_activation_function,output_activation_function=encoder_output_activation_function)
        self.decoder = Decoder(output_dim=output_dim,latent_dim=latent_dim,activation_function=decoder_activation_function,reconstruction_activation_function=decoder_reconstruction_activation_function,output_activation_function=decoder_output_activation_function,reconstruction_layers_num=decoder_deconstruction_layers_num)
  
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        #Add KL divergence regularization loss.
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        self.add_loss(kl_loss)
        return reconstructed
    
    def encode(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return z_mean, z_log_var, z

    def decode(self, inputs):
        decoded = self.decoder(inputs)
        return decoded
    
    def get_config(self):
        return {"encoder": self.encoder, 
                "decoder": self.decoder}