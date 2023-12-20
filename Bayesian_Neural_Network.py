import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.layers import ReLU, Dropout
from tensorflow.keras.optimizers import SGD
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

from sklearn.metrics import mean_squared_error

#Initialization of neural network according to Xavier initialization
def init(shape):
    return tf.random.truncated_normal(
        shape, 
        mean=0.0,
        stddev=np.sqrt(2/sum(shape)))

class BayesianDenseLayer(tf.keras.Model):
    def __init__(self, input_data, output_data, name=None):
        
        super(BayesianDenseLayer, self).__init__(name=name)
        self.input_data = input_data
        self.output_data = output_data
        
        self.weight_loc = tf.Variable(init([input_data, output_data]), name='weight_loc')
        self.weight_std = tf.Variable(init([input_data, output_data]) -6.0, name='weight_std')
        self.bias_loc = tf.Variable(init([1, output_data]), name='bias_loc')
        self.bias_std = tf.Variable(init([1, output_data])-6.0, name='bias_std')
    
    
    def call(self, x, sampling=True):
        """Perform the forward pass"""
        
        if sampling:
        
            # Flipout-estimated weight samples
            s = tfp.random.rademacher(tf.shape(x))
            r = tfp.random.rademacher([x.shape[0], self.output_data])
            w_samples = tf.nn.softplus(self.weight_std)*tf.random.normal([self.input_data, self.output_data])
            w_perturbations = r*tf.matmul(x*s, w_samples)
            w_outputs = tf.matmul(x, self.weight_loc) + w_perturbations
            
            # Flipout-estimated bias samples
            r = tfp.random.rademacher([x.shape[0], self.output_data])
            b_samples = tf.nn.softplus(self.bias_std)*tf.random.normal([self.output_data])
            b_outputs = self.bias_loc + r*b_samples
            
            return w_outputs + b_outputs
        
        else:
            return x @ self.weight_loc + self.bias_loc
    
    @property
    def losses(self):
        """Sum of the KL divergences between priors + posteriors"""
        weight = tfd.Normal(self.weight_loc, tf.nn.softplus(self.weight_std))
        bias = tfd.Normal(self.bias_loc, tf.nn.softplus(self.bias_std))
        prior = tfd.Normal(0, 1)
        return (tf.reduce_sum(tfd.kl_divergence(weight, prior)) +
                tf.reduce_sum(tfd.kl_divergence(bias, prior)))
    
class BayesianDenseNetwork(tf.keras.Model):
    
    def __init__(self, dims, name=None):
        
        super(BayesianDenseNetwork, self).__init__(name=name)
        
        self.steps = []
        self.acts = []
        for i in range(len(dims)-1):
            self.steps += [BayesianDenseLayer(dims[i], dims[i+1])]
            self.acts += [tf.nn.relu]
            
        self.acts[-1] = lambda x: x
        
    
    def call(self, x, sampling=True):

        for i in range(len(self.steps)):
            x = self.steps[i](x, sampling=sampling)
            x = self.acts[i](x)
            
        return x
    
    @property
    def losses(self):
        """Sum of the KL divergences between priors + posteriors"""
        return tf.reduce_sum([s.losses for s in self.steps])
    
def calc_rmse(y_pred,y_true):
    y_pred_flat = np.asarray(y_pred).flatten()
    y_true_flat = np.asarray(y_true).flatten()
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    return rmse

    
class BNN_Reg(tf.keras.Model):
    
    def __init__(self, dims, name=None):
        
        super(BNN_Reg, self).__init__(name=name)
        
        # Multilayer fully-connected neural network to predict mean
        self.loc_mean = BayesianDenseNetwork(dims)
        
        # Variational distribution variables for observation error
        self.std_alpha = tf.Variable([10.0], name='std_alpha')
        self.std_beta = tf.Variable([10.0], name='std_beta')

    
    def call(self, x, sampling=True):
        
        # Predict means
        loc_preds = self.loc_mean(x, sampling=sampling)
    
        # Predict std deviation
        post_dist = tfd.Gamma(self.std_alpha, self.std_beta)
        adjust = lambda x: tf.sqrt(tf.math.reciprocal(x))
        N = x.shape[0]
        if sampling:
            std_preds = adjust(post_dist.sample([N]))
        else:
            std_preds = tf.ones([N, 1])*adjust(post_dist.mean())
    
        # Return mean and std predictions
        return tf.concat([loc_preds, std_preds], 1)
    
    
    def ll(self, x, y, sampling=True):
        mean_std = self.call(x, sampling=sampling)
        return tfd.Normal(mean_std[:,0], mean_std[:,1]).log_prob(y[:,0])
    
    def Normal_Sampling(self, x):
        preds = self.call(x)
        return tfd.Normal(preds[:,0], preds[:,1]).sample()
    
    def sampling(self, x, n = 1):
        sampling = np.zeros((x.shape[0], n))
        for k in range(n_samples):
            sampling[:,k] = self.Normal_Sampling(x)
        return sampling
    
    @property
    def loss(self):
        
        mean_loss = self.loc_mean.losses
        post_dist = tfd.Gamma(self.std_alpha, self.std_beta)
        prior = tfd.Gamma(10.0, 10.0)
        std_loss = tfd.kl_divergence(post_dist, prior)

        # Return the sum of both
        return mean_loss + std_loss
    
def process(model, optimizer, x_data, y_data, N):  
    #calculating lower bound (elbo method)
    with tf.GradientTape() as gradtape:
        ll = model.ll(x_data, y_data)
        model_loss = model.loss 
        train_cost = model_loss/N - tf.reduce_mean(ll)
    derivatives = gradtape.gradient(train_cost, model.trainable_variables)
    optimizer.apply_gradients(zip(derivatives, model.trainable_variables))
    return train_cost

def perform(model, optimizer, cycles, train_data, test_data, N):
    train_elbo = np.zeros(cycles)
    mse_root = []
    y_pred = []; y = []
    for ep in range(cycles):
            #Update weights each batch
        for X_values, y_values in train_data:
            train_elbo[ep] += process(model, optimizer, X_values, y_values, N)  

    # Evaluate performance on validation data
        for X_values, y_values in test_data:
            y_pred.append(model(X_values, sampling=False)[:, 0])
            y.append(y_values)
        mse_root.append(calc_rmse(np.asarray(y_pred), np.asarray(y)))
    return mse_root[-1]