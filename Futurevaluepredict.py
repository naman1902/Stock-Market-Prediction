
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
import math
import sklearn
import sklearn.preprocessing
import datetime
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


# In[19]:


# import dataset 
dataset = pd.read_csv('Reliance.csv')
df= dataset.copy()
df= df.dropna()
df= df[['Date','Close']]


# In[20]:


plt.figure(figsize = (18,9))
plt.plot(range(df.shape[0]),(df['Close']))
plt.xticks(range(0,df.shape[0],500),df['Date'].loc[::500],rotation=45)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close price',fontsize=18)
plt.show()


# In[21]:


#close price matrix formation and train test split
close_prices = df.loc[:,'Close'].as_matrix()
train_data = close_prices[:]
print(train_data)


# In[22]:


scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data.reshape(-1,1))
print(train_data)


# In[23]:


print(train_data)


# In[24]:


seq_len = 20 # taken sequence length as 20
data_raw=train_data
data = [] 
for index in range(len(data_raw) - seq_len):
    data.append(data_raw[index: index + seq_len])
data = np.array(data);


# In[25]:


x_train = data[:,:-1,:]
y_train = data[:,-1,:]


# In[26]:


print('x_train.shape = ',x_train.shape)
print('y_train.shape = ', y_train.shape)


# In[27]:


x_test=[]
y_test=[]
for i in range(seq_len-1,0,-1):
    x_test.append(train_data[-i])
x_test=np.array(x_test)
print(x_test)
print('x_test.shape=',x_test.shape)
x_test=[x_test]
x_test=np.array(x_test)
print('x_test.shape=',x_test.shape)


# In[42]:


# parameters & Placeholders 
n_steps = seq_len-1 
n_inputs = 1
n_neurons = 200 
n_outputs = 1
n_layers = 2
learning_rate = 0.001
batch_size = 50
n_epochs = 50 
train_set_size = x_train.shape[0]
test_set_size = x_test.shape[0]
tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_outputs])


# In[43]:


# function to get the next batch
index_in_epoch = 0;
perm_array  = np.arange(x_train.shape[0])
np.random.shuffle(perm_array)

def get_next_batch(batch_size):
    global index_in_epoch, x_train, perm_array   
    start = index_in_epoch
    index_in_epoch += batch_size 
    if index_in_epoch > x_train.shape[0]:
        np.random.shuffle(perm_array) # shuffle permutation array
        start = 0 # start next epoch
        index_in_epoch = batch_size     
    end = index_in_epoch
    return x_train[perm_array[start:end]], y_train[perm_array[start:end]]


# In[44]:


#RNN 
layers = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.elu)
         for layer in range(n_layers)]
# LSTM  
#layers = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons, activation=tf.nn.elu)
#        for layer in range(n_layers)]

#LSTM with peephole connections
#layers = [tf.contrib.rnn.LSTMCell(num_units=n_neurons, 
#                                  activation=tf.nn.leaky_relu, use_peepholes = True)
#          for layer in range(n_layers)]

#GRU 
#layers = [tf.contrib.rnn.GRUCell(num_units=n_neurons, activation=tf.nn.leaky_relu)
#          for layer in range(n_layers)] 


# In[45]:


multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)
stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons]) 
stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])
outputs = outputs[:,n_steps-1,:] # keep only last output of sequence


# In[46]:


# Cost function
loss = tf.reduce_mean(tf.square(outputs - y))

#optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) 
training_op = optimizer.minimize(loss)
                                       


# In[47]:


# Fitting the model
ytestpredictions=[]
train_set_size=x_train.shape[0]
with tf.Session() as sess: 
    sess.run(tf.global_variables_initializer())
    for iteration in range(int(n_epochs*train_set_size/batch_size)):
        x_batch, y_batch = get_next_batch(batch_size) # fetch the next training batch 
        sess.run(training_op, feed_dict={X: x_batch, y: y_batch}) 
        if iteration % int(5*train_set_size/batch_size) == 0:
            mse_train = loss.eval(feed_dict={X: x_train, y: y_train}) 
            print('%.2f epochs: MSE train = %.6f/'%(
                iteration*batch_size/train_set_size, mse_train))
            
    y_train_pred = sess.run(outputs, feed_dict={X: x_train})
    ytest_pred=sess.run(outputs,  feed_dict={X: x_test})
    ytest=ytest_pred
    print(ytest)


# In[48]:



y=scaler.inverse_transform(ytest.reshape(-1,1))
print(y)

