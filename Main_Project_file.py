
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.preprocessing
import datetime
import math
import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


# In[20]:


# import dataset 
dataset = pd.read_csv('Reliance.csv')
df_stock = dataset.copy()
df_stock = df_stock.dropna()
df_dates=df_stock[['Date']]
df_stock = df_stock[['Close']]


# In[21]:


df_stock_norm = df_stock.copy()
scaler = sklearn.preprocessing.MinMaxScaler()
df_stock_norm['Close'] = scaler.fit_transform(df_stock_norm['Close'].values.reshape(-1,1))


# In[22]:


# Splitting the dataset into Train, Valid & test data and creation of timesteps
valid_set_size_percentage = 10 
test_set_size_percentage = 10 
seq_len = 20 # taken sequence length as 20
def load_data(stock, seq_len):
    data_raw = stock.as_matrix() 
    data = [] 
    for index in range(len(data_raw) - seq_len): 
        data.append(data_raw[index: index + seq_len])
    data = np.array(data);
    valid_set_size = int(np.round(valid_set_size_percentage/100*data.shape[0]));  
    test_set_size = int(np.round(test_set_size_percentage/100*data.shape[0]));
    train_set_size = data.shape[0] - (valid_set_size + test_set_size);
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    x_valid = data[train_set_size:train_set_size+valid_set_size,:-1,:]
    y_valid = data[train_set_size:train_set_size+valid_set_size,-1,:]
    x_test = data[train_set_size+valid_set_size:,:-1,:]
    y_test = data[train_set_size+valid_set_size:,-1,:]
    return [x_train, y_train, x_valid, y_valid, x_test, y_test]
x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(df_stock_norm, seq_len)


# In[23]:


print('x_train.shape = ',x_train.shape)
print('y_train.shape = ', y_train.shape)
print('x_valid.shape = ',x_valid.shape)
print('y_valid.shape = ', y_valid.shape)
print('x_test.shape = ', x_test.shape)
print('y_test.shape = ',y_test.shape)


# In[24]:


"""Building the Model"""

# parameters & Placeholders 
n_steps = seq_len-1 
n_inputs = 1
n_neurons = 200 
n_outputs = 1
n_layers = 2
learning_rate = 0.001
batch_size = 50
n_epochs = 100 
train_set_size = x_train.shape[0]
test_set_size = x_test.shape[0]
tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_outputs])


# In[25]:


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


# In[26]:


#RNN
layers = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.elu)
         for layer in range(n_layers)]


# In[27]:


multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)
stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons]) 
stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])
outputs = outputs[:,n_steps-1,:] # keep only last output of sequence
                                              
# Cost function
loss = tf.reduce_mean(tf.square(outputs - y))

#optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) 
training_op = optimizer.minimize(loss)


# In[28]:


# Fitting the model
MSE=[]
with tf.Session() as sess: 
    sess.run(tf.global_variables_initializer())
    for iteration in range(int(n_epochs*train_set_size/batch_size)):
        x_batch, y_batch = get_next_batch(batch_size) # fetch the next training batch 
        sess.run(training_op, feed_dict={X: x_batch, y: y_batch}) 
        if iteration % int(5*train_set_size/batch_size) == 0:
            mse_train = loss.eval(feed_dict={X: x_train, y: y_train}) 
            MSE.append(mse_train)
            mse_valid = loss.eval(feed_dict={X: x_valid, y: y_valid}) 
            print('%.2f epochs: MSE train/valid = %.6f/%.6f'%(
                iteration*batch_size/train_set_size, mse_train, mse_valid))
            
# Predictions
    y_test_pred = sess.run(outputs, feed_dict={X: x_test})
    
#checking prediction output nos 
y_test_pred.shape


# In[29]:


l=len(MSE)*5
fig=plt.figure(figsize=(18,9))
plt.plot(range(0,l,5),MSE[:],color='red',label='MSE error')
fig.suptitle('General MSE error curve', fontsize=20)
plt.xlabel('Epochs',fontsize=18)
plt.ylabel('MSE error',fontsize=18)
plt.legend()
plt.show()
fig.savefig('MSE.jpg')


# In[30]:


y_test=scaler.inverse_transform(y_test.reshape(-1,1))
y_test_pred=scaler.inverse_transform(y_test_pred.reshape(-1,1))


# In[31]:


print('x_test\t\t x_test_pred')
for i in range(len(y_test)):
    print(y_test[i][0],'\t',y_test_pred[i][0])


# In[33]:


# ploting the graph
l=len(df_dates)
df_dates=df_dates[l-y_test_pred.shape[0]:]
comp = pd.DataFrame({'Column1':y_test[:,0],'Column2':y_test_pred[:,0]})
fig=plt.figure(figsize=(18,9))
plt.plot(range(df_dates.shape[0]),comp['Column1'], color='blue', label='Test')
plt.plot(range(df_dates.shape[0]),comp['Column2'], color='black', label='Prediction')
plt.xticks(range(0,df_dates.shape[0],20),df_dates['Date'].loc[::20],rotation=20)
fig.suptitle('Reliance Closing Stock Price', fontsize=20)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close price',fontsize=18)
plt.legend()
plt.show()
fig.savefig('Reliance.jpg')


# In[34]:


#accuracy calculation
a,b=[],[]
for i in range(len(y_test)-1):
    if ((y_test_pred[i+1][0]-y_test_pred[i][0])>=0):
        a.append(1)
    else:
        a.append(-1)
    if((y_test[i+1][0]-y_test[i][0])>=0):
        b.append(1)
    else:
        b.append(-1)
c=0
for i in range(len(a)):
    if a[i]==b[i]:
        c=c+1
accuracy=(c/len(y_test))*100
print('correct movement trends=',c)
print('total predictions=',len(y_test))
print('accuracy=',accuracy,'%')


# In[ ]:




