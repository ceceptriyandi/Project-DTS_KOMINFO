#!/usr/bin/env python3

import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from math import sqrt

# function drop data zero or NaN
def drop_data(df):    
    df = df[df['Volume'] != 0]
    df = df[df['Volume'].notnull()]
    return df

# function describe data
def shape_data(df):    
    shape = df.shape
    return shape
  
# function describe data
def describe_data(df):    
    describe = df.describe()
    return describe
    
# function correlation data
def correlation_data(df):    
    correlations = df.corr(method='pearson')
    return correlations

# function skew data
def skew_data(df):    
    skew = df.skew()
    return skew

# function min-max normalization
def normalization_data(df):
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    df['Open'] = min_max_scaler.fit_transform(df['Open'].values.reshape(-1,1))
    df['High'] = min_max_scaler.fit_transform(df['High'].values.reshape(-1,1))
    df['Low'] = min_max_scaler.fit_transform(df['Low'].values.reshape(-1,1))
    df['Close'] = min_max_scaler.fit_transform(df['Close'].values.reshape(-1,1))
    df['Volume'] = min_max_scaler.fit_transform(df['Volume'].values.reshape(-1,1))
    return df

# function create train, validation, test data given stock data and sequence length
def load_data(df, seq_len):    
    # split data in 80%/10%/10% train/validation/test sets
    valid_set_size_percentage = 10 
    test_set_size_percentage = 10

    data_raw = df.as_matrix() # convert to numpy array
    data = []
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - seq_len): 
        data.append(data_raw[index: index + seq_len])
    
    data = np.array(data)
    valid_set_size = int(np.round(valid_set_size_percentage/100*data.shape[0]))  
    test_set_size = int(np.round(test_set_size_percentage/100*data.shape[0]))
    train_set_size = data.shape[0] - (valid_set_size + test_set_size)
    
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    x_valid = data[train_set_size:train_set_size+valid_set_size,:-1,:]
    y_valid = data[train_set_size:train_set_size+valid_set_size,-1,:]
    
    x_test = data[train_set_size+valid_set_size:,:-1,:]
    y_test = data[train_set_size+valid_set_size:,-1,:]
    
    print('x_train.shape = ',x_train.shape)
    print('y_train.shape = ', y_train.shape)
    print('x_valid.shape = ',x_valid.shape)
    print('y_valid.shape = ', y_valid.shape)
    print('x_test.shape = ', x_test.shape)
    print('y_test.shape = ',y_test.shape)
    print('\n')
    
    return [x_train, y_train, x_valid, y_valid, x_test, y_test]

# function to get the next batch
def get_next_batch(x_train,y_train,perm_array,batch_size): 
    global index_in_epoch
    start = index_in_epoch
    index_in_epoch += batch_size
    
    if index_in_epoch > x_train.shape[0]:
        np.random.shuffle(perm_array) # shuffle permutation array
        start = 0 # start next epoch
        index_in_epoch = batch_size
        
    end = index_in_epoch
    return x_train[perm_array[start:end]], y_train[perm_array[start:end]]

# model RNN
def rnn(df, model): 
    # create train, validation and test data
    seq_len = 20 # choose sequence length
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(df, seq_len)
    
    perm_array  = np.arange(x_train.shape[0])
    np.random.shuffle(perm_array)
    
    # parameters
    n_steps = seq_len-1 
    n_inputs = 5 
    n_neurons = 100
    n_outputs = 5
    n_layers = 5
    learning_rate = 0.001
    batch_size = 100
    n_epochs = 50
    train_set_size = x_train.shape[0]
    test_set_size = x_test.shape[0]
    
    tf.reset_default_graph()
    
    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_outputs])
    
    if model == 0:
        # use Basic LSTM Cell 
        layers = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons, activation=tf.nn.elu) for layer in range(n_layers)]
    elif model == 1:
        # use GRU cell
        layers = [tf.contrib.rnn.GRUCell(num_units=n_neurons, activation=tf.nn.leaky_relu) for layer in range(n_layers)]
    elif model == 2:
        # use Basic RNN Cell
        layers = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.elu) for layer in range(n_layers)]
    elif model == 3:
        # use LSTM Cell with peephole connections
        layers = [tf.contrib.rnn.LSTMCell(num_units=n_neurons, activation=tf.nn.leaky_relu, use_peepholes = True) for layer in range(n_layers)]
                                                                         
    multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
    rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)
    
    stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons]) 
    stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
    outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])
    outputs = outputs[:,n_steps-1,:] # keep only last output of sequence
                                                  
    loss = tf.reduce_mean(tf.square(outputs - y)) # loss function = mean squared error 
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) 
    training_op = optimizer.minimize(loss)
                                                  
    # run graph
    with tf.Session() as sess: 
        sess.run(tf.global_variables_initializer())
        for iteration in range(int(n_epochs*train_set_size/batch_size)):
            x_batch, y_batch = get_next_batch(x_train,y_train,perm_array,batch_size) # fetch the next training batch 
            sess.run(training_op, feed_dict={X: x_batch, y: y_batch}) 
            if iteration % int(5*train_set_size/batch_size) == 0:
                mse_train = loss.eval(feed_dict={X: x_train, y: y_train}) 
                mse_valid = loss.eval(feed_dict={X: x_valid, y: y_valid}) 
                print('%.0f epochs: loss train/valid = %.6f/%.6f'%(iteration*batch_size/train_set_size, mse_train, mse_valid))
           
        y_train_pred = sess.run(outputs, feed_dict={X: x_train})
        y_valid_pred = sess.run(outputs, feed_dict={X: x_valid})
        y_test_pred = sess.run(outputs, feed_dict={X: x_test})
        
        mse_train = mean_squared_error(y_train, y_train_pred)
        rms_train = sqrt(mean_squared_error(y_train, y_train_pred))
        mse_valid = mean_squared_error(y_valid, y_valid_pred)
        rms_valid = sqrt(mean_squared_error(y_valid, y_valid_pred))        
        mse_test = mean_squared_error(y_test, y_test_pred)
        rms_test = sqrt(mean_squared_error(y_test, y_test_pred))
        
        print('\n\nMSE train: %.6f'%(mse_train))
        print('RMSE train: %.6f'%(rms_train) + '\n\n')
        print('MSE valid: %.6f'%(mse_valid))
        print('RMSE valid: %.6f'%(rms_valid) + '\n\n')
        print('MSE test: %.6f'%(mse_test))
        print('RMSE test: %.6f'%(rms_test) + '\n')
        
    return x_train, y_train, x_valid, y_valid, x_test, y_test, y_train_pred, y_valid_pred, y_test_pred

def visualization_result(index, x_train, y_train, x_valid, y_valid, x_test, y_test, y_train_pred, y_valid_pred, y_test_pred):
    # visualization
    plt.figure(figsize=(20, 6))
    plt.subplot(1,2,1);
    plt.plot(np.arange(y_train.shape[0]), y_train[:,index], color='blue', label='train target')
    plt.plot(np.arange(y_train.shape[0], y_train.shape[0]+y_valid.shape[0]), y_valid[:,index],color='gray', label='valid target')
    plt.plot(np.arange(y_train.shape[0]+y_valid.shape[0],y_train.shape[0]+y_test.shape[0]+y_test.shape[0]),y_test[:,index], color='black', label='test target')
    plt.plot(np.arange(y_train_pred.shape[0]),y_train_pred[:,index], color='red',label='train prediction')
    plt.plot(np.arange(y_train_pred.shape[0], y_train_pred.shape[0]+y_valid_pred.shape[0]),y_valid_pred[:,index], color='orange', label='valid prediction')
    plt.plot(np.arange(y_train_pred.shape[0]+y_valid_pred.shape[0],y_train_pred.shape[0]+y_valid_pred.shape[0]+y_test_pred.shape[0]),y_test_pred[:,index], color='green', label='test prediction')
    plt.xlabel('time [days]')
    plt.ylabel('normalized price')
    plt.legend(loc='best');
    plt.show()
    
    print('\n')
    
    plt.figure(figsize=(20, 6))
    plt.subplot(1,2,2);
    plt.plot(np.arange(y_train.shape[0], y_train.shape[0]+y_test.shape[0]),y_test[:,index], color='black', label='test target')
    plt.plot(np.arange(y_train_pred.shape[0], y_train_pred.shape[0]+y_test_pred.shape[0]),y_test_pred[:,index], color='green', label='test prediction')
    plt.xlabel('time [days]')
    plt.ylabel('normalized price')
    plt.legend(loc='best');
    plt.show()
     
def signal_ma(df):
    short_price = 30
    long_price = 90
    
    dataframe = pd.DataFrame(df, columns=['Open','High','Low','Close','Volume'])
    
    # Initialize the `signals` DataFrame with the `signal` column
    signals = pd.DataFrame(index=dataframe.index)
    signals['Signal'] = 0.0

    # Create short simple moving average over the short window
    signals['Long_MA'] = dataframe['Close'].rolling(window = short_price, min_periods=1, center=False).mean()
    # Create long simple moving average over the long window
    signals['Short_MA'] = dataframe['Close'].rolling(window = long_price, min_periods=1, center=False).mean()

    # Create signals
    signals['Signal'][short_price:] = np.where(signals['Short_MA'][short_price:] > signals['Long_MA'][short_price:], 1.0, 0.0)   

    # Generate trading orders
    signals['Positions'] = signals['Signal'].diff()
    
    # Initialize the plot figure
    fig = plt.figure(figsize=(20, 6))
    
    # Add a subplot and label for y-axis
    ax = fig.add_subplot(111,  ylabel='Price')
    # Plot the closing price
    dataframe['Close'].plot(ax = ax, lw = 1.)
    
    # Plot the short and long moving averages
    signals[['Short_MA', 'Long_MA']].plot(ax = ax, lw = 1.)

    # Plot the buy signals
    ax.plot(signals.loc[signals.Positions == -1.0].index,
             signals.Long_MA[signals.Positions == -1.0],
             '^', markersize=8, color='g')
    
    # Plot the sell signals
    ax.plot(signals.loc[signals.Positions == 1.0].index, 
             signals.Short_MA[signals.Positions == 1.0],
             'v', markersize=8, color='r')
    
    plt.show()
    
###############################################################################

# dataset 
df = pd.read_csv("BBNI_15y.csv",usecols=['Open','Low','High','Close','Volume'])

# preprocessing and data understanding
shape = shape_data(df)
print(shape)
print('\n')
df = drop_data(df)
describe = describe_data(df)
print(describe)
print('\n')
correlation = correlation_data(df)
print(correlation)
print('\n')
skew = skew_data(df)
print(skew)
print('\n')
df = normalization_data(df)

# compile
# 0 = Basic LSTM Cell, 1 = GRU cell, 2 = Basic RNN Cell, 3 = LSTM Cell with peephole connections
for model in range(0,3):
    index_in_epoch = 0
    x_train, y_train, x_valid, y_valid, x_test, y_test, y_train_pred, y_valid_pred, y_test_pred = rnn(df, model) 
    # visualization and result
    # 0 = open, 1 = high, 2 = low, 3 = close
    visualization_result(0, x_train, y_train, x_valid, y_valid, x_test, y_test, y_train_pred, y_valid_pred, y_test_pred)
    # signal MA
    signal_ma(y_test_pred)
    print('\n')
    # 0 = open, 1 = high, 2 = low, 3 = close
    visualization_result(3, x_train, y_train, x_valid, y_valid, x_test, y_test, y_train_pred, y_valid_pred, y_test_pred)
    # signal MA
    signal_ma(y_test_pred)
    print('\n')
