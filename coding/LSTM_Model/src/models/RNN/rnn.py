# Note: All calls to tf.name_scope or tf.summary.* support TensorBoard visualization.

import os
import tensorflow as tf
from configparser import ConfigParser

from models.RNN.utils import variable_on_cpu

def LSTM(conf_path, batch_x, seq_length, n_input, n_context):
    '''
    This function was initially based on open source code from Mozilla DeepSpeech:
    https://github.com/mozilla/DeepSpeech/blob/master/DeepSpeech.py

    # This Source Code Form is subject to the terms of the Mozilla Public
    # License, v. 2.0. If a copy of the MPL was not distributed with this
    # file, You can obtain one at http://mozilla.org/MPL/2.0/.
    '''
    parser = ConfigParser(os.environ)
    parser.read(conf_path)

    # SimpleLSTM
    n_character = parser.getint('simplelstm', 'n_character')
    relu_clip = parser.getint('simplelstm', 'relu_clip')
    b1_stddev = parser.getfloat('simplelstm', 'b1_stddev')
    h1_stddev = parser.getfloat('simplelstm', 'h1_stddev')
    b2_stddev = parser.getfloat('simplelstm', 'b2_stddev')  
    h2_stddev = parser.getfloat('simplelstm', 'h2_stddev')
    b3_stddev = parser.getfloat('simplelstm', 'b3_stddev')  
    h3_stddev = parser.getfloat('simplelstm', 'h3_stddev')
    b4_stddev = parser.getfloat('simplelstm', 'b4_stddev')  
    h4_stddev = parser.getfloat('simplelstm', 'h4_stddev')
    dropout = [float(x) for x in parser.get('simplelstm', 'dropout_rates').split(',')]
    n_layers = parser.getint('simplelstm', 'n_layers')
    n_hidden_units = parser.getint('simplelstm', 'n_hidden_units')
    n_hidden_units_1 = parser.getint('simplelstm', 'n_hidden_units_1')
    n_hidden_units_2 = parser.getint('simplelstm', 'n_hidden_units_2')
    n_hidden_units_3 = parser.getint('simplelstm', 'n_hidden_units_3')
    n_hidden_units_4 = parser.getint('simplelstm', 'n_hidden_units_4')

    # Input shape: [batch_size, n_steps, n_input + 2*n_input*n_context]
    # batch_x_shape = tf.shape(batch_x)

    #input_tensor_shape = tf.shape(input_tensor)
    #n_items = input_tensor_shape[0]

    # Input shape: [batch_size, n_steps, n_input + 2*n_input*n_context]
    batch_x_shape = tf.shape(batch_x)
    n_items = batch_x_shape[0]
    print(batch_x_shape)
    print(batch_x)
    # Reshaping `batch_x` to a tensor with shape `[n_steps*batch_size, n_input + 2*n_input*n_context]`.
    # This is done to prepare the batch for input into the first layer which expects a tensor of rank `2`.

    # Permute n_steps and batch_size
    batch_x = tf.transpose(batch_x, [1, 0, 2])
    # Reshape to prepare input for first layer
    temp_shape = tf.shape(batch_x)
    print(temp_shape)
    batch_x = tf.reshape(batch_x,
                         [-1, n_input + 2 * n_input * n_context])  # (n_steps*batch_size, n_input + 2*n_input*n_context)

    with tf.name_scope('fc1'):
        # Initialize weights and biases
        # with tf.device('/cpu:0'):
        # b = tf.get_variable('b', initializer=tf.zeros_initializer([n_character]))
       
        #Initialize weight 1
        W1 = tf.get_variable('W1', shape=[n_input + 2 * n_input * n_context, n_hidden_units_1],
                            initializer=tf.random_normal_initializer(stddev=h1_stddev),
                            )
        #Initialize bias 1
        b1 = tf.get_variable('b1', shape=[n_hidden_units_1],
                            initializer=tf.random_normal_initializer(stddev=b1_stddev)
                            )

        fc_layer1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(batch_x, W1),b1)),relu_clip)
        fc_layer1 = tf.nn.dropout(fc_layer1,(1.0 - dropout[0]))
        
        # tf.summary.histogram("weights", W1)
        # tf.summary.histogram("biases", b1)
        # tf.summary.histogram("activations", fc_layer1)  

    with tf.name_scope('fc2'):
        #Initialize weight 2
        W2 = tf.get_variable('W2', shape=[n_hidden_units_1, n_hidden_units_2],
                            initializer=tf.random_normal_initializer(stddev=h2_stddev),
                            )
        #Initialize bias 2
        b2 = tf.get_variable('b2', shape=[n_hidden_units_2],
                            initializer=tf.random_normal_initializer(stddev=b2_stddev)
                            )
        fc_layer2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(fc_layer1, W2),b2)),relu_clip)
        fc_layer2 = tf.nn.dropout(fc_layer2,(1.0 - dropout[1]))
        
        # tf.summary.histogram("weights", W2)
        # tf.summary.histogram("biases", b2)
        # tf.summary.histogram("activations", fc_layer2) 
 
    with tf.name_scope('lstm'):
        # Define the cell
        # Can be:
        #   tf.contrib.rnn.BasicRNNCell
        #   tf.contrib.rnn.GRUCell
        
        # Stacking rnn cells
        # Changed from TF 1.2. Current way:
        stacked_rnn = []
        for cell_numb in range(n_layers):
            #stacked_rnn.append(tf.contrib.rnn.GRUCell(n_hidden_units,activation=tf.nn.relu))
            #stacked_rnn.append(tf.contrib.rnn.BasicLSTMCell(n_hidden_units, state_is_tuple=True))
            stacked_rnn.append(tf.contrib.rnn.LSTMCell(n_hidden_units,use_peepholes=True,forget_bias=1.0,activation=tf.nn.relu))

        #stack = tf.contrib.rnn.MultiRNNCell(cells=stacked_rnn, state_is_tuple=True)
        stack = tf.contrib.rnn.MultiRNNCell(cells=stacked_rnn)

        # `fc_layer3` is now reshaped back into shape of batch_X `[n_steps, batch_size, 2*n_cell_dim]`,
        #INITIAL: [batch_size, n_steps, n_input + 2*n_input*n_context]
        #                                      n_steps*batch_size, n_input + 2*n_input*n_context)
        #BEGINNING: batch_x = tf.reshape(batch_x,[-1, n_input + 2 * n_input * n_context])
        #THEIR: layer_3 = tf.reshape(layer_3, [-1, batch_x_shape[0], n_hidden_3])
        # as the LSTM BRNN expects its input to be of shape `[max_time, batch_size, input_size]`.
        #permuting n steps and batch size
        #fc_layer2 = tf.transpose(fc_layer2, [1, 0, 2])
        
        #fc_layer2 = tf.reshape(fc_layer2, [-1, batch_x_shape[0], 195])   ####CHECK THIS!
        #fc_layer2 = tf.reshape(fc_layer2, temp_shape)
        fc_layer2 = tf.reshape(fc_layer2, [-1, batch_x_shape[0], n_hidden_units])
        fc_layer2 = tf.transpose(fc_layer2, [1, 0, 2])
        #batch_x_shape[2]
        # Get layer activations (second output is the final state of the layer, do not need)
        outputs, _ = tf.nn.dynamic_rnn(stack, fc_layer2, seq_length,
                                       time_major=False, dtype=tf.float32)
        
        # tf.summary.histogram("activations", outputs)        

        # Reshape to apply the same weights over the timesteps
        outputs = tf.reshape(outputs, [-1, n_hidden_units]) 
       
     #Initialize logit weights
    with tf.name_scope('fc3'):
         #Initialize logit weights
        W3 = tf.get_variable('W3', shape=[n_hidden_units, n_hidden_units_3],
                            # initializer=tf.truncated_normal_initializer(stddev=h1_stddev),
                            initializer=tf.random_normal_initializer(stddev=h3_stddev),
                            )
        #Initialize logit bias
        b3 = tf.get_variable('b3', shape=[n_hidden_units_3],
                            # initializer=tf.constant_initializer(value=0),
                            initializer=tf.random_normal_initializer(stddev=b3_stddev),
                            )
        fc_layer3 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(outputs, W3),b3)),relu_clip)
        fc_layer3 = tf.nn.dropout(fc_layer3,(1.0 - dropout[2]))
        
        # tf.summary.histogram("weights", W3)
        # tf.summary.histogram("biases", b3)
        # tf.summary.histogram("activations", fc_layer3)   

    with tf.name_scope('logits'):
        W = tf.get_variable('W', shape=[n_hidden_units_3, n_character],
                            # initializer=tf.truncated_normal_initializer(stddev=h1_stddev),
                            initializer=tf.random_normal_initializer(stddev=h4_stddev),
                            )
        #Initialize logit bias
        b = tf.get_variable('b', shape=[n_character],
                            # initializer=tf.constant_initializer(value=0),
                            initializer=tf.random_normal_initializer(stddev=b4_stddev),
                            )
        # Perform affine transformation to layer output:
        # multiply by weights (linear transformation), add bias (translation)
        logits = tf.add(tf.matmul(fc_layer3, W), b)         #logit shape [-1, n_char_]

        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", logits)   
             
        
        ##Normal layer 1 with clipped relu activation
        #normal_layer_1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(outputs,W2),b2)),relu_clip)
        #normal_layer_1 = tf.nn.dropout(normal_layer_1,(1.0 - dropout))
        
        # New logits
        #logits = tf.add(tf.matmul(normal_layer_1,W),b)
        #logits = tf.minimum(tf.nn.relu(tf.add(tf.matmul(outputs,W3),b3)),relu_clip)

        # Reshaping back to the original shape
    logits = tf.reshape(logits, [n_items, -1, n_character])

        # Put time as the major axis
    logits = tf.transpose(logits, (1, 0, 2))

    summary_op = tf.summary.merge_all()

    return logits, summary_op

def SimpleLSTM(conf_path, input_tensor, seq_length):
    '''
    This function was initially based on open source code from Mozilla DeepSpeech:
    https://github.com/mozilla/DeepSpeech/blob/master/DeepSpeech.py

    # This Source Code Form is subject to the terms of the Mozilla Public
    # License, v. 2.0. If a copy of the MPL was not distributed with this
    # file, You can obtain one at http://mozilla.org/MPL/2.0/.
    '''
    print("###################################################################")
    print("\n \n \n SIMPLE LSTM!!\n \n \n")
    print("####################################################################")

    parser = ConfigParser(os.environ)
    parser.read(conf_path)

    # SimpleLSTM
    n_character = parser.getint('simplelstm', 'n_character')
    b1_stddev = parser.getfloat('simplelstm', 'b1_stddev')
    h1_stddev = parser.getfloat('simplelstm', 'h1_stddev')
    n_layers = parser.getint('simplelstm', 'n_layers')
    n_hidden_units = parser.getint('simplelstm', 'n_hidden_units')

    # Input shape: [batch_size, n_steps, n_input + 2*n_input*n_context]
    # batch_x_shape = tf.shape(batch_x)

    input_tensor_shape = tf.shape(input_tensor)
    n_items = input_tensor_shape[0]

    with tf.name_scope("lstm"):
        # Initialize weights
        # with tf.device('/cpu:0'):
        W = tf.get_variable('W', shape=[n_hidden_units, n_character],
                            # initializer=tf.truncated_normal_initializer(stddev=h1_stddev),
                            initializer=tf.random_normal_initializer(stddev=h1_stddev),
                            )
        # Initialize bias
        # with tf.device('/cpu:0'):
        # b = tf.get_variable('b', initializer=tf.zeros_initializer([n_character]))
        b = tf.get_variable('b', shape=[n_character],
                            # initializer=tf.constant_initializer(value=0),
                            initializer=tf.random_normal_initializer(stddev=b1_stddev),
                            )

        # Define the cell
        # Can be:
        #   tf.contrib.rnn.BasicRNNCell
        #   tf.contrib.rnn.GRUCell
        #cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, state_is_tuple=True)

        # Stacking rnn cells
        #stack = tf.contrib.rnn.MultiRNNCell([cell] * n_layers, state_is_tuple=True)
        
        stacked_rnn = []
        for cell_numb in range(n_layers):
            #stacked_rnn.append(tf.contrib.rnn.GRUCell(n_hidden_units,activation=tf.nn.relu))
            stacked_rnn.append(tf.contrib.rnn.BasicLSTMCell(n_hidden_units, state_is_tuple=True))
        
        stack = tf.contrib.rnn.MultiRNNCell(cells=stacked_rnn)

        # Get layer activations (second output is the final state of the layer, do not need)
        outputs, _ = tf.nn.dynamic_rnn(stack, input_tensor, seq_length,
                                       time_major=False, dtype=tf.float32)

        # Reshape to apply the same weights over the timesteps
        outputs = tf.reshape(outputs, [-1, n_hidden_units]) 

        # Perform affine transformation to layer output:
        # multiply by weights (linear transformation), add bias (translation)
        logits = tf.add(tf.matmul(outputs, W), b)         #logit shape [-1, n_char_]

        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", logits)

        # Reshaping back to the original shape
        logits = tf.reshape(logits, [n_items, -1, n_character])

        # Put time as the major axis
        logits = tf.transpose(logits, (1, 0, 2))

        summary_op = tf.summary.merge_all()

    return logits, summary_op


#def SimpleLSTM(conf_path, input_tensor, seq_length):
#    '''
#    This function was initially based on open source code from Mozilla DeepSpeech:
#    https://github.com/mozilla/DeepSpeech/blob/master/DeepSpeech.py

#    # This Source Code Form is subject to the terms of the Mozilla Public
#    # License, v. 2.0. If a copy of the MPL was not distributed with this
#    # file, You can obtain one at http://mozilla.org/MPL/2.0/.
#    '''
#    parser = ConfigParser(os.environ)
#    parser.read(conf_path)

#    # SimpleLSTM
#    n_character = parser.getint('simplelstm', 'n_character')
#    relu_clip = parser.getint('simplelstm', 'relu_clip')
#    b1_stddev = parser.getfloat('simplelstm', 'b1_stddev')
#    h1_stddev = parser.getfloat('simplelstm', 'h1_stddev')
#    b2_stddev = parser.getfloat('simplelstm', 'b2_stddev')  
#    h2_stddev = parser.getfloat('simplelstm', 'h2_stddev')
#    b3_stddev = parser.getfloat('simplelstm', 'b3_stddev')  
#    h3_stddev = parser.getfloat('simplelstm', 'h3_stddev')
#    dropout = [float(x) for x in parser.get('simplelstm', 'dropout_rates').split(',')]
#    n_layers = parser.getint('simplelstm', 'n_layers')
#    #n_hidden_units = parser.getint('simplelstm', 'n_hidden_units')
#    #n_hidden_units_1 = parser.getint('simplelstm', 'n_hidden_units_1')

#    # Input shape: [batch_size, n_steps, n_input + 2*n_input*n_context]
#    # batch_x_shape = tf.shape(batch_x)

#    #input_tensor_shape = tf.shape(input_tensor)
#    #n_items = input_tensor_shape[0]

#    # Input shape: [batch_size, n_steps, n_input + 2*n_input*n_context]
#    input_tensor_shape = tf.shape(batch_x)

#    # Reshaping `batch_x` to a tensor with shape `[n_steps*batch_size, n_input + 2*n_input*n_context]`.
#    # This is done to prepare the batch for input into the first layer which expects a tensor of rank `2`.

#    # Permute n_steps and batch_size
#    batch_x = tf.transpose(batch_x, [1, 0, 2])
#    # Reshape to prepare input for first layer
#    batch_x = tf.reshape(batch_x,
#                         [-1, n_input + 2 * n_input * n_context])  # (n_steps*batch_size, n_input + 2*n_input*n_context)

#    with tf.name_scope("lstm"):
#        # Initialize weights and biases
#        # with tf.device('/cpu:0'):
#        # b = tf.get_variable('b', initializer=tf.zeros_initializer([n_character]))
       
#        #Initialize weight 1
#        W1 = tf.get_variable('W1', shape=[n_hidden_units_1, n_hidden_units_2],
#                            initializer=tf.random_normal_initializer(stddev=h2_stddev),
#                            )
#        #Initialize bias 1
#        b1 = tf.get_variable('b1', shape=[n_hidden_units_2],
#                            initializer=tf.random_normal_initializer(stddev=b2_stddev)
#                            )
#        #Initialize weight 2
#        W2 = tf.get_variable('W2', shape=[n_hidden_units_2, n_hidden_units_3],
#                            initializer=tf.random_normal_initializer(stddev=h2_stddev),
#                            )
#        #Initialize bias 2
#        b2 = tf.get_variable('b2', shape=[n_hidden_units_3],
#                            initializer=tf.random_normal_initializer(stddev=b2_stddev)
#                            )
#        #Initialize logit weights
#        W3 = tf.get_variable('W3', shape=[n_hidden_units, n_character],
#                            # initializer=tf.truncated_normal_initializer(stddev=h1_stddev),
#                            initializer=tf.random_normal_initializer(stddev=h1_stddev),
#                            )
#        #Initialize logit bias
#        b3 = tf.get_variable('b3', shape=[n_character],
#                            # initializer=tf.constant_initializer(value=0),
#                            initializer=tf.random_normal_initializer(stddev=b1_stddev),
#                            )
       

#        # Define the cell
#        # Can be:
#        #   tf.contrib.rnn.BasicRNNCell
#        #   tf.contrib.rnn.GRUCell
        
#        # Stacking rnn cells
#        # Changed from TF 1.2. Current way:
#        stacked_rnn = []
#        for cell_numb in range(n_layers):
#            #stacked_rnn.append(tf.contrib.rnn.GRUCell(n_hidden_units,activation=tf.nn.relu))
#            stacked_rnn.append(tf.contrib.rnn.BasicLSTMCell(n_hidden_units, state_is_tuple=True))

#        #stack = tf.contrib.rnn.MultiRNNCell(cells=stacked_rnn, state_is_tuple=True)
#        stack = tf.contrib.rnn.MultiRNNCell(cells=stacked_rnn)
        
#        fc_layer1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(input_tensor,W1),b1)),relu_clip)
#        fc_layer1 = tf.nn.dropout(fc_layer1,(1.0 - dropout))

#        # Get layer activations (second output is the final state of the layer, do not need)
#        outputs, _ = tf.nn.dynamic_rnn(stack, input_tensor, seq_length,
#                                       time_major=False, dtype=tf.float32)

#        # Reshape to apply the same weights over the timesteps
#        outputs = tf.reshape(outputs, [-1, n_hidden_units]) 
        
#        ##Normal layer 1 with clipped relu activation
#        #normal_layer_1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(outputs,W2),b2)),relu_clip)
#        #normal_layer_1 = tf.nn.dropout(normal_layer_1,(1.0 - dropout))
        
#        # New logits
#        #logits = tf.add(tf.matmul(normal_layer_1,W),b)
#        #logits = tf.minimum(tf.nn.relu(tf.add(tf.matmul(outputs,W3),b3)),relu_clip)
#        # Perform affine transformation to layer output:
#        # multiply by weights (linear transformation), add bias (translation)
#        logits = tf.add(tf.matmul(outputs, W3), b3)         #logit shape [-1, n_char_]

#        tf.summary.histogram("weights", W3)
#        tf.summary.histogram("biases", b3)
#        tf.summary.histogram("activations", logits)

#        # Reshaping back to the original shape
#        logits = tf.reshape(logits, [n_items, -1, n_character])

#        # Put time as the major axis
#        logits = tf.transpose(logits, (1, 0, 2))

#        summary_op = tf.summary.merge_all()

#    return logits, summary_op

def BiRNN_V3(conf_path, batch_x, seq_length, n_input, n_context):

    parser = ConfigParser(os.environ)
    parser.read(conf_path)

    dropout = [float(x) for x in parser.get('birnn_V3', 'dropout_rates').split(',')]
    relu_clip = parser.getint('birnn_V3', 'relu_clip')

    b1_stddev = parser.getfloat('birnn_V3', 'b1_stddev')
    h1_stddev = parser.getfloat('birnn_V3', 'h1_stddev')
    b2_stddev = parser.getfloat('birnn_V3', 'b2_stddev')
    h2_stddev = parser.getfloat('birnn_V3', 'h2_stddev')
    b3_stddev = parser.getfloat('birnn_V3', 'b3_stddev')
    h3_stddev = parser.getfloat('birnn_V3', 'h3_stddev')
    b5_stddev = parser.getfloat('birnn_V3', 'b5_stddev')
    h5_stddev = parser.getfloat('birnn_V3', 'h5_stddev')
    b6_stddev = parser.getfloat('birnn_V3', 'b6_stddev')
    h6_stddev = parser.getfloat('birnn_V3', 'h6_stddev')

    n_hidden_1 = parser.getint('birnn_V3', 'n_hidden_1')
    n_hidden_2 = parser.getint('birnn_V3', 'n_hidden_2')
    n_hidden_5 = parser.getint('birnn_V3', 'n_hidden_5')
    n_cell_dim = parser.getint('birnn_V3', 'n_cell_dim')

    n_hidden_3 = int(eval(parser.get('birnn_V3', 'n_hidden_3')))
    n_hidden_6 = parser.getint('birnn_V3', 'n_hidden_6')

    # Input shape: [batch_size, n_steps, n_input + 2*n_input*n_context]
    batch_x_shape = tf.shape(batch_x)

    # Reshaping `batch_x` to a tensor with shape `[n_steps*batch_size, n_input + 2*n_input*n_context]`.
    # This is done to prepare the batch for input into the first layer which expects a tensor of rank `2`.

    # Permute n_steps and batch_size
    batch_x = tf.transpose(batch_x, [1, 0, 2])
    # Reshape to prepare input for first layer
    batch_x = tf.reshape(batch_x,
                         [-1, n_input + 2 * n_input * n_context])  # (n_steps*batch_size, n_input + 2*n_input*n_context)

    # The next three blocks will pass `batch_x` through three hidden layers with
    # clipped RELU activation and dropout.
    # 1st layer
    with tf.name_scope('fc1'):
        b1 = variable_on_cpu('b1', [n_hidden_1], tf.random_normal_initializer(stddev=b1_stddev))
        h1 = variable_on_cpu('h1', [n_input + 2 * n_input * n_context, n_hidden_1],
                             tf.random_normal_initializer(stddev=h1_stddev))
        layer_1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(batch_x, h1), b1)), relu_clip)         # clipped relu op
        layer_1 = tf.nn.dropout(layer_1, (1.0 - dropout[0]))

        tf.summary.histogram("weights", h1)
        tf.summary.histogram("biases", b1)
        tf.summary.histogram("activations", layer_1)

    # 2nd layer
    with tf.name_scope('fc2'):
        b2 = variable_on_cpu('b2', [n_hidden_3], tf.random_normal_initializer(stddev=b2_stddev))
        h2 = variable_on_cpu('h2', [n_hidden_1, n_hidden_3], tf.random_normal_initializer(stddev=h2_stddev))
        layer_2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_1, h2), b2)), relu_clip)
        layer_2 = tf.nn.dropout(layer_2, (1.0 - dropout[1]))

        tf.summary.histogram("weights", h2)
        tf.summary.histogram("biases", b2)
        tf.summary.histogram("activations", layer_2)

    with tf.name_scope('lstm_1'):
        # Forward direction cell:
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True)
        #lstm_fw_cell = tf.contrib.rnn.GRUCell(n_cell_dim,activation=tf.nn.relu)
        lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell,
                                                     input_keep_prob=1.0 - dropout[3],
                                                     output_keep_prob=1.0 - dropout[3],
                                                     # seed=random_seed,
                                                     )
        # Backward direction cell:
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True)
        #lstm_bw_cell = tf.contrib.rnn.GRUCell(n_cell_dim,activation=tf.nn.relu)
        lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell,
                                                     input_keep_prob=1.0 - dropout[4],
                                                     output_keep_prob=1.0 - dropout[4],
                                                     # seed=random_seed,
                                                     )

        # `layer_3` is now reshaped into `[n_steps, batch_size, 2*n_cell_dim]`,
        # as the LSTM BRNN expects its input to be of shape `[max_time, batch_size, input_size]`.
        #layer_3 = tf.reshape(layer_3, [-1, batch_x_shape[0], n_hidden_3])
        layer_2 = tf.reshape(layer_2, [-1, batch_x_shape[0], n_hidden_3])

        # Now we feed `layer_3` into the LSTM BRNN cell and obtain the LSTM BRNN output.
        outputs_LSTM1, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                                 cell_bw=lstm_bw_cell,
                                                                 inputs=layer_2,      #inputs=layer_3
                                                                 dtype=tf.float32,
                                                                 time_major=True,
                                                                 sequence_length=seq_length)

        tf.summary.histogram("activations", outputs_LSTM1)
        ##########################################################################
        #outputs_LSTM1=tf.concat(outputs_LSTM1,2)
        #tensor_LSTM1_output = tf.reshape(outputs_LSTM1, [-1, batch_x_shape[0], n_hidden_2])
        ##########################################################################

        #tensor_LSTM1_output=tf.convert_to_tensor(outputs_LSTM1)

         #Reshape outputs from two tensors each of shape [n_steps, batch_size, n_cell_dim]
         #to a single tensor of shape [n_steps*batch_size, 2*n_cell_dim]
        #outputs = tf.concat(outputs, 2)
        #outputs = tf.reshape(outputs, [-1, 2 * n_cell_dim])

        outputs_LSTM1 = tf.concat(outputs_LSTM1, 2)
        outputs_LSTM1 = tf.reshape(outputs_LSTM1, [-1, 2 * n_cell_dim])


    # 3rd layer
    with tf.name_scope('fc3'):
        b3 = variable_on_cpu('b3', [n_hidden_5], tf.random_normal_initializer(stddev=b3_stddev))
        h3 = variable_on_cpu('h3', [(2*n_cell_dim), n_hidden_5], tf.random_normal_initializer(stddev=h3_stddev))
        layer_3 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(outputs_LSTM1, h3), b3)), relu_clip)
        layer_3 = tf.nn.dropout(layer_3, (1.0 - dropout[2]))

        tf.summary.histogram("weights", h3)
        tf.summary.histogram("biases", b3)
        tf.summary.histogram("activations", layer_3)
    
    with tf.name_scope('fc3_2'):
        b3_2 = variable_on_cpu('b3_2', [n_hidden_3], tf.random_normal_initializer(stddev=b3_stddev))
        h3_2 = variable_on_cpu('h3_2', [n_hidden_5, n_hidden_3], tf.random_normal_initializer(stddev=h3_stddev))
        layer_3_2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_3, h3_2), b3_2)), relu_clip)
        layer_3_2 = tf.nn.dropout(layer_3_2, (1.0 - dropout[2]))

        tf.summary.histogram("weights", h3_2)
        tf.summary.histogram("biases", b3_2)
        tf.summary.histogram("activations", layer_3_2)
     #Create the forward and backward LSTM units. Inputs have length `n_cell_dim`.
     #LSTM forget gate bias initialized at `1.0` (default), meaning less forgetting
     #at the beginning of training (remembers more previous info)

    with tf.name_scope('lstm_2'):
        # Forward direction cell:
        lstm_fw_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True,reuse=True)
        #lstm_fw_cell = tf.contrib.rnn.GRUCell(n_cell_dim,activation=tf.nn.relu)
        lstm_fw_cell_2 = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell_2,
                                                     input_keep_prob=1.0 - dropout[3],
                                                     output_keep_prob=1.0 - dropout[3],
                                                     # seed=random_seed,
                                                     )
        # Backward direction cell:
        lstm_bw_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True,reuse=True)
        #lstm_bw_cell = tf.contrib.rnn.GRUCell(n_cell_dim,activation=tf.nn.relu)
        lstm_bw_cell_2 = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell_2,
                                                     input_keep_prob=1.0 - dropout[4],
                                                     output_keep_prob=1.0 - dropout[4],
                                                     # seed=random_seed,
                                                     )

        # `layer_3` is now reshaped into `[n_steps, batch_size, 2*n_cell_dim]`,
        # as the LSTM BRNN expects its input to be of shape `[max_time, batch_size, input_size]`.
        layer_3_2 = tf.reshape(layer_3_2, [-1, batch_x_shape[0], n_hidden_3])
        #layer_2 = tf.reshape(layer_2, [-1, batch_x_shape[0], n_hidden_2])
        # Now we feed `layer_3` into the LSTM BRNN cell and obtain the LSTM BRNN output.
        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell_2,
                                                                 cell_bw=lstm_bw_cell_2,
                                                                 inputs=layer_3_2,      #inputs=layer_3
                                                                 dtype=tf.float32,
                                                                 time_major=True,
                                                                 sequence_length=seq_length)

        tf.summary.histogram("activations", outputs)

        # Reshape outputs from two tensors each of shape [n_steps, batch_size, n_cell_dim]
        # to a single tensor of shape [n_steps*batch_size, 2*n_cell_dim]
        outputs = tf.concat(outputs, 2)
        outputs = tf.reshape(outputs, [-1, 2 * n_cell_dim])

    with tf.name_scope('fc5'):
        # Now we feed `outputs` to the fifth hidden layer with clipped RELU activation and dropout
        b5 = variable_on_cpu('b5', [n_hidden_5], tf.random_normal_initializer(stddev=b5_stddev))
        h5 = variable_on_cpu('h5', [(2 * n_cell_dim), n_hidden_5], tf.random_normal_initializer(stddev=h5_stddev))
        layer_5 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(outputs, h5), b5)), relu_clip)
        layer_5 = tf.nn.dropout(layer_5, (1.0 - dropout[5]))

        tf.summary.histogram("weights", h5)
        tf.summary.histogram("biases", b5)
        tf.summary.histogram("activations", layer_5)

    with tf.name_scope('fc6'):
        # Now we apply the weight matrix `h6` and bias `b6` to the output of `layer_5`
        # creating `n_classes` dimensional vectors, the logits.
        b6 = variable_on_cpu('b6', [n_hidden_6], tf.random_normal_initializer(stddev=b6_stddev))
        h6 = variable_on_cpu('h6', [n_hidden_5, n_hidden_6], tf.random_normal_initializer(stddev=h6_stddev))
        layer_6 = tf.add(tf.matmul(layer_5, h6), b6)

        tf.summary.histogram("weights", h6)
        tf.summary.histogram("biases", b6)
        tf.summary.histogram("activations", layer_6)

    # Finally we reshape layer_6 from a tensor of shape [n_steps*batch_size, n_hidden_6]
    # to the slightly more useful shape [n_steps, batch_size, n_hidden_6].
    # Note, that this differs from the input in that it is time-major.
    layer_6 = tf.reshape(layer_6, [-1, batch_x_shape[0], n_hidden_6])

    summary_op = tf.summary.merge_all()

    # Output shape: [n_steps, batch_size, n_hidden_6]
    return layer_6, summary_op



def BiRNN_V2(conf_path, batch_x, seq_length, n_input, n_context):

    parser = ConfigParser(os.environ)
    parser.read(conf_path)

    dropout = [float(x) for x in parser.get('birnn_V2', 'dropout_rates').split(',')]
    relu_clip = parser.getint('birnn_V2', 'relu_clip')

    b1_stddev = parser.getfloat('birnn_V2', 'b1_stddev')
    h1_stddev = parser.getfloat('birnn_V2', 'h1_stddev')
    b2_stddev = parser.getfloat('birnn_V2', 'b2_stddev')
    h2_stddev = parser.getfloat('birnn_V2', 'h2_stddev')
    b3_stddev = parser.getfloat('birnn_V2', 'b3_stddev')
    h3_stddev = parser.getfloat('birnn_V2', 'h3_stddev')
    b5_stddev = parser.getfloat('birnn_V2', 'b5_stddev')
    h5_stddev = parser.getfloat('birnn_V2', 'h5_stddev')
    b6_stddev = parser.getfloat('birnn_V2', 'b6_stddev')
    h6_stddev = parser.getfloat('birnn_V2', 'h6_stddev')

    n_hidden_1 = parser.getint('birnn_V2', 'n_hidden_1')
    n_hidden_2 = parser.getint('birnn_V2', 'n_hidden_2')
    n_hidden_5 = parser.getint('birnn_V2', 'n_hidden_5')
    n_cell_dim = parser.getint('birnn_V2', 'n_cell_dim')

    n_hidden_3 = int(eval(parser.get('birnn_V2', 'n_hidden_3')))
    n_hidden_6 = parser.getint('birnn_V2', 'n_hidden_6')

    # Input shape: [batch_size, n_steps, n_input + 2*n_input*n_context]
    batch_x_shape = tf.shape(batch_x)

    # Reshaping `batch_x` to a tensor with shape `[n_steps*batch_size, n_input + 2*n_input*n_context]`.
    # This is done to prepare the batch for input into the first layer which expects a tensor of rank `2`.

    # Permute n_steps and batch_size
    batch_x = tf.transpose(batch_x, [1, 0, 2])
    # Reshape to prepare input for first layer
    batch_x = tf.reshape(batch_x,
                         [-1, n_input + 2 * n_input * n_context])  # (n_steps*batch_size, n_input + 2*n_input*n_context)

    # The next three blocks will pass `batch_x` through three hidden layers with
    # clipped RELU activation and dropout.

    # 1st layer
    with tf.name_scope('fc1'):
        b1 = variable_on_cpu('b1', [n_hidden_1], tf.random_normal_initializer(stddev=b1_stddev))
        h1 = variable_on_cpu('h1', [n_input + 2 * n_input * n_context, n_hidden_1],
                             tf.random_normal_initializer(stddev=h1_stddev))
        layer_1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(batch_x, h1), b1)), relu_clip)         # clipped relu op
        layer_1 = tf.nn.dropout(layer_1, (1.0 - dropout[0]))

        tf.summary.histogram("weights", h1)
        tf.summary.histogram("biases", b1)
        tf.summary.histogram("activations", layer_1)

    # 2nd layer
    with tf.name_scope('fc2'):
        b2 = variable_on_cpu('b2', [n_hidden_3], tf.random_normal_initializer(stddev=b2_stddev))
        h2 = variable_on_cpu('h2', [n_hidden_1, n_hidden_3], tf.random_normal_initializer(stddev=h2_stddev))
        layer_2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_1, h2), b2)), relu_clip)
        layer_2 = tf.nn.dropout(layer_2, (1.0 - dropout[1]))

        tf.summary.histogram("weights", h2)
        tf.summary.histogram("biases", b2)
        tf.summary.histogram("activations", layer_2)

    ## 3rd layer
    #with tf.name_scope('fc3'):
    #    b3 = variable_on_cpu('b3', [n_hidden_3], tf.random_normal_initializer(stddev=b3_stddev))
    #    h3 = variable_on_cpu('h3', [n_hidden_2, n_hidden_3], tf.random_normal_initializer(stddev=h3_stddev))
    #    layer_3 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_2, h3), b3)), relu_clip)
    #    layer_3 = tf.nn.dropout(layer_3, (1.0 - dropout[2]))

    #    tf.summary.histogram("weights", h3)
    #    tf.summary.histogram("biases", b3)
    #    tf.summary.histogram("activations", layer_3)

    # Create the forward and backward LSTM units. Inputs have length `n_cell_dim`.
    # LSTM forget gate bias initialized at `1.0` (default), meaning less forgetting
    # at the beginning of training (remembers more previous info)
    with tf.name_scope('lstm'):
        # Forward direction cell:
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True)
        #lstm_fw_cell = tf.contrib.rnn.GRUCell(n_cell_dim,activation=tf.nn.relu)
        lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell,
                                                     input_keep_prob=1.0 - dropout[3],
                                                     output_keep_prob=1.0 - dropout[3],
                                                     # seed=random_seed,
                                                     )
        # Backward direction cell:
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True)
        #lstm_bw_cell = tf.contrib.rnn.GRUCell(n_cell_dim,activation=tf.nn.relu)
        lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell,
                                                     input_keep_prob=1.0 - dropout[4],
                                                     output_keep_prob=1.0 - dropout[4],
                                                     # seed=random_seed,
                                                     )

        # `layer_3` is now reshaped into `[n_steps, batch_size, 2*n_cell_dim]`,
        # as the LSTM BRNN expects its input to be of shape `[max_time, batch_size, input_size]`.
        #layer_3 = tf.reshape(layer_3, [-1, batch_x_shape[0], n_hidden_3])
        layer_2 = tf.reshape(layer_2, [-1, batch_x_shape[0], n_hidden_3])

        # Now we feed `layer_3` into the LSTM BRNN cell and obtain the LSTM BRNN output.
        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                                 cell_bw=lstm_bw_cell,
                                                                 inputs=layer_2,      #inputs=layer_3
                                                                 dtype=tf.float32,
                                                                 time_major=True,
                                                                 sequence_length=seq_length)

        tf.summary.histogram("activations", outputs)

        # Reshape outputs from two tensors each of shape [n_steps, batch_size, n_cell_dim]
        # to a single tensor of shape [n_steps*batch_size, 2*n_cell_dim]
        outputs = tf.concat(outputs, 2)
        outputs = tf.reshape(outputs, [-1, 2 * n_cell_dim])

    with tf.name_scope('fc4'):
        # Now we feed `outputs` to the fifth hidden layer with clipped RELU activation and dropout
        b5 = variable_on_cpu('b5', [n_hidden_5], tf.random_normal_initializer(stddev=b5_stddev))
        h5 = variable_on_cpu('h5', [(2 * n_cell_dim), n_hidden_5], tf.random_normal_initializer(stddev=h5_stddev))
        layer_5 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(outputs, h5), b5)), relu_clip)
        layer_5 = tf.nn.dropout(layer_5, (1.0 - dropout[5]))

        tf.summary.histogram("weights", h5)
        tf.summary.histogram("biases", b5)
        tf.summary.histogram("activations", layer_5)

    with tf.name_scope('fc5'):
        # Now we apply the weight matrix `h6` and bias `b6` to the output of `layer_5`
        # creating `n_classes` dimensional vectors, the logits.
        b6 = variable_on_cpu('b6', [n_hidden_6], tf.random_normal_initializer(stddev=b6_stddev))
        h6 = variable_on_cpu('h6', [n_hidden_5, n_hidden_6], tf.random_normal_initializer(stddev=h6_stddev))
        layer_6 = tf.add(tf.matmul(layer_5, h6), b6)

        tf.summary.histogram("weights", h6)
        tf.summary.histogram("biases", b6)
        tf.summary.histogram("activations", layer_6)

    # Finally we reshape layer_6 from a tensor of shape [n_steps*batch_size, n_hidden_6]
    # to the slightly more useful shape [n_steps, batch_size, n_hidden_6].
    # Note, that this differs from the input in that it is time-major.
    layer_6 = tf.reshape(layer_6, [-1, batch_x_shape[0], n_hidden_6])

    summary_op = tf.summary.merge_all()

    # Output shape: [n_steps, batch_size, n_hidden_6]
    return layer_6, summary_op

def BiRNN(conf_path, batch_x, seq_length, n_input, n_context):
    """
    This function was initially based on open source code from Mozilla DeepSpeech:
    https://github.com/mozilla/DeepSpeech/blob/master/DeepSpeech.py

    # This Source Code Form is subject to the terms of the Mozilla Public
    # License, v. 2.0. If a copy of the MPL was not distributed with this
    # file, You can obtain one at http://mozilla.org/MPL/2.0/.
    """
    parser = ConfigParser(os.environ)
    parser.read(conf_path)

    dropout = [float(x) for x in parser.get('birnn', 'dropout_rates').split(',')]
    relu_clip = parser.getint('birnn', 'relu_clip')

    b1_stddev = parser.getfloat('birnn', 'b1_stddev')
    h1_stddev = parser.getfloat('birnn', 'h1_stddev')
    b2_stddev = parser.getfloat('birnn', 'b2_stddev')
    h2_stddev = parser.getfloat('birnn', 'h2_stddev')
    b3_stddev = parser.getfloat('birnn', 'b3_stddev')
    h3_stddev = parser.getfloat('birnn', 'h3_stddev')
    b5_stddev = parser.getfloat('birnn', 'b5_stddev')
    h5_stddev = parser.getfloat('birnn', 'h5_stddev')
    b6_stddev = parser.getfloat('birnn', 'b6_stddev')
    h6_stddev = parser.getfloat('birnn', 'h6_stddev')

    n_hidden_1 = parser.getint('birnn', 'n_hidden_1')
    n_hidden_2 = parser.getint('birnn', 'n_hidden_2')
    n_hidden_5 = parser.getint('birnn', 'n_hidden_5')
    n_cell_dim = parser.getint('birnn', 'n_cell_dim')

    n_hidden_3 = int(eval(parser.get('birnn', 'n_hidden_3')))
    n_hidden_6 = parser.getint('birnn', 'n_hidden_6')

    # Input shape: [batch_size, n_steps, n_input + 2*n_input*n_context]
    batch_x_shape = tf.shape(batch_x)

    # Reshaping `batch_x` to a tensor with shape `[n_steps*batch_size, n_input + 2*n_input*n_context]`.
    # This is done to prepare the batch for input into the first layer which expects a tensor of rank `2`.

    # Permute n_steps and batch_size
    batch_x = tf.transpose(batch_x, [1, 0, 2])
    # Reshape to prepare input for first layer
    batch_x = tf.reshape(batch_x,
                         [-1, n_input + 2 * n_input * n_context])  # (n_steps*batch_size, n_input + 2*n_input*n_context)

    # The next three blocks will pass `batch_x` through three hidden layers with
    # clipped RELU activation and dropout.

    # 1st layer
    with tf.name_scope('fc1'):
        b1 = variable_on_cpu('b1', [n_hidden_1], tf.random_normal_initializer(stddev=b1_stddev))
        h1 = variable_on_cpu('h1', [n_input + 2 * n_input * n_context, n_hidden_1],
                             tf.random_normal_initializer(stddev=h1_stddev))
        layer_1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(batch_x, h1), b1)), relu_clip)         # clipped relu op
        layer_1 = tf.nn.dropout(layer_1, (1.0 - dropout[0]))

        tf.summary.histogram("weights", h1)
        tf.summary.histogram("biases", b1)
        tf.summary.histogram("activations", layer_1)

    # 2nd layer
    with tf.name_scope('fc2'):
        b2 = variable_on_cpu('b2', [n_hidden_2], tf.random_normal_initializer(stddev=b2_stddev))
        h2 = variable_on_cpu('h2', [n_hidden_1, n_hidden_2], tf.random_normal_initializer(stddev=h2_stddev))
        layer_2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_1, h2), b2)), relu_clip)
        layer_2 = tf.nn.dropout(layer_2, (1.0 - dropout[1]))

        tf.summary.histogram("weights", h2)
        tf.summary.histogram("biases", b2)
        tf.summary.histogram("activations", layer_2)

    # 3rd layer
    with tf.name_scope('fc3'):
        b3 = variable_on_cpu('b3', [n_hidden_3], tf.random_normal_initializer(stddev=b3_stddev))
        h3 = variable_on_cpu('h3', [n_hidden_2, n_hidden_3], tf.random_normal_initializer(stddev=h3_stddev))
        layer_3 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_2, h3), b3)), relu_clip)
        layer_3 = tf.nn.dropout(layer_3, (1.0 - dropout[2]))

        tf.summary.histogram("weights", h3)
        tf.summary.histogram("biases", b3)
        tf.summary.histogram("activations", layer_3)

    # Create the forward and backward LSTM units. Inputs have length `n_cell_dim`.
    # LSTM forget gate bias initialized at `1.0` (default), meaning less forgetting
    # at the beginning of training (remembers more previous info)
    with tf.name_scope('lstm'):
        # Forward direction cell:
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True)
        #lstm_fw_cell = tf.contrib.rnn.GRUCell(n_cell_dim,activation=tf.nn.relu)
        lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell,
                                                     input_keep_prob=1.0 - dropout[3],
                                                     output_keep_prob=1.0 - dropout[3],
                                                     # seed=random_seed,
                                                     )
        # Backward direction cell:
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True)
        #lstm_bw_cell = tf.contrib.rnn.GRUCell(n_cell_dim,activation=tf.nn.relu)
        lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell,
                                                     input_keep_prob=1.0 - dropout[4],
                                                     output_keep_prob=1.0 - dropout[4],
                                                     # seed=random_seed,
                                                     )

        # `layer_3` is now reshaped into `[n_steps, batch_size, 2*n_cell_dim]`,
        # as the LSTM BRNN expects its input to be of shape `[max_time, batch_size, input_size]`.
        layer_3 = tf.reshape(layer_3, [-1, batch_x_shape[0], n_hidden_3])

        # Now we feed `layer_3` into the LSTM BRNN cell and obtain the LSTM BRNN output.
        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                                 cell_bw=lstm_bw_cell,
                                                                 inputs=layer_3,
                                                                 dtype=tf.float32,
                                                                 time_major=True,
                                                                 sequence_length=seq_length)

        tf.summary.histogram("activations", outputs)

        # Reshape outputs from two tensors each of shape [n_steps, batch_size, n_cell_dim]
        # to a single tensor of shape [n_steps*batch_size, 2*n_cell_dim]
        outputs = tf.concat(outputs, 2)
        outputs = tf.reshape(outputs, [-1, 2 * n_cell_dim])

    with tf.name_scope('fc5'):
        # Now we feed `outputs` to the fifth hidden layer with clipped RELU activation and dropout
        b5 = variable_on_cpu('b5', [n_hidden_5], tf.random_normal_initializer(stddev=b5_stddev))
        h5 = variable_on_cpu('h5', [(2 * n_cell_dim), n_hidden_5], tf.random_normal_initializer(stddev=h5_stddev))
        layer_5 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(outputs, h5), b5)), relu_clip)
        layer_5 = tf.nn.dropout(layer_5, (1.0 - dropout[5]))

        tf.summary.histogram("weights", h5)
        tf.summary.histogram("biases", b5)
        tf.summary.histogram("activations", layer_5)

    with tf.name_scope('fc6'):
        # Now we apply the weight matrix `h6` and bias `b6` to the output of `layer_5`
        # creating `n_classes` dimensional vectors, the logits.
        b6 = variable_on_cpu('b6', [n_hidden_6], tf.random_normal_initializer(stddev=b6_stddev))
        h6 = variable_on_cpu('h6', [n_hidden_5, n_hidden_6], tf.random_normal_initializer(stddev=h6_stddev))
        layer_6 = tf.add(tf.matmul(layer_5, h6), b6)

        tf.summary.histogram("weights", h6)
        tf.summary.histogram("biases", b6)
        tf.summary.histogram("activations", layer_6)

    # Finally we reshape layer_6 from a tensor of shape [n_steps*batch_size, n_hidden_6]
    # to the slightly more useful shape [n_steps, batch_size, n_hidden_6].
    # Note, that this differs from the input in that it is time-major.
    layer_6 = tf.reshape(layer_6, [-1, batch_x_shape[0], n_hidden_6])

    summary_op = tf.summary.merge_all()

    # Output shape: [n_steps, batch_size, n_hidden_6]
    return layer_6, summary_op
