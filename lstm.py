import tensorflow as tf
from tensorflow.models.rnn import rnn_cell, seq2seq, rnn
from ops import linear,lrelu
import numpy as np


LENGTH = 1024
WAVELONS = LENGTH//4//16
def discriminator(input, state, cell, memory=16, name="lstm_discriminator", reuse=None):
     with tf.variable_scope(name):
        print("REUSE", reuse)
        cell_input = tf.split(1, WAVELONS, input)
        states = [state]
        outputs = []
        i=0
        for inp in cell_input:
            with tf.variable_scope(name+'rnns'):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                output, new_state = cell(inp, states[-1])


                outputs.append(output)
                states.append(new_state)
                i+=1

        if(reuse):
           tf.get_variable_scope().reuse_variables()
        output_w = tf.get_variable("output_w", [memory, 1])
        output_b = tf.get_variable("output_b", [1])
        output = tf.reshape(tf.concat(1, outputs), [-1, memory])
        output = tf.nn.xw_plus_b(output, output_w, output_b)
        #return 1- tf.reduce_max(tf.square(tf.nn.softmax(output)), 1)
        return output, states[-1]
 
def z_gates(input, state, cell, memory=32, name='g_lstm_generator', softmax=True, batch_size=-1):
    with tf.variable_scope(name):
       input_dim = input.get_shape()[1]
       cell_input = tf.split(1, input_dim, input)
       states = [state]
       outputs = []
       i=0
       for inp in cell_input:
           with tf.variable_scope(name+'rnns'):
               if i > 0:
                   tf.get_variable_scope().reuse_variables()
               output, new_state = cell(inp, states[-1])


               outputs.append(output)
               states.append(new_state)
               i+=1

       output = tf.concat(1, outputs)
       output = tf.reshape(output, [batch_size, -1])
       output_w = tf.get_variable("output_w", [output.get_shape()[1], 256], initializer=tf.random_normal_initializer(stddev=0.003))
       output_b = tf.get_variable("output_b", [256])
       output = tf.nn.xw_plus_b(output, output_w, output_b)
       #output = tf.matmul(output, output_w)
       print("shape of output is ", output)
       return output, states[-1]
      
def seq2seq_graph(input, predicted):
    with tf.variable_scope("lstm_generator"):
        cell_input = [input]
        zeros = [tf.zeros_like(input)]
        vocab_size = output_size
        memory = 256
        print('input',input)
        cell = rnn_cell.GRUCell(memory)
        stacked_cell = rnn_cell.MultiRNNCell([cell]*2)
        logits, state = seq2seq.basic_rnn_seq2seq(cell_input, zeros, stacked_cell)

        labels = [predicted]
        weights = [tf.ones_like(labels_t, dtype=tf.float32)
                           for labels_t in labels]


        loss = seq2seq.sequence_loss(logits, labels, weights)
        logits_ = tf.concat(1, logits)
        print("logits_", logits_)
        # if it's a repeat, one of the memory cells should fire harsh
        #w = tf.get_variable('g_softmax_w', [memory, vocab_size], dtype=tf.float32, initializer=tf.truncated_normal_initializer(0, 0.1))
        #b = tf.get_variable('g_softmax_b', [vocab_size], dtype=tf.float32, initializer=tf.constant_initializer(0))
        #wx_b = tf.nn.xw_plus_b(logits_, w, b)
        #softmax = tf.square(tf.nn.softmax(wx_b))
        #print('wx_b', wx_b, softmax)
        return loss#softmax
        #is_repeat = tf.reduce_max(tf.square(tf.nn.softmax(wx_b)), 1)
        #print("Output shape is", output, state)

