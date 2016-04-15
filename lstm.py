import tensorflow as tf
from tensorflow.models.rnn import rnn_cell, seq2seq, rnn
from ops import linear,lrelu
import numpy as np


def discriminator(input, vocab_size, name="lstm_discriminator"):
     with tf.variable_scope(name):
        cell_input = [input]
        zeros = [tf.zeros_like(input)]
        memory = 128
        cell = rnn_cell.BasicLSTMCell(memory)
        stacked_cell = rnn_cell.MultiRNNCell([cell]*2)
        cell_input = [input]
        outputs, state = rnn.rnn(stacked_cell, cell_input, dtype=tf.float32)
        print("Discrim output", outputs)
        output = outputs[0]
        #w = tf.get_variable('d_softmax_w', [memory, vocab_size], dtype=tf.float32, initializer=tf.truncated_normal_initializer(0, 0.1))
        #b = tf.get_variable('d_softmax_b', [vocab_size], dtype=tf.float32, initializer=tf.constant_initializer(0))
        return 1- tf.reduce_max(tf.square(tf.nn.softmax(output)), 1)
 
def generator(input, name='lstm_generator', split=5, softmax=True):
     with tf.variable_scope(name):
        batch_size = input.get_shape()[0]
        vocab_size = input.get_shape()[1]
        cell_input = [input]#tf.split(1, split, input)
        zeros = [tf.zeros_like(input)]
        memory = 512
        cell = rnn_cell.BasicLSTMCell(memory)
        stacked_cell = rnn_cell.MultiRNNCell([cell]*1)
        outputs, state = rnn.rnn(stacked_cell, cell_input, dtype=tf.float32)
        print(outputs)
        output=outputs[0]
        print("output is", output)

        return output
        w = tf.get_variable('g_softmax_w', [memory, vocab_size], dtype=tf.float32, initializer=tf.truncated_normal_initializer(0, 0.1))
        b = tf.get_variable('g_softmax_b', [vocab_size], dtype=tf.float32, initializer=tf.constant_initializer(0))
        if softmax:
            return tf.nn.softmax(tf.matmul(output, w)+b)
        else:
            return tf.matmul(output, w)+b
       
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

