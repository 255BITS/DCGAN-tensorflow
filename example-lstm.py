from tensorflow.models.rnn import rnn_cell, seq2seq
import tensorflow as tf
import numpy as np
import tempfile

seq_length = 5
batch_size = 64

vocab_size = 7
embedding_dim = 50

memory_dim = 100
number_of_items = 1# this is always one because of the softmax cross entropy in the loss function

sess = tf.InteractiveSession()

enc_inp = [tf.placeholder(tf.float32, shape=(batch_size,number_of_items),
                              name="inp-%i" % t)
                                         for t in range(seq_length)]

labels = [tf.placeholder(tf.float32, shape=(batch_size,number_of_items),
                            name="labels-%i" % t)
                                      for t in range(seq_length)]

weights = [tf.ones_like(labels_t, dtype=tf.float32)
                   for labels_t in labels]

# Decoder input: prepend some "GO" token and drop the final
# token of the encoder input
dec_inp = ([tf.zeros_like(enc_inp[0], dtype=np.float32, name="GO")]
                   + enc_inp[:-1])

# Initial memory value for recurrence.
prev_mem = tf.zeros((batch_size, memory_dim))

cell = rnn_cell.BasicLSTMCell(memory_dim)


#enc_inp = np.tile(enc_inp, 2).tolist()
logits, state = seq2seq.basic_rnn_seq2seq(
        enc_inp, dec_inp, cell)#, vocab_size, vocab_size)

for i, inp in enumerate(enc_inp):
    print(i, inp)
print("logits", logits)
print('labels', labels)
loss = seq2seq.sequence_loss(logits, labels, weights)
summary_op = tf.scalar_summary("loss", loss)

square = tf.square(state)
sum = tf.reduce_sum(square)
magnitude = tf.sqrt(sum)
tf.scalar_summary("magnitude at t=1", magnitude)

learning_rate = 0.05
momentum = 0.9
optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
train_op = optimizer.minimize(loss)


logdir = tempfile.mkdtemp()
print(logdir)
summary_writer = tf.train.SummaryWriter(logdir, sess.graph_def)

sess.run(tf.initialize_all_variables())



def train_batch(batch_size):
    X = np.random.normal(0,0.5, (seq_length, batch_size, number_of_items))
    Y = X[:]
    
    feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}
    feed_dict.update({labels[t]: np.roll(np.array(Y[t]), -1) for t in range(seq_length)})

    _, loss_t, summary = sess.run([train_op, loss, summary_op], feed_dict)
    return loss_t, summary


for t in range(500):
    loss_t, summary = train_batch(batch_size)
    summary_writer.add_summary(summary, t)
summary_writer.flush()


X_batch = np.random.normal(0,0.5,(seq_length, batch_size, number_of_items))

feed_dict = {enc_inp[t]: X_batch[t] for t in range(seq_length)}
logits_batch = sess.run(logits, feed_dict)


