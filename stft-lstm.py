from tensorflow.models.rnn import rnn_cell, seq2seq
import tensorflow as tf
import numpy as np
import tempfile
import tensorflow_wav
import os

seq_length = 64
batch_size = 40

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



def train_batch(X, Y, batch_size):
    #X = np.random.normal(0,0.5, (seq_length, batch_size, number_of_items))
    
    feed_dict = {enc_inp[t]: X[:,t,:] for t in range(seq_length)}
    feed_dict.update({labels[t]: np.array(Y[:,t,:]) for t in range(seq_length)})

    _, loss_t, summary = sess.run([train_op, loss, summary_op], feed_dict)
    return loss_t, summary


def get_data(stft, roll):
    X = stft['data']
    X = X.reshape([-1])
    max_items = int(X.shape[0]/batch_size/seq_length)*batch_size*seq_length
    X = X[:max_items]

    X=X.reshape([-1])
    #print("Rolling by ", roll)
    X = np.roll(X, roll*seq_length*batch_size)
    Y = np.roll(X[:], -seq_length)
    X = X[:batch_size*seq_length]
    Y = Y[:batch_size*seq_length]
    X=X.reshape([-1, seq_length, 1])
    Y=Y.reshape([-1, seq_length, 1])
    print(X[0][0])
    print(X[1][0])
    print(Y[0][0])
    #Y = X[:]
    return X,Y


saver = tf.train.Saver()
def save(sess, step):

    checkpoint_dir = 'checkpoint'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver.save(sess,
            checkpoint_dir+'/lstm',
            global_step=step)


stft = tensorflow_wav.get_stft("input.wav.stft")

#probs = tf.nn.softmax(logits)

for t in range(1000000):
    X, Y = get_data(stft, t)
    loss_t, summary = train_batch(X, Y, batch_size)
    summary_writer.add_summary(summary, t)
    print("Loss: ", loss_t)
    SAVE_COUNT = 3
    if(t % SAVE_COUNT == 2):
        print("Saving ...")
        save(sess, t)
        summary_writer.flush()
    X_batch = np.random.normal(0,0.5,(seq_length, batch_size, number_of_items))

    feed_dict = {enc_inp[t]: X_batch[t] for t in range(seq_length)}
    logits_batch = sess.run(logits, feed_dict)
    print("Generated probs of shape", np.shape(logits_batch))
    #print(logits_batch)





