import os
import time
from glob import glob
import tensorflow as tf

from ops import *
from utils import *

WAV_SIZE=64
WAV_HEIGHT=64
class DCGAN(object):
    def __init__(self, sess, wav_size=WAV_SIZE, is_crop=True,
                 batch_size=64, sample_size = 64, wav_shape=[WAV_SIZE, WAV_HEIGHT, 3],
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                 checkpoint_dir='checkpoint'):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen untis for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of wav color. [3]
        """
        self.sess = sess
        self.is_crop = is_crop
        self.batch_size = batch_size
        self.wav_size = wav_size
        self.sample_size = sample_size
        self.wav_shape = wav_shape

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = 3

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(batch_size, name='d_bn1')
        self.d_bn2 = batch_norm(batch_size, name='d_bn2')

        if not self.y_dim:
            self.d_bn3 = batch_norm(batch_size, name='d_bn3')

        self.g_bn0 = batch_norm(batch_size, name='g_bn0')
        self.g_bn1 = batch_norm(batch_size, name='g_bn1')
        self.g_bn2 = batch_norm(batch_size, name='g_bn2')

        if not self.y_dim:
            self.g_bn3 = batch_norm(batch_size, name='g_bn3')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):
        if self.y_dim:
            self.y= tf.placeholder(tf.float32, [None, self.y_dim], name='y')

        self.wavs = tf.placeholder(tf.float32, [self.batch_size] + self.wav_shape,
                                    name='real_wavs')
        self.sample_wavs= tf.placeholder(tf.float32, [self.sample_size] + self.wav_shape,
                                        name='sample_wavs')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim],
                                name='z')

        self.z_sum = tf.histogram_summary("z", self.z)

        self.G = self.generator(self.z)
        #self.G = self.generator_wav(self.z)
        print("G is", self.G.get_shape())
        self.D = self.discriminator(self.wavs)
        #self.D = self.discriminator_wav(self.wavs)

        self.sampler = self.sampler(self.z)
        #self.D_ = self.discriminator(self.G, reuse=True)
        self.D_ = self.discriminator_wav(self.G, reuse=True)

        self.d_sum = tf.histogram_summary("d", self.D)
        self.d__sum = tf.histogram_summary("d_", self.D_)

        self.d_loss_real = binary_cross_entropy_with_logits(tf.ones_like(self.D), self.D)
        self.d_loss_fake = binary_cross_entropy_with_logits(tf.zeros_like(self.D_), self.D_)
        self.g_loss = binary_cross_entropy_with_logits(tf.ones_like(self.D_), self.D_)

        self.d_loss_real_sum = tf.scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.scalar_summary("d_loss_fake", self.d_loss_fake)
                                                    
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = tf.scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        """Train DCGAN"""
        data = glob(os.path.join("./training", config.dataset, "*.wav"))
        #np.random.shuffle(data)

        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)

        self.saver = tf.train.Saver()
        self.g_sum = tf.merge_summary([self.z_sum, self.d__sum, 
            self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.merge_summary([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.train.SummaryWriter("./logs", self.sess.graph_def)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_size , self.z_dim))
        sample_file = data[0]
        sample =get_wav(sample_file, self.wav_size, is_crop=self.is_crop) #[get_wav(sample_file, self.wav_size, is_crop=self.is_crop) for sample_file in sample_files]
        sample_wavs = np.array(sample).astype(np.float32)

        counter = 1
        start_time = time.time()
        tf.initialize_all_variables().run()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        print('epoch', config.epoch)

        for epoch in range(config.epoch):
            batch_files = glob(os.path.join("./training", "*.wav"))

            def get_wav_content(files):
                for filee in files:
                    print("Yielding ", filee)
                    yield get_wav(filee, self.wav_size, is_crop=self.is_crop)

            #print(batch)
            idx=0
            batch_idxs=0
            for batch_item in get_wav_content(batch_files):
                max_items = int(len(batch_item)/64)*64
                batch_item = batch_item[:max_items]
                batch_wavs_multiple = batch_item.reshape([-1, 64, WAV_SIZE,WAV_HEIGHT,3])
                sample_wavs = sample_wavs[:max_items].reshape([-1, 64, WAV_SIZE,WAV_HEIGHT,3])
                batch_idxs+=1
                for i, batch_wavs in enumerate(batch_wavs_multiple):
                    if(len(batch_wavs)!=64):
                        break
                    idx+=1
                    batch_wavs = np.array(batch_wavs)
                    batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                                .astype(np.float32)

                    # Update D network
                    _, summary_str = self.sess.run([d_optim, self.d_sum],
                        feed_dict={ self.wavs: batch_wavs, self.z: batch_z })
                    self.writer.add_summary(summary_str, counter)

                    # Update G network
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                        feed_dict={ self.z: batch_z })
                    self.writer.add_summary(summary_str, counter)

                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                        feed_dict={ self.z: batch_z })
                    self.writer.add_summary(summary_str, counter)

                    errD_fake = self.d_loss_fake.eval({self.z: batch_z})
                    errD_real = self.d_loss_real.eval({self.wavs: batch_wavs})
                    errG = self.g_loss.eval({self.z: batch_z})

                    counter += 1
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                        % (epoch, idx, batch_idxs,
                            time.time() - start_time, errD_fake+errD_real, errG))

                    if np.mod(counter, 100) == 2:
                        samples, d_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.g_loss],
                            feed_dict={self.z: sample_z, self.wavs: sample_wavs[0]}
                        )
                        save_wav(samples, 64,
                                    './samples/train_%s_%s.png' % (epoch, idx))
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                    if np.mod(counter, 30) == 2:
                        self.save(config.checkpoint_dir, counter)


    def discriminator_wav(self, wav, reuse=False, y=None):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        if not self.y_dim:
            h0 = lrelu(conv2d(wav, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4)
        else:
            print("WTF")
            return None

    def discriminator(self, wav, reuse=False, y=None):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        if not self.y_dim:
            h0 = lrelu(conv2d(wav, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4)
        else:
            yb = tf.reshape(y, [None, 1, 1, self.y_dim])
            x = conv_cond_concat(wav, yb)

            h0 = lrelu(spatial_conv(x, self.c_dim + self.y_dim))
            h0 = conv_cond_concat(h0, yb)

            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim)))
            h1 = tf.reshape(h1, [h1.get_shape()[0], -1])
            h1 = tf.concat(1, [h1, y])

            h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
            h2 = tf.concat(1, [h2, y])
            print('shape', h2.get_shape())

            return tf.nn.sigmoid(linear(h2, 1, 'd_h3_lin'))

    def generator(self, z, y=None):
        if not self.y_dim:
            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*4*4, 'g_h0_lin', with_w=True)

            self.h0 = tf.reshape(self.z_, [-1, 4, 4, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(self.h0))

            self.h1, self.h1_w, self.h1_b = deconv2d(h0, 
                [self.batch_size, 8, 8, self.gf_dim*4], name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))

            h2, self.h2_w, self.h2_b = deconv2d(h1,
                [self.batch_size, 16, 16, self.gf_dim*2], name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3, self.h3_w, self.h3_b = deconv2d(h2,
                [self.batch_size, 32, 32, self.gf_dim*1], name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4, self.h4_w, self.h4_b = deconv2d(h3,
                [self.batch_size, WAV_SIZE, WAV_HEIGHT, 3], name='g_h4', with_w=True)

            return tf.nn.tanh(h4)
        else:
            yb = tf.reshape(y, [None, 1, 1, self.y_dim])
            z = tf.concat(1, [z, y])

            h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
            h0 = tf.concat(1, [h0, y])

            h1 = tf.nn.relu(self.g_bn1(linear(z, self.gf_dim*2*7*7, 'g_h1_lin')))
            h1 = tf.reshape(h1, [None, 7, 7, self.gf_dim * 2])
            h1 = conv_cond_concat(h1, yb)

            h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, self.gf_dim, name='g_h2')))
            h2 = conv_cond_concat(h2, yb)

            return tf.nn.sigmoid(deconv2d(h2, self.c_dim, name='g_h3'))

    def sampler(self, z, y=None):
        tf.get_variable_scope().reuse_variables()

        if not self.y_dim:
            # project `z` and reshape
            h0 = tf.reshape(linear(z, self.gf_dim*8*4*4, 'g_h0_lin'),
                            [-1, 4, 4, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(h0, train=False))

            h1 = deconv2d(h0, [self.batch_size, 8, 8, self.gf_dim*4], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1, train=False))

            h2 = deconv2d(h1, [self.batch_size, 16, 16, self.gf_dim*2], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2, train=False))

            h3 = deconv2d(h2, [self.batch_size, 32, 32, self.gf_dim*1], name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3, train=False))

            h4 = deconv2d(h3, [self.batch_size, 64, 64, 3], name='g_h4')

            return tf.nn.tanh(h4)
        else:
            yb = tf.reshape(y, [None, 1, 1, self.y_dim])
            z = tf.concat(1, [z, y])

            h0 = tf.nn.relu(self.bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
            h0 = tf.concat(1, [h0, y])

            h1 = tf.nn.relu(self.g_bn1(linear(z, self.gf_dim*2*7*7, 'g_h1_lin')))
            h1 = tf.reshape(h1, [None, 7, 7, self.gf_dim * 2])
            h1 = conv_cond_concat(h1, yb)

            h2 = tf.nn.relu(self.bn2(deconv2d(h1, self.gf_dim, name='g_h2')))
            h2 = conv_cond_concat(h2, yb)

            return tf.nn.sigmoid(deconv2d(h2, self.c_dim, name='g_h3'))

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("TRUE")
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            print("FALSE")
            return False
