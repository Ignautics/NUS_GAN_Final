from __future__ import division
import os
import time
from glob import glob
from PIL import Image
import scipy.io as sio

from ops import *
from utils import *


import matplotlib.image as mtim
def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class NUS_GAN(object):
    def __init__(self, sess, input_height=218, input_width=178, crop=True, batch_size=50, sample_num=50,
                 output_height=218, output_width=178, image_num = 50, y_dim=40, z_dim=100, gf_dim=64,
                 L1_lambda=100, L2_lambda1=50, L2_lambda2=50, L2_lambda3=50,
                 df_dim=64, gfc_dim=1024, dfc_dim=1024, dataset_name='celeb', input_fname_pattern='*.jpg',
                 checkpoint_dir=None, train_num = 12, labels = np.ones(40, np.int),
                 c_dim = 3, sample_dir=None, data_dir='./data', train = False):
        """

        Args:
          sess: TensorFlow session
          batch_size: The size of batch. Should be specified before training.
          y_dim: (optional) Dimension of dim for y. [None]
          z_dim: (optional) Dimension of dim for Z. [100]
          gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
          df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
          gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
          dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
          c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.train = train
        self.brain_label = labels
        self.train_number = train_num
        self.image_num = image_num
        self.sess = sess
        self.crop = crop

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.L1_lambda = L1_lambda
        self.L2_lambda1 = L2_lambda1
        self.L2_lambda2 = L2_lambda2
        self.L2_lambda3 = L2_lambda3

        self.y_dim = y_dim
        self.z_dim = z_dim
        
        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir

        self.c_dim = c_dim

        print("start load image")
        if (train):
            self.train_x = self.load_image()
            self.train_y = self.load_label(len(self.train_x))
            print("load label ok")

        self.grayscale = (self.c_dim == 1)


        self.build_model()
        print("build model ok")

    def build_model(self):
        
        if self.y_dim:
            self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
        else:
            self.y = None

        if self.crop:
            image_dims = [self.output_height, self.output_width, self.c_dim]
        else:
            image_dims = [self.input_height, self.input_width, self.c_dim]

        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_images')
        
        inputs = self.inputs
        

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        #self.z_sum = histogram_summary("z", self.z)

       
        self.G = self.generator(self.z, self.y)
        self.sampler = self.sampler(self.z, self.y)

        self.D_real_logits, self.y_real_logits = \
            self.discriminator(inputs, reuse=False) #batch_size*1
        self.D_fake_logits, self.y_fake_logits = \
            self.discriminator(self.G, reuse=True)

        self.D_real = tf.nn.sigmoid(self.D_real_logits)
        self.D_fake = tf.nn.sigmoid(self.D_fake_logits)
        
        self.y_real = tf.nn.sigmoid(self.y_real_logits)
        
        self.y_fake = tf.nn.sigmoid(self.y_fake_logits)
        
        
        #self.d_real_sum = histogram_summary("d_real", self.D_real)
        #self.d_fake_sum = histogram_summary("d_fake", self.D_fake)
        
        #self.y_real_sum = histogram_summary("d_c2_real", self.y_real)
        #self.y_fake_sum = histogram_summary("d_c2_fake", self.y_fake)
        
        #self.G_sum = image_summary("G", self.G)

        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)
        #g_loss
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits, labels=tf.ones_like(self.D_fake))) \
                        + self.L1_lambda * tf.reduce_mean(tf.abs(inputs - self.G))

        #d_loss
        self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_real_logits, tf.ones_like(self.D_real)))
        self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_fake_logits, tf.zeros_like(self.D_fake)))
        
        self.y_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.y_real_logits, self.y))
        self.y_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.y_fake_logits, self.y))
        
        
        self.d_loss = self.d_loss_real + self.d_loss_fake + \
                      self.L2_lambda2 * (self.y_loss_real + self.y_loss_fake) #class weight

        #summary

        #self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        #self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
        
        #self.y_loss_real_sum = scalar_summary("d_c2_loss_real", self.y_loss_real)
        #self.y_loss_fake_sum = scalar_summary("d_c2_loss_fake", self.y_loss_fake)
        
        
        #self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        #self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        
        self.saver = tf.train.Saver()
    def train(self, config):
        print("start train")
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        #self.g_sum = tf.summary.merge([self.d_fake_sum, self.y_fake_sum,
                                       #self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        #self.d_sum = tf.summary.merge([self.d_real_sum, self.y_real_sum,
                                       #self.d_loss_real_sum, self.y_loss_real_sum, self.d_loss_sum])
        #self.writer = SummaryWriter("./logs", self.sess.graph)


        #order = np.concatenate((np.arange(210, 4200, 1),np.arange(4480, 9520, 1)),axis=0)
        order = np.arange(0, len(self.train_x), 1)
        train_x = self.train_x
        train_y = self.train_y

        #test_order = np.concatenate((np.arange(0, 210, 10), np.arange(4200, 4480, 10)), axis=0)
        self.sample_inputs = self.train_x[0:self.sample_num]
        self.sample_y = self.train_y[0:self.sample_num]
        self.sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))

        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")


        print("step trianing")
        for epoch in range(config.epoch):
            batch_idxs = min(len(train_x), config.train_size) // config.batch_size

            for idx in range(0, batch_idxs):
                batch_images = train_x[idx*config.batch_size:(idx+1)*config.batch_size]
                batch_y = train_y[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

                #print("epoch begin training")
                if self.y_dim:
                    # Update D network
                    #print("updating D")
                    _ = self.sess.run([d_optim],
                                                   feed_dict={
                                                       self.inputs:batch_images,
                                                       self.z:batch_z,
                                                       self.y:batch_y
                                                   })

                    #print("updated D")
                    #self.writer.add_summary(summary_str, counter)
                    # Update G network
                    #print("updating G")
                    _  = self.sess.run([g_optim],
                                                   feed_dict={
                                                       self.inputs:batch_images,
                                                       self.z:batch_z,
                                                       self.y:batch_y
                                                   })
                    #self.writer.add_summary(summary_str, counter)

                    #print("updated G")
                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                    #print("updating G again")
                    _  = self.sess.run([g_optim],
                                                   feed_dict={
                                                       self.inputs:batch_images,
                                                       self.z:batch_z,
                                                       self.y:batch_y
                                                   })
                    #self.writer.add_summary(summary_str, counter)
                    #print("updated G again")
                    errD_fake = self.d_loss_fake.eval({self.inputs:batch_images,
                                                       self.z:batch_z,
                                                       self.y:batch_y})
                    errD_real = self.d_loss_real.eval({self.inputs:batch_images,
                                                       self.z:batch_z,
                                                       self.y:batch_y})
                    errD = self.d_loss.eval({self.inputs:batch_images, 
                                                 self.z: batch_z, 
                                                 self.y: batch_y})


                    errG = self.g_loss.eval({self.inputs:batch_images,
                                                       self.z:batch_z,
                                                       self.y:batch_y})
                else:
                    print("ERROR: invalid y_dim")

                counter += 1
                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"
                      % (epoch, config.epoch, idx, batch_idxs,
                         time.time() - start_time, errD, errG))

                if idx == batch_idxs - 1:
                    if self.y_dim:
                        samples, d_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.g_loss],
                            feed_dict={
                                self.inputs: self.sample_inputs,
                                self.z: self.sample_z,
                                self.y: self.sample_y
                            }
                        )
                        save_images(samples, self.sample_inputs, image_manifold_size(samples.shape[0]),
                                    './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)

            random.shuffle(order)
            train_x = self.train_x[order]
            train_y = self.train_y[order]

    def discriminator(self, image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
	
            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            # h0 is (128 x 128 x self.df_dim)
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
            # h1 is (64 x 64 x self.df_dim*2)
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
            # h2 is (32x 32 x self.df_dim*4)
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, d_h=1, d_w=1, name='d_h3_conv')))
            # h3 is (16 x 16 x self.df_dim*8)
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
            #h4_1 = linear(tf.reshape(h3, [self.batch_size, -1]), self.c1_dim, 'c1_h3_lin')
            h4_2 = linear(tf.reshape(h3, [self.batch_size, -1]), self.y_dim, 'c2_h3_lin')
            #h4_3 = linear(tf.reshape(h3, [self.batch_size, -1]), self.c3_dim, 'c3_h3_lin')

            return h4, h4_2

    def generator(self, z, y=None):
        with tf.variable_scope("generator") as scope:
        
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            z = concat([z, y], 1)  # add condition y to z, z is 50*(100+40)

            z0 = linear(z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin')

            self.h0 = tf.reshape(
                z0, [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(self.h0))

            self.h1, self.h1_w, self.h1_b = deconv2d(
                h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))

            h2, self.h2_w, self.h2_b = deconv2d(
                h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3, self.h3_w, self.h3_b = deconv2d(
                h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4, self.h4_w, self.h4_b = deconv2d(
                h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

            return tf.nn.tanh(h4)

    def sampler(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            z = concat([z, y], 1)  # add condition y to z

            z0 = linear(z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin')

            self.h0 = tf.reshape(
                z0, [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(self.h0))

            self.h1, self.h1_w, self.h1_b = deconv2d(
                h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))

            h2, self.h2_w, self.h2_b = deconv2d(
                h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3, self.h3_w, self.h3_b = deconv2d(
                h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4, self.h4_w, self.h4_b = deconv2d(
                h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

            return tf.nn.tanh(h4)

    def load_label(self, length):
        
        train_y = np.zeros((length, self.y_dim), dtype=np.float)
        
        cnt = 0
        for line in open("./data/label.txt", "r"):
            train_y[cnt] = np.array([np.float(ch) for ch in line[0:-1]])
            cnt = cnt + 1

        return train_y

    def load_image(self):
        data_path = os.path.join("./data", self.dataset_name)
        image = glob((os.path.join(data_path, '*.jpg')))
        image.sort()
        self.y_dim = 40

        img_B = np.zeros((len(image), self.output_height, self.output_width, self.c_dim), dtype=np.float)
        for im in range(len(image)):
            #print(image[im])
            tmp_B = np.array(mtim.imread(image[im]), dtype=float)
            tmp_B = np.resize(tmp_B, (self.output_height, self.output_width, self.c_dim))#scipy.misc.imresize(sour(image[im], False), [self.output_height, self.output_width, self.c_dim])
            #print("path:%s" % image[im])
            #print("tmp_B:", tmp_B)
            img_B[im] = np.array(tmp_B)/127.5 - 1.
            print("tdc tst %d" % im)
        #img_B = img_B.reshape(img_B.shape[0], img_B.shape[1], img_B.shape[2], 1)
        print("for circulation is over")
        order = np.arange(0, img_B.shape[0], 1)
        print("for tdc test 0")
        random.shuffle(order)
        print("for tdc test 1")
        #wrg_B = img_B[order]

        print("load_image ok")
        return img_B

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(self.dataset_name, self.batch_size, self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print("checkpoint_dir: %s ckpt_name: %s" % (checkpoint_dir, ckpt_name))
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
