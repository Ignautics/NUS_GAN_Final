"""
Some codes from https://github.com/Newmu/nus_gan_code
"""
from __future__ import division
import math
import json
import random
import os
import matplotlib.image as matimage
import pprint
import scipy.misc
from glob import glob
import numpy as np
from time import gmtime, strftime
from six.moves import xrange


import tensorflow as tf
#import tensorflow.contrib.slim as slim

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def show_all_variables():
  model_vars = tf.trainable_variables()
  #slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True, grayscale=False):
  image = imread(image_path, grayscale)
  return transform(image, input_height, input_width,
                   resize_height, resize_width, crop)

def save_images(images, ori, size, image_path):
  return imsave(inverse_transform(images), inverse_transform(ori), size, image_path)

def imread(path, grayscale = False):
  print("read paht:%s" % path)
  ans = np.array(mping.imread(path), dtype = float)
  return ans

def merge_images(images, size):
  return inverse_transform(images)

def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  if (images.shape[3] in (3,4)):
    c = images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img
  elif images.shape[3]==1:
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
    return img
  else:
    raise ValueError('in merge(images,size) images parameter '
                     'must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, ori, size, path):#add contrast(target image)
  image = np.squeeze(merge(images, size))
  ori = np.squeeze(merge(ori, size))
  concat = np.zeros((image.shape[0], image.shape[1]*2, 3))
  concat[:,image.shape[1]:] = ori
  concat[:,:image.shape[1]] = image
  return scipy.misc.imsave(path, concat)

def label2onehot(label, len, cls, times):  # cls:class_num, times:magnify_times
  rst = np.zeros((len, cls * times))
  for i in range(len):
    temp = onehot(label[i], cls)
    one = temp
    for j in range(times - 1):
      one = np.append(one, temp, axis=0)
    rst[i] = one
  return rst

def class2onehot(label, len, cls):  # cls:class_num
  rst = np.zeros((len, cls))
  for i in range(len):
    temp = onehot(label[i], cls)
    rst[i] = temp
  return rst

def onehot(label, len):
  onehot = np.zeros(len)
  onehot[int(label)] = 1
  return onehot

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width, 
              resize_height=64, resize_width=64, crop=True):
  if crop:
    cropped_image = center_crop(
      image, input_height, input_width, 
      resize_height, resize_width)
  else:
    cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
  return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
  return (images+1.)/2.

def to_json(output_path, *layers):
  with open(output_path, "w") as layer_f:
    lines = ""
    for w, b, bn in layers:
      layer_idx = w.name.split('/')[0].split('h')[1]

      B = b.eval()

      if "lin/" in w.name:
        W = w.eval()
        depth = W.shape[1]
      else:
        W = np.rollaxis(w.eval(), 2, 0)
        depth = W.shape[0]

      biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
      if bn != None:
        gamma = bn.gamma.eval()
        beta = bn.beta.eval()

        gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
        beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
      else:
        gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
        beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

      if "lin/" in w.name:
        fs = []
        for w in W.T:
          fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

        lines += """
          var layer_%s = {
            "layer_type": "fc", 
            "sy": 1, "sx": 1, 
            "out_sx": 1, "out_sy": 1,
            "stride": 1, "pad": 0,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
      else:
        fs = []
        for w_ in W:
          fs.append({"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

        lines += """
          var layer_%s = {
            "layer_type": "deconv", 
            "sy": 5, "sx": 5,
            "out_sx": %s, "out_sy": %s,
            "stride": 2, "pad": 1,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx, 2**(int(layer_idx)+2), 2**(int(layer_idx)+2),
               W.shape[0], W.shape[3], biases, gamma, beta, fs)
    layer_f.write(" ".join(lines.replace("'","").split()))

def make_gif(images, fname, duration=2, true_image=False):
  pass

def load_sample_inputs(config, nus_gan):
  path = os.path.join(config.test_dir, config.test_dataset)
  image = glob((os.path.join(path, '*.jpg')))
  image.sort()

  img = np.zeros((len(image), config.output_height, config.output_width, nus_gan.c_dim), dtype=np.float)
  for im in range(len(image)):
    tmp = np.resize(matimage.imread(image[im]), (config.output_height, config.output_width, nus_gan.c_dim))
    img[im] = np.array(tmp)/127.5 - 1.

  return img

def load_sample_y(config, nus_gan):
  sample_y = np.zeros((nus_gan.sample_num, nus_gan.y_dim), dtype=np.float)
  
  cnt = 0
  data_path = os.path.join(config.test_dir, config.test_dataset, "test_label.txt")
  for line in open(data_path, "r"):
      sample_y[cnt] = np.array([np.float(ch) for ch in line[0:-1]])
      cnt = cnt + 1

  return sample_y

def visualize(sess, nus_gan, config):

  sample_z = np.random.uniform(-1, 1, size=(nus_gan.sample_num, nus_gan.z_dim))
  #load inputs
  sample_inputs = load_sample_inputs(config, nus_gan)
  print(" [*] Success to load test inputs")
  #load sample_y
  sample_y = load_sample_y(config, nus_gan)
  print(" [*] Success to load test label")

  samples, d_loss, g_loss = sess.run(
          [nus_gan.sampler, nus_gan.d_loss, nus_gan.g_loss],
          feed_dict={
              nus_gan.inputs: sample_inputs,
              nus_gan.z: sample_z,
              nus_gan.y: sample_y
          }
  )
  
  out_path = os.path.join(config.test_dir, config.test_out, "test_sample.png")
  save_images(samples, sample_inputs, image_manifold_size(samples.shape[0]), out_path)
  print(" [*] Success to generate images using test inputs and label")
  print(" [*TEST] d_loss: %0.8f g_loss: %0.8f" % (d_loss, g_loss))


def image_manifold_size(num_images):
  manifold_h = int(np.floor(np.sqrt(num_images)))
  manifold_w = int(np.ceil(np.sqrt(num_images)))
  #assert manifold_h * manifold_w == num_images
  return manifold_h, manifold_w, 3

def gen_gender(len):
    label = np.zeros(len)
    hf = 60*7*10
    label[:hf] = 0
    label[hf:len] = 1
    return label
