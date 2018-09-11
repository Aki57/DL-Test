"""
cd /home/ubuntu/
python
"""
from generate_face import *
import tensorflow as tf

hparams = tf.contrib.training.HParams(
    data_root = './img_align_celeba',
    crop_h = 108,
    crop_w = 108,
    resize_h = 64,
    resize_w = 64,
    is_crop = True,
    z_dim = 100,
    batch_size = 64,
    sample_size = 64,
    output_h = 64,
    output_w = 64,
    gf_dim = 64,
    df_dim = 64)
face = generateFace(hparams)

img,z = face.next_batch(1)
z
save_images(img,(1,1),"test.jpg")
