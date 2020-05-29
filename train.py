import tensorflow as tf
from utils import read_video_landmark
from networks import Generator, Discriminator, Embedder, vgg19
from ops import l1loss
from config import NUM_VIDEOS, WEIGHT_VGG, WEIGHT_MCH, WEIGHT_FM, IMG_HEIGHT, IMG_WIDTH, K_FRAMES
from PIL import Image
import numpy as np


def train():
    landmarks = tf.placeholder(tf.float32, [NUM_VIDEOS, K_FRAMES + 1, IMG_HEIGHT, IMG_WIDTH, 3])
    frames = tf.placeholder(tf.float32, [NUM_VIDEOS, K_FRAMES + 1, IMG_HEIGHT, IMG_WIDTH, 3])
    y = tf.placeholder(tf.int32, [NUM_VIDEOS])
    x_i = frames[:, :K_FRAMES]
    x_t = frames[:, K_FRAMES]
    y_i = landmarks[:, :K_FRAMES]
    y_t = landmarks[:, K_FRAMES]

    embedder = Embedder("embedder")
    G = Generator("generator")
    D = Discriminator("discriminator")
    y_i_splits = tf.split(y_i, NUM_VIDEOS)
    x_i_splits = tf.split(x_i, NUM_VIDEOS)
    embeddings = []
    for y_i_, x_i_ in zip(y_i_splits, x_i_splits):
        y_i_ = tf.squeeze(y_i_, axis=0)
        x_i_ = tf.squeeze(x_i_, axis=0)
        embedding = embedder(tf.concat([x_i_, y_i_], axis=-1))
        embedding = tf.reduce_mean(embedding, axis=0, keep_dims=True)
        embeddings.append(embedding)
    embeddings = tf.concat(embeddings, axis=0)
    fake = G(y_t, embeddings)
    fake_cat = tf.concat([fake, y_t], axis=-1)
    real_cat = tf.concat([x_t, y_t], axis=-1)
    fake_logits, w_i, fake_feats = D(fake_cat, y)
    real_logits, _, real_feats = D(real_cat, y, "NO_OPS")
    L_DSC = tf.reduce_mean(tf.nn.relu(1 + fake_logits) + tf.nn.relu(1 - real_logits))
    feats_vgg = vgg19(tf.concat([fake, x_t], axis=0))
    L_CNT = []
    for feat in feats_vgg:
        L_CNT.append(l1loss(feat[:NUM_VIDEOS], feat[NUM_VIDEOS:]))
    L_CNT = tf.reduce_sum(L_CNT)
    L_FM = []
    for f1, f2 in zip(fake_feats, real_feats):
        L_FM.append(l1loss(f1, f2))
    L_FM = tf.reduce_sum(L_FM)
    L_ADV = -tf.reduce_mean(fake_logits) + WEIGHT_FM * L_FM
    L_MCH = l1loss(embeddings, w_i)
    L_G = L_CNT * WEIGHT_VGG + L_ADV + L_MCH * WEIGHT_MCH

    Opt_G = tf.train.AdamOptimizer(5e-5, beta1=0, beta2=0.9).minimize(L_G, var_list=G.var_list() + embedder.var_list())
    Opt_D = tf.train.AdamOptimizer(2e-4, beta1=0, beta2=0.9).minimize(L_DSC, var_list=D.var_list())

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="vgg_19"))
    saver.restore(sess, "./pretrained/vgg.ckpt")
    saver = tf.train.Saver()
    saver.restore(sess, "./models/model.ckpt")
    for i in range(200000):
        FRAMES, LANDMARKS, LABELS = read_video_landmark()
        for j in range(2):
            sess.run(Opt_D, feed_dict={frames: FRAMES, landmarks: LANDMARKS, y: LABELS})
        [FAKE, D_LOSS, G_LOSS, _] = sess.run([fake, L_DSC, L_G, Opt_G], feed_dict={frames: FRAMES, landmarks: LANDMARKS, y: LABELS})
        if i % 100 == 0:
            print("Iteration: %d, D_loss: %f, G_loss: %f"%(i, D_LOSS, G_LOSS))
            out = np.concatenate((FRAMES[0, -1], LANDMARKS[0, -1], FAKE[0]), axis=1)
            Image.fromarray(np.uint8((out + 1) * 127.5)).save("./results/" + str(i) + ".jpg")
        if i % 1000 == 0:
            saver.save(sess, "./models/model.ckpt")
            pass

if __name__ == "__main__":
    train()
