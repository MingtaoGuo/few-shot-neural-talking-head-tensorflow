from ops import *
from vgg import vgg_19

class Generator:
    def __init__(self, name):
        self.name = name

    def __call__(self, inputs, embedding):
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            inputs = G_DownResblock("ResBlock1", inputs, 32, None)
            inputs = G_DownResblock("ResBlock2", inputs, 64, None)
            inputs = G_DownResblock("ResBlock3", inputs, 128, None)
            inputs = non_local("Non-local1", inputs, None, True)
            inputs = G_DownResblock("ResBlock4", inputs, 256, None)

            inputs = G_Resblock("ResBlock5", inputs, 512, embedding)
            inputs = G_Resblock("ResBlock6", inputs, 512, embedding)
            inputs = G_Resblock("ResBlock7", inputs, 512, embedding)
            inputs = G_Resblock("ResBlock8", inputs, 512, embedding)

            inputs = G_UPResblock("UPResBlock1", inputs, 256, embedding)
            inputs = G_UPResblock("UPResBlock2", inputs, 128, embedding)
            inputs = non_local("Non-local2", inputs, None, True)
            inputs = G_UPResblock("UPResBlock3", inputs, 64, embedding)
            inputs = G_UPResblock("UPResBlock4", inputs, 32, embedding)
            inputs = relu(adaptive_instance_norm("in", inputs, embedding))
            inputs = conv("conv", inputs, k_size=3, nums_out=3, strides=1)
        return tf.nn.tanh(inputs)

    def var_list(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)

class Discriminator:
    def __init__(self, name):
        self.name = name

    def __call__(self, inputs, y, update_collection=None):
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            feats = []
            inputs = D_FirstResblock("ResBlock1", inputs, 64, update_collection, is_down=True)
            feats.append(inputs)
            inputs = D_Resblock("ResBlock2", inputs, 128, update_collection, is_down=True)
            feats.append(inputs)
            inputs = D_Resblock("ResBlock3", inputs, 256, update_collection, is_down=True)
            feats.append(inputs)
            inputs = non_local("Non-local", inputs, update_collection, True)
            inputs = D_Resblock("ResBlock4", inputs, 512, update_collection, is_down=True)
            feats.append(inputs)
            inputs = D_Resblock("ResBlock5", inputs, 512, update_collection, is_down=True)
            feats.append(inputs)
            inputs = D_Resblock("ResBlock6", inputs, 512, update_collection, is_down=True)
            feats.append(inputs)
            inputs = global_sum_pooling(inputs)
            inputs = relu(inputs)
            temp, w = Inner_product(inputs, y, update_collection)
            inputs = dense("dense", inputs, 1, update_collection, is_sn=True)
            inputs= temp + inputs
            return inputs, w, feats

    def var_list(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)

class Embedder:
    def __init__(self, name):
        self.name = name

    def __call__(self, inputs):
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            inputs = D_FirstResblock("ResBlock1", inputs, 64, None, is_down=True)
            inputs = D_Resblock("ResBlock2", inputs, 128, None, is_down=True)
            inputs = D_Resblock("ResBlock3", inputs, 256, None, is_down=True)
            inputs = non_local("Non-local", inputs, None, True)
            inputs = D_Resblock("ResBlock4", inputs, 512, None, is_down=True)
            inputs = D_Resblock("ResBlock5", inputs, 512, None, is_down=True)
            # inputs = D_Resblock("ResBlock6", inputs, 512, None, is_down=True)
            inputs = global_sum_pooling(inputs)
            inputs = relu(inputs)
            return inputs

    def var_list(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)

def vgg19(inputs):
    feats = vgg_19(inputs)
    return feats

def vggface(inputs):
    return inputs

if __name__ == "__main__":
    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    z = tf.placeholder(tf.float32, [None, 100])
    y = tf.placeholder(tf.float32, [None, 100])
    train_phase = tf.placeholder(tf.bool)
    G = Generator("generator")
    D = Discriminator("discriminator")
    fake_img = G(z, train_phase)
    fake_logit = D(fake_img)
    aaa = 0