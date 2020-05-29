import tensorflow as tf
from config import NUM_CLASS

def l1loss(x1, x2):
    return tf.reduce_mean(tf.abs(x1 - x2))

def adaptive_instance_norm(name, x, embedding):
    nums_out = x.shape[-1]
    beta = dense(name+"beta", embedding, nums_out, is_sn=True)
    gamma = dense(name + "gamma", embedding, nums_out, is_sn=True)
    beta = tf.reshape(beta, [-1, 1, 1, nums_out])
    gamma = tf.reshape(gamma, [-1, 1, 1, nums_out])
    mu, var = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
    x = (x - mu) * beta / tf.sqrt(var + 1e-10) + gamma
    return x

def instance_norm(name, x):
    nums_out = x.shape[-1]
    beta = tf.get_variable(name+"beta", [1, 1, 1, nums_out], initializer=tf.orthogonal_initializer(1.0))
    gamma = tf.get_variable(name+"gamma", [1, 1, 1, nums_out], initializer=tf.constant_initializer([0.0]))
    mu, var = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
    x = (x - mu) * beta / tf.sqrt(var + 1e-10) + gamma
    return x

def non_local(name, inputs, update_collection, is_sn):
    h, w, num_channels = inputs.shape[1], inputs.shape[2], inputs.shape[3]
    location_num = h * w
    downsampled_num = location_num // 4
    with tf.variable_scope(name):
        theta = conv("f", inputs, num_channels // 8, 1, 1, update_collection, is_sn)
        theta = tf.reshape(theta, [-1, location_num, num_channels // 8])
        phi = conv("h", inputs, num_channels // 8, 1, 1, update_collection, is_sn)
        phi = downsampling(phi)
        phi = tf.reshape(phi, [-1, downsampled_num, num_channels // 8])
        attn = tf.matmul(theta, phi, transpose_b=True)
        attn = tf.nn.softmax(attn)
        g = conv("g", inputs, num_channels // 2, 1, 1, update_collection, is_sn)
        g = downsampling(g)
        g = tf.reshape(g, [-1, downsampled_num, num_channels // 2])
        attn_g = tf.matmul(attn, g)
        attn_g = tf.reshape(attn_g, [-1, h, w, num_channels // 2])
        sigma = tf.get_variable("sigma_ratio", [], initializer=tf.constant_initializer(0.0))
        attn_g = conv("attn", attn_g, num_channels, 1, 1, update_collection, is_sn)
        return inputs + sigma * attn_g

def conv(name, inputs, nums_out, k_size, strides, update_collection=None, is_sn=False):
    nums_in = inputs.shape[-1]
    with tf.variable_scope(name):
        W = tf.get_variable("W", [k_size, k_size, nums_in, nums_out], initializer=tf.orthogonal_initializer())
        b = tf.get_variable("b", [nums_out], initializer=tf.constant_initializer([0.0]))
        if is_sn:
            W = spectral_normalization("sn", W, update_collection=update_collection)
        con = tf.nn.conv2d(inputs, W, [1, strides, strides, 1], "SAME")
    return tf.nn.bias_add(con, b)

def upsampling(inputs):
    H = inputs.shape[1]
    W = inputs.shape[2]
    return tf.image.resize_nearest_neighbor(inputs, [H * 2, W * 2])

def downsampling(inputs):
    return tf.nn.avg_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

def relu(inputs):
    return tf.nn.relu(inputs)

def global_sum_pooling(inputs):
    inputs = tf.reduce_sum(inputs, [1, 2], keep_dims=False)
    return inputs

def Hinge_loss(real_logits, fake_logits):
    D_loss = -tf.reduce_mean(tf.minimum(0., -1.0 + real_logits)) - tf.reduce_mean(tf.minimum(0., -1.0 - fake_logits))
    G_loss = -tf.reduce_mean(fake_logits)
    return D_loss, G_loss

def ortho_reg(vars_list):
    s = 0
    for var in vars_list:
        if "W" in var.name:
            if var.shape.__len__() == 4:
                nums_kernel = int(var.shape[-1])
                W = tf.transpose(var, perm=[3, 0, 1, 2])
                W = tf.reshape(W, [nums_kernel, -1])
                ones = tf.ones([nums_kernel, nums_kernel])
                eyes = tf.eye(nums_kernel, nums_kernel)
                y = tf.matmul(W, W, transpose_b=True) * (ones - eyes)
                s += tf.nn.l2_loss(y)
    return s

def dense(name, inputs, nums_out, update_collection=None, is_sn=False):
    nums_in = inputs.shape[-1]
    with tf.variable_scope(name):
        W = tf.get_variable("W", [nums_in, nums_out], initializer=tf.orthogonal_initializer())
        b = tf.get_variable("b", [nums_out], initializer=tf.constant_initializer([0.0]))
        if is_sn:
            W = spectral_normalization("sn", W, update_collection=update_collection)
    return tf.nn.bias_add(tf.matmul(inputs, W), b)

def Inner_product(global_pooled, y, update_collection=None):
    W = global_pooled.shape[-1]
    V = tf.get_variable("V", [NUM_CLASS, W], initializer=tf.orthogonal_initializer())
    V = tf.transpose(V)
    V = spectral_normalization("embed", V, update_collection=update_collection)
    V = tf.transpose(V)
    temp = tf.nn.embedding_lookup(V, y)
    # w0 = tf.get_variable("w0", [1, W], initializer=tf.orthogonal_initializer())
    # b0 = tf.get_variable("b0", [1])
    logit = tf.reduce_sum(temp * global_pooled, axis=1, keep_dims=True)
    return logit, temp

def G_DownResblock(name, inputs, nums_out, update_collection):
    with tf.variable_scope(name):
        temp = tf.identity(inputs)
        inputs = conv("conv1", inputs, nums_out, 3, 1, update_collection=update_collection, is_sn=True)
        inputs = instance_norm("in1", inputs)
        inputs = relu(inputs)
        inputs = conv("conv2", inputs, nums_out, 3, 1, update_collection=update_collection, is_sn=True)
        inputs = instance_norm("in2", inputs)
        inputs = downsampling(inputs)
        temp = downsampling(temp)
        temp = conv("identity", temp, nums_out, 1, 1, update_collection=update_collection, is_sn=True)
    return inputs + temp

def G_Resblock(name, inputs, nums_out, embedding):
    with tf.variable_scope(name):
        temp = tf.identity(inputs)
        inputs = adaptive_instance_norm("in1", inputs, embedding)
        inputs = relu(inputs)
        inputs = conv("conv1", inputs, nums_out, 3, 1, is_sn=True)
        inputs = adaptive_instance_norm("in2", inputs, embedding)
        inputs = relu(inputs)
        inputs = conv("conv2", inputs, nums_out, 3, 1, is_sn=True)
        #Identity mapping
        temp = conv("identity", temp, nums_out, 1, 1, is_sn=True)
    return inputs + temp

def G_UPResblock(name, inputs, nums_out, embedding):
    with tf.variable_scope(name):
        temp = tf.identity(inputs)
        inputs = adaptive_instance_norm("in1", inputs, embedding)
        inputs = relu(inputs)
        inputs = upsampling(inputs)
        inputs = conv("conv1", inputs, nums_out, 3, 1, is_sn=True)
        inputs = adaptive_instance_norm("in2", inputs, embedding)
        inputs = relu(inputs)
        inputs = conv("conv2", inputs, nums_out, 3, 1, is_sn=True)
        #Identity mapping
        temp = upsampling(temp)
        temp = conv("identity", temp, nums_out, 1, 1, is_sn=True)
    return inputs + temp

def D_Resblock(name, inputs, nums_out, update_collection=None, is_down=True):
    with tf.variable_scope(name):
        temp = tf.identity(inputs)
        inputs = relu(inputs)
        inputs = conv("conv1", inputs, nums_out, 3, 1, update_collection, is_sn=True)
        inputs = relu(inputs)
        inputs = conv("conv2", inputs, nums_out, 3, 1, update_collection, is_sn=True)
        if is_down:
            inputs = downsampling(inputs)
            #Identity mapping
            temp = conv("identity", temp, nums_out, 1, 1, update_collection, is_sn=True)
            temp = downsampling(temp)
        else:
            temp = conv("identity", temp, nums_out, 1, 1, update_collection, is_sn=True)

    return inputs + temp

def D_FirstResblock(name, inputs, nums_out, update_collection, is_down=True):
    with tf.variable_scope(name):
        temp = tf.identity(inputs)
        inputs = conv("conv1", inputs, nums_out, 3, 1, update_collection=update_collection, is_sn=True)
        inputs = relu(inputs)
        inputs = conv("conv2", inputs, nums_out, 3, 1, update_collection=update_collection, is_sn=True)
        if is_down:
            inputs = downsampling(inputs)
            #Identity mapping
            temp = downsampling(temp)
            temp = conv("identity", temp, nums_out, 1, 1, update_collection=update_collection, is_sn=True)
    return inputs + temp



def _l2normalize(v, eps=1e-12):
  """l2 normize the input vector."""
  return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

def spectral_normalization(name, weights, num_iters=1, update_collection=None,
                           with_sigma=False):
  w_shape = weights.shape.as_list()
  w_mat = tf.reshape(weights, [-1, w_shape[-1]])  # [-1, output_channel]
  u = tf.get_variable(name + 'u', [1, w_shape[-1]],
                      initializer=tf.truncated_normal_initializer(),
                      trainable=False)
  u_ = u
  for _ in range(num_iters):
    v_ = _l2normalize(tf.matmul(u_, w_mat, transpose_b=True))
    u_ = _l2normalize(tf.matmul(v_, w_mat))

  sigma = tf.squeeze(tf.matmul(tf.matmul(v_, w_mat), u_, transpose_b=True))
  w_mat /= sigma
  if update_collection is None:
    with tf.control_dependencies([u.assign(u_)]):
      w_bar = tf.reshape(w_mat, w_shape)
  else:
    w_bar = tf.reshape(w_mat, w_shape)
    if update_collection != 'NO_OPS':
      tf.add_to_collection(update_collection, u.assign(u_))
  if with_sigma:
    return w_bar, sigma
  else:
    return w_bar

