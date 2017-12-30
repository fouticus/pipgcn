import numpy as np
import tensorflow as tf

__all__ = [
    "no_conv",
    "diffusion_convolution",
    "single_weight_matrix",
    "node_average",
    "node_edge_average",
    "order_dependent",
    "deep_tensor_conv",
    "dense", 
    "merge",
    "average_predictions",
    "initializer",
    "nonlinearity",
]

""" ====== Layers ====== """
""" All layers have as first two parameters:
        - input: input tensor or tuple of input tensors
        - params: dictionary of parameters, could be None
    and return tuple containing:
        - output: output tensor or tuple of output tensors
        - params: dictionary of parameters, could be None
"""


def no_conv(input, params, filters=None, dropout_keep_prob=1.0, trainable=True, **kwargs):
    vertices, _, _ = input
    vertices = tf.nn.dropout(vertices, dropout_keep_prob)
    v_shape = vertices.get_shape()
    if params is None:
        # create new weights
        Wvc = tf.Variable(initializer("he", (v_shape[1].value, filters)), name="Wvc", trainable=trainable)  # (v_dims, filters)
        bv = tf.Variable(initializer("zero", (filters,)), name="bv", trainable=trainable)
    else:
        # use shared weights
        Wvc = params["Wvc"]
        bv = params["bv"]
        filters = Wvc.get_shape()[-1].value
    params = {"Wvc": Wvc, "bv": bv}

    # generate vertex signals
    Zc = tf.matmul(vertices, Wvc, name="Zc")  # (n_verts, filters)
    nonlin = nonlinearity("relu")
    sig = Zc + bv
    z = tf.reshape(nonlin(sig), tf.constant([-1, filters]))
    z = tf.nn.dropout(z, dropout_keep_prob)
    return z, params



def node_edge_average(input, params, filters=None, dropout_keep_prob=1.0, trainable=True, **kwargs):
    vertices, edges, nh_indices = input
    nh_indices = tf.squeeze(nh_indices, axis=2)
    v_shape = vertices.get_shape()
    e_shape = edges.get_shape()
    nh_sizes = tf.expand_dims(tf.count_nonzero(nh_indices + 1, axis=1, dtype=tf.float32), -1)  # for fixed number of neighbors, -1 is a pad value
    if params is None:
        # create new weights
        Wvc = tf.Variable(initializer("he", (v_shape[1].value, filters)), name="Wvc", trainable=trainable)  # (v_dims, filters)
        bv = tf.Variable(initializer("zero", (filters,)), name="bv", trainable=trainable)
        Wvn = tf.Variable(initializer("he", (v_shape[1].value, filters)), name="Wvn", trainable=trainable)  # (v_dims, filters)
        We = tf.Variable(initializer("he", (e_shape[2].value, filters)), name="We", trainable=trainable)  # (e_dims, filters)
    else:
        # use shared weights
        Wvn, We = params["Wvn"], params["We"]
        Wvc = params["Wvc"]
        bv = params["bv"]
        filters = Wvc.get_shape()[-1].value
    params = {"Wvn": Wvn, "We": We, "Wvc": Wvc, "bv": bv}

    # generate vertex signals
    Zc = tf.matmul(vertices, Wvc, name="Zc")  # (n_verts, filters)
    # create neighbor signals
    e_We = tf.tensordot(edges, We, axes=[[2], [0]], name="e_We")  # (n_verts, n_nbors, filters)
    v_Wvn = tf.matmul(vertices, Wvn, name="v_Wvn")  # (n_verts, filters)

    Zn = tf.divide(tf.reduce_sum(tf.gather(v_Wvn, nh_indices), 1) + tf.reduce_sum(e_We, 1),
                   tf.maximum(nh_sizes, tf.ones_like(nh_sizes)))  # (n_verts, v_filters)
    nonlin = nonlinearity("relu")
    sig = Zn + Zc + bv
    z = tf.reshape(nonlin(sig), tf.constant([-1, filters]))
    z = tf.nn.dropout(z, dropout_keep_prob)
    return z, params


def dense(input, params, out_dims=None, dropout_keep_prob=1.0, nonlin=True, trainable=True, **kwargs):
    input = tf.nn.dropout(input, dropout_keep_prob)
    in_dims = input.get_shape()[-1].value
    out_dims = in_dims if out_dims is None else out_dims
    if params is None:
        W = tf.Variable(initializer("he", [in_dims, out_dims]), name="w", trainable=trainable)
        b = tf.Variable(initializer("zero", [out_dims]), name="b", trainable=trainable)
        params = {"W": W, "b": b}
    else:
        W, b = params["W"], params["b"]
    Z = tf.matmul(input, W) + b
    if(nonlin):
        nonlin = nonlinearity("relu")
        Z = nonlin(Z)
    Z = tf.nn.dropout(Z, dropout_keep_prob)
    return Z, params


def merge(input, _, **kwargs):
    input1, input2, examples = input
    out1 = tf.gather(input1, examples[:, 0])
    out2 = tf.gather(input2, examples[:, 1])
    output1 = tf.concat([out1, out2], axis=0)
    output2 = tf.concat([out2, out1], axis=0)
    return tf.concat((output1, output2), axis=1), None


def diffusion_conv(input, params, dropout_keep_prob=1.0, trainable=True, **kwargs):
    vertices, power_trans = input
    in_dims = vertices.get_shape()[-1].value
    power_hops = power_trans.get_shape()[1].value
    if params is None:
        W = tf.Variable(initializer("he", (1, 1, power_hops, in_dims)), name="w", trainable=trainable)
    else:
        W = params["W"]
    params = {"W": W}
    PX = tf.expand_dims(tf.tensordot(power_trans, vertices, axes=[[-1], [0]]), axis=1)
    Z = W * PX
    Z = tf.reshape(Z, shape=[-1, in_dims * power_hops])
    nonlin = nonlinearity("relu")
    Z = tf.nn.dropout(Z, dropout_keep_prob)
    return nonlin(Z), params


def single_weight_matrix(input, params, filters=None, dropout_keep_prob=1.0, trainable=True, **kwargs):
    vertices, edges, nh_indices = input
    nh_indices = tf.squeeze(nh_indices, axis=2)
    v_shape = vertices.get_shape()
    nh_sizes = tf.expand_dims(tf.count_nonzero(nh_indices + 1, axis=1, dtype=tf.float32), -1)  # for fixed number of
    if params is None:
        # create new weights
        W = tf.Variable(initializer("he", (v_shape[1].value, filters)), name="W", trainable=trainable)  # (v_dims, filters)
        b = tf.Variable(initializer("zero", (filters,)), name="b", trainable=trainable)
    else:
        # use shared weights
        W = params["W"]
        b = params["b"]
        filters = W.get_shape()[-1].value
    params = {"W": W, "b": b}

    # generate vertex signals
    Zc = tf.matmul(vertices, W, name="Zc")  # (n_verts, filters)
    # create neighbor signals
    v_W = tf.matmul(vertices, W, name="v_W")  # (n_verts, filters)
    Zn = tf.divide(tf.reduce_sum(tf.gather(v_W, nh_indices), 1), tf.maximum(nh_sizes, tf.ones_like(nh_sizes)))  # (n_verts, v_filters)

    nonlin = nonlinearity("relu")

    sig = Zc + Zn + b
    h = tf.reshape(nonlin(sig), tf.constant([-1, filters]))
    h = tf.nn.dropout(h, dropout_keep_prob)
    return h, params


def node_average(input, params, filters=None, dropout_keep_prob=1.0, trainable=True, **kwargs):
    vertices, edges, nh_indices = input
    nh_indices = tf.squeeze(nh_indices, axis=2)
    v_shape = vertices.get_shape()
    nh_sizes = tf.expand_dims(tf.count_nonzero(nh_indices + 1, axis=1, dtype=tf.float32), -1)  # for fixed number of neighbors, -1 is a pad value

    if params is None:
        # create new weights
        Wc = tf.Variable(initializer("he", (v_shape[1].value, filters)), name="Wc", trainable=trainable)  # (v_dims, filters)
        Wn = tf.Variable(initializer("he", (v_shape[1].value, filters)), name="Wn", trainable=trainable)  # (v_dims, filters)
        b = tf.Variable(initializer("zero", (filters,)), name="b", trainable=trainable)
    else:
        Wn, Wc = params["Wn"], params["Wc"]
        filters = Wc.get_shape()[-1].value
        b = params["b"]
    params = {"Wn": Wn, "Wc": Wc, "b": b}

    # generate vertex signals
    Zc = tf.matmul(vertices, Wc, name="Zc")  # (n_verts, filters)
    # create neighbor signals
    v_Wn = tf.matmul(vertices, Wn, name="v_Wn")  # (n_verts, filters)
    Zn = tf.divide(tf.reduce_sum(tf.gather(v_Wn, nh_indices), 1),
                   tf.maximum(nh_sizes, tf.ones_like(nh_sizes)))  # (n_verts, v_filters)

    nonlin = nonlinearity("relu")
    sig = Zn + Zc + b
    h = tf.reshape(nonlin(sig), tf.constant([-1, filters]))
    h = tf.nn.dropout(h, dropout_keep_prob)
    return h, params


def order_dependent(input, params, filters=None, dropout_keep_prob=1.0, trainable=True, **kwargs):
    vertices, edges, nh_indices = input
    v_shape = vertices.get_shape()
    e_shape = edges.get_shape()
    nh_indices = tf.squeeze(nh_indices, axis=2)
    nh_size = nh_indices.get_shape()[1].value

    if params is None:
        # create new weights
        Wc = tf.Variable(initializer("he", (v_shape[1].value, filters)), name="Wn", trainable=trainable)  # (v_dims, filters)
        Wn = [tf.Variable(initializer("he", (v_shape[1].value, filters)), name="Wc{}".format(i), trainable=trainable) for i in range(nh_size)]  # (v_dims, filters)
        We = tf.Variable(initializer("he", (nh_size, e_shape[2].value, filters)), name="We", trainable=trainable)  # (n_nbors, e_dims, filters)
        b = tf.Variable(initializer("zero", (filters,)), name="b", trainable=trainable)
    else:
        # use shared weights
        Wc = params["Wc"]
        Wn = [params["Wn{}".format(i)] for i in range(nh_size)]
        We = params["We"]
        b = params["b"]
        filters = Wc.get_shape()[-1].value
    params = {"Wc": Wc, "We": We, "b":b}
    params.update({"Wn{}".format(i): Wn[i] for i in range(nh_size)})

    # generate vertex signals
    Zc = tf.matmul(vertices, Wc, name="Zc")  # (n_verts, filters)
    # create neighbor signals
    # for each neighbor, calculate signals:
    Zn = tf.zeros_like(Zc)
    for i in range(nh_size):
        Zn += tf.matmul(tf.gather(vertices, nh_indices[:, i]), Wn[i])
    Ze = tf.tensordot(edges, We, axes=[[1, 2], [0, 1]])  # (n_verts, filters)
    Zn = tf.divide(Zn, tf.constant(nh_size, dtype=tf.float32))
    Ze = tf.divide(Ze, tf.constant(nh_size, dtype=tf.float32))

    nonlin = nonlinearity("relu")
    sig = Zn + Ze + Zc + b
    h = tf.reshape(nonlin(sig), tf.constant([-1, filters]))
    h = tf.nn.dropout(h, dropout_keep_prob)
    return h, params


def deep_tensor_conv(input, params, factors=None, dropout_keep_prob=1.0, trainable=True, **kwargs):
    vertices, edges, nh_indices = input
    nh_indices = tf.squeeze(nh_indices, axis=2)
    v_shape = vertices.get_shape()
    e_shape = edges.get_shape()
    nh_size = nh_indices.get_shape()[1].value
    if params is None:
        # create new weights
        Wdf = tf.Variable(initializer("he", (e_shape[2].value, factors)), name="Wdf", trainable=trainable)  # (e_dims, factors)
        Wcf = tf.Variable(initializer("he", (v_shape[1].value, factors)), name="Wcf", trainable=trainable)  # (v_dims, factors)
        Wfc = tf.Variable(initializer("he", (factors, v_shape[1].value)), name="Wfc", trainable=trainable)  # (factors, v_dims)
        bf1 = tf.Variable(initializer("zero", (factors,)), name="be", trainable=trainable)
        bf2 = tf.Variable(initializer("zero", (factors,)), name="bv", trainable=trainable)
        params = {name: thing for name, thing in [("Wdf", Wdf), ("Wcf", Wcf), ("Wfc", Wfc), ("bf1", bf1), ("bf2", bf2)]}
    else:
        Wdf, Wcf, Wfc = params["Wdf"], params["Wcf"], params["Wfc"]
        bf1, bf2 = params["bf1"], params["bf2"]

    #nonlinearity
    nonlin = nonlinearity("tanh")

    # create neighbor signals
    v_Wcf = tf.matmul(vertices, Wcf, name="v_Wcf")  # (n_verts, factors)
    e_Wdf = tf.tensordot(edges, Wdf, axes=[[2], [0]], name="e_Wdf")  # (n_verts, n_nbors, factors)
    v_Wcf += bf1
    e_Wdf += bf2
    Zn = tf.gather(v_Wcf, nh_indices) * e_Wdf  # (n_verts, n_nbors, factors)
    V_ij = nonlin(tf.tensordot(Zn, Wfc, axes=[[2], [0]], name="Vij"))  # (n_verts, n_nbors, v_dims)

    sig = vertices + tf.divide(tf.reduce_sum(V_ij, axis=1), tf.constant(nh_size, dtype=tf.float32))  # (n_verts, v_dims)
    z = tf.reshape(sig, tf.constant([-1, v_shape[1].value]))
    z = tf.nn.dropout(z, dropout_keep_prob)
    return z, params


def average_predictions(input, _, **kwargs):
    combined = tf.reduce_mean(tf.stack(tf.split(input, 2)), 0)
    return combined, None



""" ======== Non Layers ========= """


def initializer(init, shape):
    if init == "zero":
        return tf.zeros(shape)
    elif init == "he":
        fan_in = np.prod(shape[0:-1])
        std = 1/np.sqrt(fan_in)
        return tf.random_uniform(shape, minval=-std, maxval=std)


def nonlinearity(nl):
    if nl == "relu":
        return tf.nn.relu
    elif nl == "tanh":
        return tf.nn.tanh
    elif nl == "linear" or nl == "none":
        return lambda x: x
