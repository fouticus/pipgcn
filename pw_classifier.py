import os
import copy

import numpy as np
import tensorflow as tf

import nn_components

__all__ = [
    "PWClassifier",
]


class PWClassifier(object):
    def __init__(self, layer_specs, layer_args, train_data, learning_rate, pn_ratio, outdir):
        """ Assumes same dims and nhoods for l_ and r_ """
        self.layer_args = layer_args
        self.params = {}
        # tf stuff:
        self.graph = tf.Graph()
        self.sess = None
        self.preds = None
        self.labels = None
        #################################################################
        # get details of data
        self.diffusion = ('l_edge' not in train_data[0])  # which type of convolution is being performed?
        self.in_nv_dims = train_data[0]["l_vertex"].shape[-1]
        if self.diffusion:
            self.maxpower= train_data[0]["l_power_series"].shape[1]
        else:
            self.in_ne_dims = train_data[0]["l_edge"].shape[-1]
            self.in_nhood_size = train_data[0]["l_hood_indices"].shape[1]
        with self.graph.as_default():
            # shapes and tf variables
            self.in_vertex1 = tf.placeholder(tf.float32, [None, self.in_nv_dims], "vertex1")
            self.in_vertex2 = tf.placeholder(tf.float32, [None, self.in_nv_dims], "vertex2")
            if self.diffusion:
                self.power_transition1 = tf.placeholder(tf.float32, [None, self.maxpower, None], name="power_transition_matrices")
                self.power_transition2 = tf.placeholder(tf.float32, [None, self.maxpower, None], name="power_transition_matrices")
                input1 = self.in_vertex1, self.power_transition1
                input2 = self.in_vertex2, self.power_transition2
            else:
                self.in_edge1 = tf.placeholder(tf.float32, [None, self.in_nhood_size, self.in_ne_dims], "edge1")
                self.in_edge2 = tf.placeholder(tf.float32, [None, self.in_nhood_size, self.in_ne_dims], "edge2")
                self.in_hood_indices1 = tf.placeholder(tf.int32, [None, self.in_nhood_size, 1], "hood_indices1")
                self.in_hood_indices2 = tf.placeholder(tf.int32, [None, self.in_nhood_size, 1], "hood_indices2")
                input1 = self.in_vertex1, self.in_edge1, self.in_hood_indices1
                input2 = self.in_vertex2, self.in_edge2, self.in_hood_indices2
            self.examples = tf.placeholder(tf.int32, [None, 2], "examples")
            self.labels = tf.placeholder(tf.float32, [None], "labels")
            self.dropout_keep_prob = tf.placeholder(tf.float32, shape=[], name="dropout_keep_prob")
            #### make layers
            legs = True
            i = 0
            while i < len(layer_specs):
                layer = layer_specs[i]
                args = copy.deepcopy(layer_args)
                args["dropout_keep_prob"] = self.dropout_keep_prob
                type = layer[0]
                args2 = layer[1] if len(layer) > 1 else {}
                flags = layer[2] if len(layer) > 2 else None
                args.update(args2)  # local layer args override global layer args
                layer_fn = getattr(nn_components, type)
                # if "merge" flag is in this layer, then this is a merge layer and every subsequent layer is a merged layer
                if flags is not None and "merge" in flags:
                    legs = False
                    input = input1[0], input2[0], self.examples  # take vertex features only
                if legs:
                    # make leg layers (everything up to the merge layer)
                    name = "leg1_{}_{}".format(type, i)
                    with tf.name_scope(name):
                        output, params = layer_fn(input1, None, **args)
                        if params is not None:
                            self.params.update({"{}_{}".format(name, k): v for k, v in params.items()})
                        if self.diffusion:
                            input1 = output, self.power_transition1
                        else:
                            input1 = output, self.in_edge1, self.in_hood_indices1
                    name = "leg2_{}_{}".format(type, i)
                    with tf.name_scope(name):
                        output, _ = layer_fn(input2, params, **args)
                        if self.diffusion:
                            input2 = output, self.power_transition2
                        else:
                            input2 = output, self.in_edge2, self.in_hood_indices2
                else:
                    # merged layers
                    name = "{}_{}".format(type, i)
                    with tf.name_scope(name):
                        input, params = layer_fn(input, None, **args)
                        if params is not None and len(params.items()) > 0:
                            self.params.update({"{}_{}".format(name, k): v for k, v in params.items()})
                i += 1
            self.preds = input

            # Loss and optimizer
            with tf.name_scope("loss"):
                scale_vector = (pn_ratio * (self.labels - 1) / -2) + ((self.labels + 1) / 2)
                logits = tf.concat([-self.preds, self.preds], axis=1)
                labels = tf.stack([(self.labels - 1) / -2, (self.labels + 1) / 2], axis=1)
                self.loss = tf.losses.softmax_cross_entropy(labels, logits, weights=scale_vector)
            with tf.name_scope("optimizer"):
                # generate an op which trains the model
                self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)

            # set up tensorflow session
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            # Uncomment these to record compute graph:
            #self.summaries = tf.summary.merge_all()
            #self.summary_writer = tf.summary.FileWriter(outdir, self.graph)

    def run_graph(self, outputs, data, tt, options=None, run_metadata=None):
        with self.graph.as_default():
            dropout_keep = 1.0
            if tt == "train" and "dropout_keep_prob" in self.layer_args:
                dropout_keep = self.layer_args["dropout_keep_prob"]
            if self.diffusion:
                feed_dict = {
                    self.in_vertex1: data["l_vertex"],
                    self.in_vertex2: data["r_vertex"],
                    self.power_transition1: data["l_power_series"],
                    self.power_transition2: data["r_power_series"],
                    self.examples: data["label"][:, :2],
                    self.labels: data["label"][:, 2],
                    self.dropout_keep_prob: dropout_keep}
            else:
                feed_dict = {
                    self.in_vertex1: data["l_vertex"], self.in_edge1: data["l_edge"],
                    self.in_vertex2: data["r_vertex"], self.in_edge2: data["r_edge"],
                    self.in_hood_indices1: data["l_hood_indices"],
                    self.in_hood_indices2: data["r_hood_indices"],
                    self.examples: data["label"][:, :2],
                    self.labels: data["label"][:, 2],
                    self.dropout_keep_prob: dropout_keep}
            return self.sess.run(outputs, feed_dict=feed_dict, options=options, run_metadata=run_metadata)

    def get_labels(self, data):
        return {"label": data["label"][:, 2, np.newaxis]}

    def predict(self, data):
        results = self.run_graph([self.loss, self.preds], data, "test")
        results = {"label": results[1], "loss": results[0]}
        return results

    def loss(self, data):
        return self.run_graph(self.loss, data, "test")

    def train(self, data):
        return self.run_graph([self.train_op, self.loss], data, "train")

    def get_nodes(self):
        return [n for n in self.graph.as_graph_def().node]

    def close(self):
        with self.graph.as_default():
            self.sess.close()
