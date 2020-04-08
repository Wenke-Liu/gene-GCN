import os
import re
import tensorflow as tf
import layers
from datetime import datetime


class GCN:
    """
    Graph Convolution model for semi-supervised learning of node classification
    Based on Kipf & Welling (2016) https://arxiv.org/pdf/1609.02907.pdf
    """
    RESTORE_KEY = "to_restore"

    def __init__(self,
                 nodes=None,
                 features=None,
                 hidden_sizes=(64, 2),
                 learning_rate=0.01,
                 lambda_l2_reg=0.01,
                 dropout=0.8,
                 n_class=2,
                 meta_graph=None,
                 save_graph_def=True,
                 log_dir='./log'):

        self.nodes = nodes
        self.features = features
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.lambda_l2_reg = lambda_l2_reg
        self.dropout = dropout
        self.n_class = n_class
        self.epoch_trained = 0

        self.sesh = tf.Session()

        if not meta_graph:  # new model
            self.datetime = datetime.now().strftime(r"%y%m%d_%H%M")

            # build graph
            handles = self._buildGraph()
            for handle in handles:
                tf.add_to_collection(GCN.RESTORE_KEY, handle)
            self.sesh.run(tf.global_variables_initializer())

        else:  # restore saved model
            model_datetime, model_name = os.path.basename(meta_graph).split("_GCN_")
            self.datetime = "{}_reloaded".format(model_datetime)
            arch_par, hyper_par = model_name.split("_lr_")
            *model_architecture, _ = re.split("_|-", arch_par)
            self.hidden_sizes = (int(n) for n in model_architecture)
            self.learning_rate, _, self.dropout, _, self.lambda_l2_reg = re.split('_', hyper_par)
            for par in [self.learning_rate, self.dropout, self.lambda_l2_reg]:
                par = float(par)

            # rebuild graph
            meta_graph = os.path.abspath(meta_graph)
            tf.train.import_meta_graph(meta_graph + ".meta").restore(
                self.sesh, meta_graph)
            handles = self.sesh.graph.get_collection(GCN.RESTORE_KEY)

            # unpack handles for tensor ops to feed or fetch
        (self.x_in, self.y_in, self.y_filtered,
         self.adj_coo, self.adj_val, self.adj_dim,
         self.training_status,
         self.logits, self.pred, self.cost,
         self.global_step, self.train_op, self.merged_summary) = handles

        if save_graph_def:  # tensorboard
            try:
                os.mkdir(log_dir + '/training')
                # os.mkdir(log_dir + '/validation')
            except FileExistsError:
                pass
            self.train_logger = tf.summary.FileWriter(log_dir + '/training', self.sesh.graph)
            # self.valid_logger = tf.summary.FileWriter(log_dir + '/validation', self.sesh.graph)

    def _buildGraph(self):
        x_in = tf.placeholder(dtype=tf.float32, name="x", shape=(self.nodes, self.features))
        print(tf.shape(x_in))
        y_in = tf.placeholder(dtype=tf.int16, name="y", shape=[self.nodes, None])
        adj_coo = tf.placeholder(dtype=tf.int64)
        adj_val = tf.placeholder(dtype=tf.float32)
        adj_dim = tf.placeholder(dtype=tf.int64)
        is_train = tf.placeholder_with_default(False, shape=[], name="is_train")

        adj = tf.SparseTensor(adj_coo, adj_val, adj_dim)
        thrld = tf.Variable(self.n_class, trainable=False)
        thrld = tf.cast(thrld, dtype=tf.int16)
        y_mask = tf.less(y_in, thrld)
        y_filtered = tf.boolean_mask(y_in, y_mask)
        onehot_labels = tf.one_hot(indices=tf.cast(y_filtered, tf.int32), depth=self.n_class)

        dropout = tf.placeholder_with_default(1., shape=[], name="dropout")

        h_encoded = x_in
        for idx, hidden_size in enumerate(self.hidden_sizes):
            h_encoded = layers.gcn_layer(x=h_encoded,
                                         adj=adj,
                                         size=int(hidden_size),
                                         scope="h" + str(idx) + "_GCN_" + str(hidden_size),
                                         dropout=dropout)

        logits = layers.fc_dropout(h_encoded, scope="classification",
                                   size=self.n_class,
                                   activation=tf.identity,
                                   dropout=dropout)
        logits = tf.reshape(logits, shape=[self.nodes, self.n_class])
        y_mask = tf.reshape(y_mask, shape=[self.nodes])
        pred = tf.nn.softmax(logits, name="prediction")

        # classification loss: cross-entropy with the gene knockdown labels
        pred_loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,
                                                    logits=tf.boolean_mask(logits, y_mask, axis=0))

        tf.summary.scalar("pred_cost", pred_loss)

        with tf.name_scope("l2_regularization"):
            regularizers = [tf.nn.l2_loss(var) for var in self.sesh.graph.get_collection(
                "trainable_variables") if "weights" in var.name]
            l2_reg = self.lambda_l2_reg * tf.add_n(regularizers)

        with tf.name_scope("cost"):

            cost = pred_loss
            cost += l2_reg

            tf.summary.scalar("cost", cost)

        # optimization
        global_step = tf.Variable(0, trainable=False)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            train_op = tf.contrib.layers.optimize_loss(
                loss=cost,
                learning_rate=self.learning_rate,
                global_step=global_step,
                optimizer="Adam")

        merged_summary = tf.summary.merge_all()

        return (x_in, y_in, y_filtered,
                adj_coo, adj_val, adj_dim,
                is_train,
                logits, pred, cost,
                global_step, train_op, merged_summary)

    def inference(self, x, adj_pars):
        feed_dict = {self.x_in: x,
                     self.adj_coo: adj_pars[0], self.adj_val: adj_pars[1], self.adj_dim: adj_pars[2],
                     self.training_status: False}
        return self.sesh.run(self.pred, feed_dict=feed_dict)

    def train(self, x, y, adj_pars, model_dir='./model', res_dir='.', n_epoch=10, save=True):

        assert (x.shape[0] == self.nodes and x.shape[1] == self.features), "check input dimensions!"

        if save:
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
            outfile = os.path.join(os.path.abspath(model_dir), "{}_GCN_{}_lr_{}_drop_{}_l2_{}".format(
                str(self.datetime), "_".join(map(str, self.hidden_sizes)),
                str(self.learning_rate), str(self.dropout), str(self.lambda_l2_reg)))

        now = datetime.now().isoformat()[11:]
        print("------- Training begin: {} -------\n".format(now))
        try:
            epochs = 0
            for epoch in range(n_epoch):
                feed = {self.x_in: x, self.y_in: y,
                        self.adj_coo: adj_pars[0],
                        self.adj_val: adj_pars[1],
                        self.adj_dim: adj_pars[2],
                        self.training_status: True}
                fetches = [self.merged_summary, self.logits, self.pred, self.y_filtered,
                            self.cost, self.global_step, self.train_op]
                summary, logits, pred, y_filtered, cost, i, _ = self.sesh.run(fetches, feed_dict=feed)
                self.epoch_trained += 1
                epochs += 1
                print(self.epoch_trained)
                print(y_filtered)
                if epochs > n_epoch:
                    break

            try:
                self.train_logger.flush()
                self.train_logger.close()
            except AttributeError:  # not logging
                print('Not logging')

        except KeyboardInterrupt:
            pass

        now = datetime.now().isoformat()[11:]
        print("------- Training end: {} -------\n".format(now), flush=True)
        print('Epochs trained: {}'.format(str(self.epoch_trained)))
        i = self.global_step.eval(session=self.sesh)
        print('Global steps: {}'.format(str(i)))

        if save:
            saver.save(self.sesh, outfile, global_step=None)
            print('Trained model saved to {}'.format(outfile))

