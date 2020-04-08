import sys
import os
import utils
import model
import numpy as np
import tensorflow as tf
import scipy.sparse as sp

"""
Inputs to the model:
X: a Nxf matrix, with N nodes (in this case genes) and f features (in this case patients)
Y: a N-element vector of labels, only a small portion of the nodes are explicitly labeled as 0s or 1s,
   the rest of them are coded as 999. 
A: A NxN symmetrical adjacent matrix without self-connection. Aij=1 indicates there is connection between the ith 
   and the jth node (gene).
   The data is a three-column array, each row is an edge: 1, i, j.  
"""

LOG_DIR = './log'
MODEL_DIR = './model'
X_DIR = '/media/lwk/data/pancancer/data/brca_retro_pr_data_gcn.txt'
Y_DIR = '/media/lwk/data/pancancer/data/brca_retro_pr_y_gcn.txt'
A_DIR = '/media/lwk/data/pancancer/data/brca_retro_pr_adj_gcn.txt'


def main(to_load=None):

    X = utils.iter_loadtxt(filename=X_DIR, delimiter='\t', skiprows=1, dtype=float)
    Y = utils.iter_loadtxt(filename=Y_DIR, delimiter='\t', skiprows=0, dtype=int)
    A = utils.iter_loadtxt(filename=A_DIR, delimiter='\t', skiprows=1, dtype=int)
    adj_raw = sp.coo_matrix((A[:, 0], (A[:, 1], A[:, 2])), shape=(X.shape[0], X.shape[0]))
    adj_train = utils.preprocess_adj(adj_raw)
    print(X.shape)

    if to_load:    # restore model, get node classification predictions
        m = model.GCN(meta_graph=to_load)
        print("Loaded!", flush=True)
        res = m.inference(X, adj_train)
        np.savetxt(fname='test_res.txt', X=res, delimiter='\t', comments='')

    else:
        m = model.GCN(log_dir=LOG_DIR, nodes=X.shape[0], features=X.shape[1],
                      hidden_sizes=(64, 5), save_graph_def=False)

        m.train(x=X, y=Y, adj_pars=adj_train, model_dir=MODEL_DIR, save=False, n_epoch=100)
        print("Trained!", flush=True)
        res = m.inference(X, adj_train)
        np.savetxt(fname='test_res.txt', X=res, delimiter='\t', comments='')


if __name__ == "__main__":

    tf.reset_default_graph()

    for DIR in (LOG_DIR, MODEL_DIR):
        try:
            os.mkdir(DIR)
        except FileExistsError:
            pass

    try:
        to_reload = sys.argv[1]
        main(to_load=to_reload)
    except IndexError:
        main()
