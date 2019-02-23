import pickle
import os
from time import time
import pickle

from dynamicgem.embedding.ae_static    import AE
from dynamicgem.embedding.dynAERNN     import DynAERNN
from dynamicgem.embedding.dynRNN       import DynRNN

def load_obj(path='unnamed_file.pkl'):
    with open(path, 'rb') as f:
        any_object = pickle.load(f)
    return any_object

#C:\Users\hasee\PycharmProjects\DynamicGEM\data

def static_AE(file_location, dim_emb = 128):
    networks = load_obj(file_location)
    for i in range(len(networks)):
        print(len(networks[i]))

    embedding = AE(d=dim_emb,
                   beta=5,
                   nu1=1e-6,
                   nu2=1e-6,
                   K=3,
                   n_units=[500, 300, ],
                   n_iter=200,
                   xeta=1e-4,
                   n_batch=100,
                   modelfile=['./intermediate/enc_modelsbm.json',
                              './intermediate/dec_modelsbm.json'],
                   weightfile=['./intermediate/enc_weightssbm.hdf5',
                               './intermediate/dec_weightssbm.hdf5'])
    embs = []
    t1 = time()
    all_emb_dicts = []
    # ae static
    for temp_var in range(len(networks)):
        emb, _ = embedding.learn_embeddings(networks[temp_var])
        emb_dict = prep_embeddings(networks[temp_var], emb)
        all_emb_dicts.append(emb_dict)
        embs.append(emb)#it is numpy

    save_embedding_result(embs)

    print(embedding._method_name + ':\n\tTraining time: %f' % (time() - t1))


def save_embedding_result(obj, path='static_AE_embedding.data'):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def prep_embeddings(single_network, embeddings):
    nodes = list(single_network.nodes())
    embed_dict = {}
    for i in len(nodes):
        embed_dict[nodes[i]] = embeddings[i]
    return embed_dict

if __name__ == '__main__':
    static_AE("C:/Users/hasee/PycharmProjects/DynamicGEM/data/as733_dyn_graphs.pkl")