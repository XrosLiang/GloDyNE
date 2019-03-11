'''
demo of evaluating node embedding in downsatream task(s)

python src/main.py --method DynRWSG --task save --graph data/cora/cora_dyn_graphs.pkl --label data/cora/cora_node_label_dict.pkl --emb-file output/cora_DynRWSG_128_embs.pkl --num-walks 20 --restart-prob 0.2 --update-threshold 0.1 --emb-dim 128 --workers 24
python src/eval.py --task all --graph data/cora/cora_dyn_graphs.pkl --label data/cora/cora_node_label_dict.pkl --emb-file output/cora_DynRWSG_128_embs.pkl

by Chengbin HOU <chengbin.hou10@foxmail.com>
'''

import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from libne.utils import load_any_obj_pkl
import os

def get_node_ID_across_graphs(nx_graphs):
    node_lists = []
    for i in range(len(nx_graphs)):
        node_lists.append(nx_graphs[i].nodes)
    node_union = set().union(*node_lists)
    return node_union

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    parser.add_argument('--graph', default='LFR_graphs.pkl',
                        help='graph/network file')
    parser.add_argument('--label', default='data/cora/cora_node_label_dict.pkl',
                        help='node label file')
    parser.add_argument('--emb-file', default='C:/Users/hasee/',
                        help='node embeddings file; suggest: data_method_dim_embs.pkl')
    parser.add_argument('--task', default='all', choices=['lp', 'nc', 'gr', 'all'],
                        help='choices of downstream tasks: lp, nc, gr, all')
    args = parser.parse_args()
    return args

def get_embedding_locations(path = "/usr/path/", startwith = "Zmatrix"):#C:\Users\hasee
    print("load embedding files from a dictionary")
    raw_embedding_file_locations = []
    for file in os.listdir(path):#the dictionary
        if file.startswith(startwith):
            raw_embedding_file_locations.append(path+file)
    if len(raw_embedding_file_locations) == 0:
        raise ValueError('No file found please check path and file names')
    else:
        embedding_file_locations = []
        for i in range(len(raw_embedding_file_locations)):
            embedding_file_locations.append(path+startwith+str(i))
    """
    for i in range(len(embedding_file_locations)):
        print(embedding_file_locations[i])
        read_single_embedding_file(embedding_file_locations[i])
    """
    #read_single_embedding_file(embedding_file_locations[0])
    return embedding_file_locations

def generate_embedding_dicts(graphs, embeeding_locations):
    node_union = list(get_node_ID_across_graphs(graphs))
    node_union.sort()
    dict_list = []
    for i in range(len(embeeding_locations)):
        single_dict = get_single_embedding_file(file = embeeding_locations[i], sorted_node_union= node_union, original_graph = graphs[i])
        dict_list.append(single_dict)
    print("finished construct embeddings")
    return dict_list

def get_single_embedding_file(file = '',sorted_node_union = '', original_graph = ''):
    f = open(file, "r")
    lines = f.readlines()
    dict = {}
    node_list = list(original_graph.nodes)
    for i in range(1,len(lines)):
        items = lines[i].split(':')
        item_info = items[0].split(',')
        node_ID = sorted_node_union[int(item_info[0])]
        #if node_ID in node_list:
        if True:
            embeddings = []
            for j in range(1,(len(items)-1)):
                #print(items[j])
                item_info = items[j].split(',')
                #print(item_info[1])
                embeddings.append(float(item_info[1]))
            final_dimension = items[len(items)-1].split(',')[1]
            embeddings.append(float(final_dimension[:-2]))
            #if len(embeddings) != 40:
            #    print(embeddings, "embeddings not 40 dimension")
            for k in range(len(embeddings) != 128):
                embeddings = [0] * 128
            dict[node_ID] = embeddings

            #print(node_ID)
            #print(embeddings)

    return dict






def main(args):
    print(f'Summary of all settings: {args}')

    # ---------------------------------------STEP1: prepare data-----------------------------------------------------
    print('\nSTEP1: start loading data......')
    t1 = time.time()
    G_dynamic = load_any_obj_pkl(args.graph)
    embeeding_locations = get_embedding_locations(args.emb_file)#this part needs update
    emb_dicts = generate_embedding_dicts(G_dynamic, embeeding_locations)
    t2 = time.time()
    print(f'STEP1: end loading data; time cost: {(t2-t1):.2f}s')

    # ---------------------------------------STEP2: downstream task(s)-----------------------------------------------
    print('\nSTEP2: start evaluating ......: ')
    t1 = time.time()   
    if args.task == 'lp' or args.task == 'all':
        from libne.downstream import lpClassifier, gen_test_edge_wrt_changes
        for t in range(len(G_dynamic)-1):
            print(f'Current time step @t: {t}')
            print(f'Link Prediction task by AUC score: use current emb @t to predict **future** changed links @t+1')
            pos_edges_with_label, neg_edges_with_label = gen_test_edge_wrt_changes(G_dynamic[t],G_dynamic[t+1])
            test_edges = [e[:2] for e in pos_edges_with_label] + [e[:2] for e in neg_edges_with_label]
            test_label = [e[2] for e in pos_edges_with_label] + [e[2] for e in neg_edges_with_label]
            ds_task = lpClassifier(emb_dict=emb_dicts[t])  # use current emb @t
            ds_task.evaluate_auc(test_edges, test_label)
    if args.task == 'nc' or args.task == 'all':
        from libne.downstream import ncClassifier
        from sklearn.linear_model import LogisticRegression  # to do... try SVM...
        try:
            label_dict = load_any_obj_pkl(args.label) # ground truth label .pkl
            for t in range(len(G_dynamic)):
                print(f'Current time step @t: {t}')
                print(f'Node Classification task by F1 score: use current emb @t to infer **current* corresponding label @t')
                X = []
                Y = []
                for node in G_dynamic[t].nodes(): # only select current available nodes for eval
                    X.append(node)
                    Y.append(str(label_dict[node])) # label as str, otherwise, sklearn error
                ds_task = ncClassifier(emb_dict=emb_dicts[t], clf=LogisticRegression())  # use current emb @t
                ds_task.split_train_evaluate(X, Y, train_precent=0.5)
        except:
            print(f'ground truth label file not exist; not support node classification task')
    if args.task == 'gr' or args.task == 'all':
        from libne.downstream import grClassifier
        for t in range(len(G_dynamic)):
            precision_at_k = 50
            print(f'Current time step @t: {t}')
            print(f'Graph Reconstruction by MAP @{precision_at_k} task: use current emb @t to reconstruct **current** graph @t')
            ds_task = grClassifier(emb_dict=emb_dicts[t], rc_graph=G_dynamic[t]) # use current emb @t
            # ds_task.evaluate_precision_k(top_k=precision_at_k)
            ds_task.evaluate_average_precision_k(top_k=precision_at_k)
    t2 = time.time()
    print(f'STEP2: end evaluating; time cost: {(t2-t1):.2f}s')


if __name__ == '__main__':
    """
    embedding_locations = get_embedding_locations(path = 'C:/Users/hasee/',startwith= 'Zmatrix')#for starwith Do NOT include number
    graphs = G_dynamic = load_any_obj_pkl("LFR_graphs.pkl")
    generate_embedding_dicts(graphs, embedding_locations)
    """
    print(f'------ START @ {time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())} ------')
    main(parse_args())
    print(f'------ END @ {time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())} ------')
