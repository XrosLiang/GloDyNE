import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
from utils import edge_s1_minus_s0, unique_nodes_from_edge_set
import random
#import gensim
#import pickle
import bisect



def add_new_nodes_up_right_corner(graph, m, n, add_node_number = 4, offset = 0):
    """
    :param graph:
    :param pos:
    :param m:
    :param n:
    :param add_node_number:
    :param offset:
    :return:
    """
    new_nodes = []
    for i in range(add_node_number):
        node_ID = (m,n-1-offset-i)
        new_nodes.append(node_ID)
        graph.add_node(node_ID)
    return graph

def remove_nodes_down_right_corner(graph, m, n, add_node_number = 4, offset = 0):
    """
    :param graph:
    :param pos:
    :param m:
    :param n:
    :param add_node_number:
    :param offset:
    :return:
    """
    for i in range(add_node_number):
        node_ID = (m-1,offset+i)
        graph.add_node(node_ID)
        graph.remove_node(node_ID)


    return graph

def add_new_edges_for_new_nodes(graph):
    nodes = list(graph.nodes)
    for i in range(len(nodes)):
        current_node = nodes[i]
        m = current_node[0]
        n = current_node[1]
        potential_neighbors = [(m-1,n),(m,n-1),(m+1,n),(m,n+1)]
        for j in range(len(potential_neighbors)):
            neighbor = potential_neighbors[j]
            if graph.has_node(neighbor):
                if graph.has_edge(current_node,neighbor) == False:
                    graph.add_edge(current_node,neighbor)
    return graph


def pos_calculation(graph, proportation = 0.1):
    nodes = list(graph.nodes)
    pos_dict = {}
    for i in range(len(nodes)):
        current_node = nodes[i]
        m = float(current_node[0])*proportation
        n = float(current_node[1])*proportation
        current_node_pos = np.array([m,n])
        pos_dict[current_node] = current_node_pos
    return pos_dict

def randomly_select(nodes, sample_number):
    nodes_number = len(nodes)
    print(sample_number,"sample number")
    print(nodes_number, "node number")
    if sample_number > nodes_number:
        sample_number = nodes_number
    random_numbers = random.sample(range(0, nodes_number), sample_number)
    selected_list = []
    for i in range(sample_number):
        selected_list.append(nodes[random_numbers[i]])
    return selected_list

def node_selecting_scheme(graph_t0, graph_t1, reservoir_dict={}, limit=0.1, local_global=0.5, scheme=1):
    ''' select nodes to be updated
         G0: previous graph @ t-1;
         G1: current graph  @ t;
         reservoir_dict: will be always maintained in ROM
         limit: fix the number of node --> the percentage of nodes of a network to be updated (exclude new nodes)
         local_global: # of nodes from recent changes v.s. from random nodes

         scheme 1: new nodes + most affected nodes
         scheme 2: new nodes + random nodes
         scheme 3: new nodes + most affected nodes + random nodes
    '''
    print("local global:",local_global)
    G0 = graph_t0.copy()
    G1 = graph_t1.copy()
    edge_add = edge_s1_minus_s0(s1=set(G1.edges()),
                                s0=set(G0.edges()))  # one may directly use streaming added edges if possible
    edge_del = edge_s1_minus_s0(s1=set(G0.edges()),
                                s0=set(G1.edges()))  # one may directly use streaming added edges if possible

    node_affected_by_edge_add = unique_nodes_from_edge_set(edge_add)  # unique
    node_affected_by_edge_del = unique_nodes_from_edge_set(edge_del)  # unique
    node_affected = list(set(node_affected_by_edge_add + node_affected_by_edge_del))  # unique
    node_add = [node for node in node_affected_by_edge_add if node not in G0.nodes()]
    node_del = [node for node in node_affected_by_edge_del if node not in G1.nodes()]
    if len(node_del) != 0:
        reservoir_key_list = list(reservoir_dict.keys())
        for node in node_del:
            if node in reservoir_key_list:
                del reservoir_dict[node]  # if node being deleted, also delete it from reservoir

    exist_node_affected = list(
        set(node_affected) - set(node_add) - set(node_del))  # affected nodes are in both G0 and G1

    t1 = time.time()
    # for fair comparsion, the number of nodes to be updated are the same for schemes 1, 2, 3
    num_limit = int(G1.number_of_nodes() * limit)
    local_limit = int(local_global * num_limit)
    global_limit = num_limit - local_limit

    node_update_list = []  # all the nodes to be updated
    if scheme == 1:
        print('scheme == 1')
        most_affected_nodes, reservoir_dict = select_most_affected_nodes(G0, G1, num_limit, reservoir_dict,
                                                                         exist_node_affected)

        if len(most_affected_nodes) != 0:
            if len(most_affected_nodes) < num_limit:  # for fairness, resample until meets num_limit
                temp_num = num_limit - len(most_affected_nodes)
                #temp_nodes = list(np.random.choice(most_affected_nodes + node_add, temp_num, replace=True))
                temp_nodes = randomly_select(list(most_affected_nodes + node_add), temp_num)
                most_affected_nodes.extend(temp_nodes)

            node_update_list = node_add + most_affected_nodes
        else:
            print('nothing changed... For fairness, randomly update some as scheme 2 does')
            all_nodes = [node for node in G1.nodes()]
            random_nodes = list(np.random.choice(all_nodes, num_limit, replace=False))
            node_update_list = node_add + random_nodes

    if scheme == 2:
        print('scheme == 2')
        all_nodes = [node for node in G1.nodes()]
        #random_nodes = list(np.random.choice(all_nodes, num_limit, replace=False))
        random_nodes = randomly_select(all_nodes, num_limit)
        #random_nodes = randomly_select(all_nodes, num_limit)
        node_update_list = node_add + random_nodes
        # node_update_list = ['1','1','1']

    if scheme == 3:
        print('scheme == 3')
        most_affected_nodes, reservoir_dict = select_most_affected_nodes(G0, G1, local_limit, reservoir_dict,
                                                                         exist_node_affected)

        """
        if len(most_affected_nodes)!= 0:
            if len(most_affected_nodes) < local_limit:  # resample until meets local_limit
                temp_num = local_limit - len(most_affected_nodes)
                #temp_nodes = list(np.random.choice(most_affected_nodes + node_add, temp_num, replace=True))
                temp_nodes = randomly_select(most_affected_nodes + node_add, temp_num)
                most_affected_nodes.extend(temp_nodes)
            tabu_nodes = list(set(node_add + most_affected_nodes))
            other_nodes = [node for node in G1.nodes() if node not in tabu_nodes]
            #random_nodes = list(np.random.choice(other_nodes, global_limit, replace=False))
            random_nodes = randomly_select(other_nodes, global_limit)
            node_update_list = node_add + most_affected_nodes + random_nodes
        else:
            print('nothing changed... For fairness, randomly update some as scheme 2 does')
            all_nodes = [node for node in G1.nodes()]
            #random_nodes = list(np.random.choice(all_nodes, num_limit, replace=False))
            random_nodes = randomly_select(all_nodes, num_limit)
            node_update_list = node_add + random_nodes
        """
    lack = local_limit - len(most_affected_nodes)
    tabu_nodes = set(node_add + most_affected_nodes)
    other_nodes = list(set(G1.nodes()) - tabu_nodes)
    #random_nodes = list(np.random.choice(other_nodes, global_limit + lack, replace=False))
    random_nodes = randomly_select(other_nodes, global_limit+lack)
    node_update_list = node_add + most_affected_nodes + random_nodes

    reservoir_key_list = list(reservoir_dict.keys())
    node_update_set = set(node_update_list)
    for node in node_update_set:
        if node in reservoir_key_list:
            del reservoir_dict[node]
    #t2 = time.time()
    #print(f'--> node selecting time; time cost: {(t2-t1):.2f}s')
    """
    print('num_limit', num_limit, 'local_limit', local_limit, 'global_limit', global_limit)
    print(f'# nodes added {len(node_add)}, # nodes deleted {len(node_del)}, # nodes updated {len(node_update_list)}')
    #print(f'# nodes affected {len(node_affected)}, # nodes most affected {len(most_affected_nodes)}')
    print(f'num of nodes in reservoir with accumulated changes but not updated {len(list(reservoir_dict))}')
    """
    print(f'num_limit {num_limit}, local_limit {local_limit}, global_limit {global_limit}, # nodes updated {len(node_update_list)}')
    return node_update_list, reservoir_dict

def select_most_affected_nodes(G0, G1, num_limit_return_nodes, reservoir_dict, exist_node_affected):
    ''' return num_limit_return_nodes to be updated
         based on the ranking of the accumulated changes w.r.t. their local connectivity
    '''
    most_affected_nodes = []
    for node in exist_node_affected:
        nbrs_set1 = set(nx.neighbors(G=G1, n=node))
        nbrs_set0 = set(nx.neighbors(G=G0, n=node))
        changes = len(nbrs_set1.union(nbrs_set0) - nbrs_set1.intersection(nbrs_set0))
        if node in reservoir_dict.keys():
            reservoir_dict[node] += changes  # accumulated changes
        else:
            reservoir_dict[node] = changes  # newly added changes

    if len(exist_node_affected) > num_limit_return_nodes:
        reservoir_dict_score = {}
        for node in exist_node_affected:
            reservoir_dict_score[node] = reservoir_dict[node] / G0.degree[node]
        # worse case O(n) https://docs.scipy.org/doc/numpy/reference/generated/numpy.partition.html
        # the largest change at num_limit_return_nodes will be returned
        cutoff_score = np.partition(list(reservoir_dict_score.values()), -num_limit_return_nodes, kind='introselect')[
            -num_limit_return_nodes]
        cnt = 0
        for node in reservoir_dict_score.keys():
            if reservoir_dict_score[
                node] >= cutoff_score:  # fix bug: there might be multiple equal cutoff_score nodes...
                if cnt == num_limit_return_nodes:  # fix bug: we need exactly the number of limit return nodes...
                    break
                most_affected_nodes.append(node)
                cnt += 1
        print(most_affected_nodes, " most affected nodes")
    else:  # NOTE: len(exist_node_affected) <= num_limit_return_nodes
        most_affected_nodes = exist_node_affected
    return most_affected_nodes, reservoir_dict

def simulate_walks(nx_graph, num_walks = 20, walk_length = 80, restart_prob=None, affected_nodes=None):
    '''
    Repeatedly simulate random walks from each node
    '''
    G = nx_graph
    walks = []

    if affected_nodes == None:  # simulate walks on every node in the graph [offline]
        nodes = list(G.nodes())
    else:  # simulate walks on affected nodes [online]
        nodes = list(affected_nodes)

    if restart_prob == None:  # naive random walk
        t1 = time.time()
        for walk_iter in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(random_walk(nx_graph=G, start_node=node, walk_length=walk_length))
        t2 = time.time()
        print(f'random walk sampling, time cost: {(t2-t1):.2f}')
    else:  # random walk with restart
        t1 = time.time()
        for walk_iter in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(random_walk_restart(nx_graph=G, start_node=node, walk_length=walk_length,
                                                 restart_prob=restart_prob))
        t2 = time.time()
        print(f'random walk sampling, time cost: {(t2-t1):.2f}')
    return walks


def random_walk(nx_graph, start_node, walk_length):
    '''
    Simulate a random walk starting from start node
    '''
    G = nx_graph
    walk = [start_node]

    while len(walk) < walk_length:
        cur = walk[-1]
        cur_nbrs = list(G.neighbors(cur))
        if len(cur_nbrs) > 0:
            walk.append(random.choice(cur_nbrs))
        else:
            break
    return walk


def random_walk_restart(nx_graph, start_node, walk_length, restart_prob):
    '''
    random walk with restart
    restart if p < restart_prob
    '''
    G = nx_graph
    walk = [start_node]

    while len(walk) < walk_length:
        p = random.uniform(0, 1)
        if p < restart_prob:
            cur = walk[0]  # restart
            walk.append(cur)
        else:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
    return walk

def format_node_IDs(original_sentices):
    new_all_sentences = []
    for i in range(len(original_sentices)):
        new_sentence = []
        for j in range(len(original_sentices[i])):
            current_ID = original_sentices[i][j]
            new_ID = str(current_ID[0])+"_"+str(current_ID[1])
            new_sentence.append(new_ID)
        new_all_sentences.append(new_sentence)
    return new_all_sentences

def get_pair_counts(sentences, windows, previous_count = []):
    for i in range(len(sentences)):
        current_sentence = sentences[i]


def sentences_count(my_list):
    import collections
    new_list = []
    for items in my_list:
        for item in items:
            new_list.append(item)
    c = collections.Counter(new_list)
    #print(c)
    return c


def draw_graph_with_node_frequency(graph, count, colors, node_add, old_graph, graph_ID):

    node_edge_color_map = []
    nodes_ID = list(graph.nodes)
    for i in range(len(nodes_ID)):
        current_node = nodes_ID[i]
        """
        if current_node in selected_nodes:
            node_inside_color_map.append('blue')
        else:
            node_inside_color_map.append('yellow')
        """
        if current_node in node_add:
            node_edge_color_map.append('red')
        else:
            node_edge_color_map.append('black')
    edge_color_map = []
    all_edges = list(graph.edges)
    for i in range(len(all_edges)):
        current_edge = all_edges[i]
        if old_graph.has_edge(current_edge[0], current_edge[1]):
            edge_color_map.append('black')
        else:
            edge_color_map.append('red')
    #nx.draw(graph, node_color = color_map)
    #nx.draw(graph, pos=pos_calculation(graph), node_color = color_map)

    #------------------------------------------------------------------------
    count_keys = list(count.keys())
    possible_color_count = len(colors)

    maximum = -1
    minimum = 100000
    for i in range(len(count_keys)):
        single_count = count[count_keys[i]]
        if maximum < single_count:
            maximum = single_count
        if minimum > single_count:
            minimum = single_count
    minimum = minimum-1
    color_slot = []
    slot = (maximum-minimum)/len(colors)
    for i in range(len(colors)):
        color_slot.append(minimum+i*slot)
    nodes_ID = list(graph.nodes)
    color_map = ['']*len(nodes_ID)

    for i in range(len(count_keys)):
        color_name_index = bisect.bisect_left(color_slot, count[count_keys[i]])-1
        ID_index = nodes_ID.index(count_keys[i])
        color_map[ID_index] = colors[color_name_index]
    for i in range(len(color_map)):
        if color_map[i] == '':
            color_map[i] = 'w'
    #nx.draw(graph, node_color = color_map)
    #nx.draw(graph, pos=pos_calculation(graph), node_color=color_map, node_size = 300)

    grahp_pos = pos_calculation(graph)
    for k, v in grahp_pos.items():
        # Shift the x values of every node by 10 to the right
        v[0] = v[0] + 3

    nx.draw_networkx_nodes(graph, pos=grahp_pos, node_color = color_map, linewidths = 1, edgecolors= node_edge_color_map, node_size=30)
    nx.draw_networkx_edges(graph, pos=grahp_pos, edge_color= edge_color_map)

    #plt.show()


def grey_colors():
    colors = ['lightgrey', 'silver', 'darkgrey', 'gray', 'dimgrey', 'black']
    return colors

def blue_colors():
    colors = ['b','blue','mediumblue','darkblue','navy','midnightblue']
    return colors

def remove_edges(graph, nodes):
    for i in range(len(nodes)):
        current_node = nodes[i]
        temp_1 = graph.copy()
        remaining_attemp = 0
        while(True):
            neighbors = list(temp_1.neighbors(current_node))
            temp_2 = temp_1.copy()
            random_number = random.sample(range(0, len(neighbors)), 1)[0]
            temp_2.remove_edge(current_node, neighbors[random_number])
            if ((nx.is_connected(temp_2) == False)):
                remaining_attemp += 1
            else:
                temp_1 = temp_2.copy()

            if remaining_attemp >= 2:
                graph = temp_1
                break

    return graph


def create_demos(m, n, use_reservior = True, initial_frequency = False, combine_frequency = False, local_global_count = 2):
    """
    this will generate 4 roads
    :param m:
    :param n:
    :return:
    """

    g0 = nx.generators.lattice.grid_2d_graph(m, n)
    draw_single_graph(g0)
    if initial_frequency == True:
        all_nodes = (g0.nodes)
        sentences = simulate_walks(nx_graph=g0, affected_nodes=all_nodes, num_walks=10, walk_length=10)
        count = sentences_count(sentences)
    else:
        g0_nodes_list = list(g0.nodes)
        count = {}
        for i in range(len(g0_nodes_list)):
            count[g0_nodes_list[i]] = 0


    g1 = remove_nodes_down_right_corner(g0.copy(), m, n)

    nodes_remove_edges = [(1,(n-2)),(3,(n-2)),(2,(n-2))]
    g1 = remove_edges(g1.copy(), nodes_remove_edges)
    for i in range(local_global_count+1):
        local_global_selected = i * 1/local_global_count
        count, reservior = draw_figures_two_graphs(g0, g1, count, local_global = local_global_selected, combine_frequency = combine_frequency, graph_ID = 0, parameter_number = i)



    g2 = remove_nodes_down_right_corner(g1.copy(), m, n, offset=4)

    nodes_remove_edges = [(1,(n-6)),(3,(n-6)),(2,(n-6))]
    g2 = remove_edges(g2.copy(), nodes_remove_edges)
    if use_reservior == False:
        reservior = {}
    for i in range(local_global_count+1):
        local_global_selected = i * 1 / local_global_count
        count, reservior = draw_figures_two_graphs(g1, g2, count, local_global = local_global_selected, reservior= reservior, combine_frequency = combine_frequency, graph_ID = 1, parameter_number = i)


    g3 = remove_nodes_down_right_corner(g2.copy(), (m-1), n, add_node_number = 8)

    nodes_remove_edges = [(4,(n-1)), (4, (n-3)),(4, (n-5))]
    g3 = remove_edges(g3.copy(), nodes_remove_edges)
    if use_reservior == False:
        reservior = {}
    for i in range(local_global_count+1):
        local_global_selected = i * 1 / local_global_count
        count, reservior = draw_figures_two_graphs(g2, g3, count, local_global = local_global_selected, reservior= reservior, combine_frequency = combine_frequency, graph_ID = 2, parameter_number = i)

    plt.show()

    """
    g4 = add_new_nodes_down_right_corner(g3.copy(), (m+2), n, add_node_number = 8)
    g4= add_new_edges_for_new_nodes(g4)
    nodes_add_edges = [(1,(n-4)),(3,(n-4)),(2,(n-4))]
    g4 = add_edges(g4.copy(), nodes_add_edges)
    if use_reservior == False:
        reservior = {}
    for i in range(local_global_count+1):
        local_global_selected = i * 1 / local_global_count
        count, reservior = draw_figures_two_graphs(g3, g4, count, local_global = local_global_selected, reservior= reservior, combine_frequency = combine_frequency, graph_ID = 3, parameter_number = i)



    g5 = add_new_nodes_down_right_corner(g4.copy(), m, n, add_node_number = 4, offset= 8)
    g5 = add_new_edges_for_new_nodes(g5)
    nodes_add_edges = [(1,(n-7)),(3,(n-7)),(2,(n-7))]
    g5 = add_edges(g5.copy(), nodes_add_edges)
    if use_reservior == False:
        reservior = {}
    for i in range(local_global_count + 1):
        local_global_selected = i * 1 / local_global_count
        count, reservior = draw_figures_two_graphs(g4, g5, count, local_global = local_global_selected, reservior= reservior, combine_frequency = combine_frequency, graph_ID = 4, parameter_number = i)


    g6 = add_new_nodes_down_right_corner(g5.copy(), m, n, add_node_number = 4, offset= 8)
    g6 = add_new_edges_for_new_nodes(g6)
    nodes_add_edges = [(6,(n-1)), (6, (n-3)),(6, (n-5))]
    g6 = add_edges(g6.copy(), nodes_add_edges)
    if use_reservior == False:
        reservior = {}
    count, reservior = draw_figures_two_graphs(g5, g6, count, reservior= reservior, combine_frequency = combine_frequency, graph_ID = 5)



    g7 = add_new_nodes_down_right_corner(g6.copy(), (m+1), n, add_node_number = 4, offset= 8)
    g7 = add_new_edges_for_new_nodes(g7)
    nodes_add_edges = [(8,(n-1)), (8, (n-3)),(8, (n-5))]
    g7 = add_edges(g6.copy(), nodes_add_edges)
    if use_reservior == False:
        reservior = {}
    count, reservior = draw_figures_two_graphs(g6, g7, count, reservior= reservior, combine_frequency = combine_frequency, graph_ID = 6)

    #add new edges for g3
    #add new edges for g3
    """
def combine_count(current_count, previous_count):
    current_count_keys = list(current_count.keys())
    previous_count_keys = list(previous_count.keys())
    all_keys = list(set(current_count.keys()).union(set(previous_count.keys())))

    combined_count = {}
    for i in range(len(all_keys)):
        single_key = all_keys[i]
        if single_key in current_count_keys:
            single_count = current_count[single_key]
        else:
            single_count = 0

        if single_key in previous_count_keys:
            previous_single_count = previous_count[single_key]
        else:
            previous_single_count = 0
        combined_count[single_key] = previous_single_count + single_count
    return combined_count

def draw_single_graph(g):
    nx.draw_networkx_nodes(g, pos=pos_calculation(g), node_color='yellow', linewidths=1, edgecolors='black', node_size=30)
    nx.draw_networkx_edges(g, pos=pos_calculation(g), edge_color='black')
    my_dpi = 100
    #plt.figure(figsize=(2, 3), dpi=my_dpi)

    plt.show()

def draw_figures_two_graphs(g0, g1, previous_count, reservior = {},  scheme = 3, limit = 0.1, local_global = 1, graph_ID = 1, combine_frequency = True, parameter_number = 0):
    print("local global", local_global)
    g0c = g0.copy()
    g1c = g1.copy()
    edge_add = edge_s1_minus_s0(s1=set(g1c.edges()),
                                s0=set(g0c.edges()))  # one may directly use streaming added edges if possible
    node_affected_by_edge_add = unique_nodes_from_edge_set(edge_add)  # unique
    node_add = [node for node in node_affected_by_edge_add if node not in g0c.nodes()]

    node_update_list, reservoir_dict = node_selecting_scheme(g0, g1, reservoir_dict = reservior, limit=limit, scheme = scheme, local_global = local_global)
    sentences = simulate_walks(nx_graph=g1, affected_nodes=node_update_list, num_walks=10, walk_length=10)
    count = sentences_count(sentences)

    all_count = combine_count(count, previous_count)

    draw_selected_nodes(g1, node_update_list, node_add, g0, graph_ID, parameter_number)
    if combine_frequency == True:
        draw_graph_with_node_frequency(g1, all_count, grey_colors(), node_add, g0, graph_ID)
    else:
        draw_graph_with_node_single_frequency(g1, count, grey_colors(), node_add, g0, graph_ID, parameter_number)

    print("graph_ID:",graph_ID, "  parameter number:",parameter_number)

    return all_count, reservoir_dict

def draw_graph_with_node_single_frequency(graph, count, colors, node_add, old_graph, graph_ID, parameter_number = 0):

    node_edge_color_map = []
    nodes_ID = list(graph.nodes)
    for i in range(len(nodes_ID)):
        current_node = nodes_ID[i]
        """
        if current_node in selected_nodes:
            node_inside_color_map.append('blue')
        else:
            node_inside_color_map.append('yellow')
        """
        if current_node in node_add:
            node_edge_color_map.append('red')
        else:
            node_edge_color_map.append('black')
    edge_color_map = []
    all_edges = list(graph.edges)
    for i in range(len(all_edges)):
        current_edge = all_edges[i]
        if old_graph.has_edge(current_edge[0], current_edge[1]):
            edge_color_map.append('black')
        else:
            edge_color_map.append('red')
    #nx.draw(graph, node_color = color_map)
    #nx.draw(graph, pos=pos_calculation(graph), node_color = color_map)

    #------------------------------------------------------------------------
    count_keys = list(count.keys())
    possible_color_count = len(colors)

    maximum = -1
    minimum = 100000
    for i in range(len(count_keys)):
        single_count = count[count_keys[i]]
        if maximum < single_count:
            maximum = single_count
        if minimum > single_count:
            minimum = single_count
    minimum = minimum-1
    color_slot = []
    slot = (maximum-minimum)/len(colors)
    for i in range(len(colors)):
        color_slot.append(minimum+i*slot)
    nodes_ID = list(graph.nodes)
    color_map = ['']*len(nodes_ID)

    for i in range(len(count_keys)):
        color_name_index = bisect.bisect_left(color_slot, count[count_keys[i]])-1
        ID_index = nodes_ID.index(count_keys[i])
        #print(ID_index," ID index ", len(nodes_ID))
        #print(color_name_index, "color name index ", len(colors))
        #print(color_slot)
        #print(count[count_keys[i]])
        color_map[ID_index] = colors[color_name_index]
    for i in range(len(color_map)):
        if color_map[i] == '':
            color_map[i] = 'w'
    #nx.draw(graph, node_color = color_map)
    #nx.draw(graph, pos=pos_calculation(graph), node_color=color_map, node_size = 300)

    graph_pos = pos_calculation(graph)
    for k, v in graph_pos.items():
        # Shift the x values of every node by 10 to the right
        v[0] = v[0] + 1.05 + 2.2*parameter_number
        v[1] = v[1] - 2.2*graph_ID


    #nx.draw_networkx_nodes(graph, pos=graph_pos, node_color = color_map, linewidths = 0.5, edgecolors= node_edge_color_map, node_size=13)
    #nx.draw_networkx_edges(graph, pos=graph_pos, edge_color= edge_color_map)

    nx.draw_networkx_nodes(graph, pos=graph_pos, node_color = color_map, linewidths = 0.5, edgecolors= 'black', node_size=11)#heat map
    #nx.draw_networkx_edges(graph, pos=graph_pos, edge_color= 'black')

    grahp_pos = pos_calculation(graph)

    #plt.show()


def draw_selected_nodes(graph, selected_nodes, node_add, old_graph, graph_ID, parameter_number = 0):
    node_inside_color_map = []
    node_edge_color_map = []
    nodes_ID = list(old_graph.nodes)
    new_node_ID = list(graph.nodes)
    for i in range(len(new_node_ID)):
        current_node = new_node_ID[i]
        if current_node in selected_nodes:
            node_inside_color_map.append('blue')
        else:
            node_inside_color_map.append('white')
        if current_node in node_add:
            node_edge_color_map.append('red')
        else:
            node_edge_color_map.append('black')
    edge_color_map = []

    all_edges = list(graph.edges)
    all_edges_old = list(old_graph.edges)

    temporal_graph = graph.copy()

    for i in range(len(all_edges_old)):
        current_edge = all_edges_old[i]
        if temporal_graph.has_edge(current_edge[0], current_edge[1]) == False:
            #print(" find a non existence edge")
            if temporal_graph.has_node(current_edge[0]) and temporal_graph.has_node(current_edge[1]):
                #print(" add a new edge for temporal graph")
                temporal_graph.add_edge(current_edge[0], current_edge[1])

    temporal_graph_all_edges = list(temporal_graph.edges)

    for i in range(len(temporal_graph_all_edges)):
        current_edge = temporal_graph_all_edges[i]
        if graph.has_edge(current_edge[0], current_edge[1]):
            edge_color_map.append('black')
        else:
            edge_color_map.append('red')
    #nx.draw(graph, node_color = color_map)
    #nx.draw(graph, pos=pos_calculation(graph), node_color = color_map)



    graph_pos = pos_calculation(temporal_graph)
    for k, v in graph_pos.items():
        # Shift the x values of every node by 10 to the right
        v[0] = v[0] + 2.2*parameter_number
        v[1] = v[1] - 2.2*graph_ID



    nx.draw_networkx_nodes(temporal_graph, graph_pos, node_color = node_inside_color_map, linewidths = 0.5, edgecolors= node_edge_color_map, node_size=8)
    nx.draw_networkx_edges(temporal_graph, graph_pos, edge_color= edge_color_map, width=1.05)
    #nx.draw_networkx_nodes(graph, graph_pos, node_color=node_inside_color_map, linewidths=0.5, edgecolors=node_edge_color_map, node_size=9)
    #nx.draw_networkx_edges(graph, graph_pos, edge_color= 'black')

if __name__ == '__main__':
    """
    g0, g1 = create_demo_network(20,30)
    node_update_list, reservoir_dict = node_selecting_scheme(g0,g1,scheme=1)
    print(node_update_list)
    #print(random.sample(range(0, 4), 3))
    sentences = simulate_walks(nx_graph=g1, affected_nodes=node_update_list, num_walks = 10, walk_length = 10)
    #print(len(sentences))
    #for i in range(len(sentences)):
    #    print(sentences[i])
    count = sentences_count(sentences)
    print(count)
    print(count[(3,1)])
    draw_selected_nodes(g1, node_update_list)
    draw_graph_with_node_frequency(g1,count,grey_colors())
    """
    #draw_graph_with_node_frequency(g1,count,blue_colors())
    #one part may cause trouble  sample > node

    create_demos(8,20, use_reservior = True, initial_frequency = False, combine_frequency = False)
    #
    """
    a = set([1,2])
    b = set([2,3])
    print(a.union(b))
    """