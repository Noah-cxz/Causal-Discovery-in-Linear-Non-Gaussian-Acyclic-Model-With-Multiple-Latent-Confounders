import numpy as np


def get_Adj_set(skeleton_matrix):
    """ Quarry by Adj_set[variable] = {adjacent variable set}
    """
    dim = skeleton_matrix.shape[1]
    Adj_set = {}

    for i in range(dim):
        Adj_set[i] = set()

    for i in range(dim):
        for j in range(dim):
            if i != j:
                if skeleton_matrix[i][j] == 1:
                    Adj_set[i].add(j)

    return Adj_set


def show_nx_graph(array, direct=True, cycle_style=True):
    """From numpy array to nx_graph.
    :param array:
    :param direct:
    :return:
    """
    import networkx as nx
    import matplotlib.pyplot as plt

    if direct:
        G = nx.from_numpy_array(array.T,create_using=nx.DiGraph)
        if cycle_style is True:
            pos = nx.circular_layout(G)
        else:
            pos = nx.random_layout(G)

        plt.figure(figsize=(5,2.7), dpi=120) # original: 90
        nx.draw(G, pos=pos, with_labels=True, node_color='black', font_color='white', font_size=15, width=1.25, arrowsize=20)
        # plt.savefig("test.png")
    else:
        G = nx.from_numpy_array(array.T)
        if cycle_style is True:
            pos = nx.circular_layout(G)
        else:
            pos = nx.random_layout(G)

        plt.figure(figsize=(5, 2.7), dpi=120)
        nx.draw(G, pos=pos, with_labels=True, node_color='black', font_color='white', font_size=15, width=1.25)
        # plt.savefig("test.png")


def subset(S, size):
    """ [1,2,3]/{1,2,3} -> [{1, 2}, {1, 3}, {2, 3}]
    """
    import itertools
    subset = list(map(set, itertools.combinations(S, size)))

    return subset

def check_vector(X):
    n = X.shape[0]
    return X.reshape(n, 1)if len(X.shape) == 1 else X

def check_list(U):
    if type(U) is not list:
        U_list = []
        U_list.append(U)
        return U_list

    return U
