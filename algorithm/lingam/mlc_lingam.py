import numpy as np
import networkx as nx
import copy
import time
from sklearn.utils import check_array

# Accompanying code for MLCLiNGAM as follow
from utils.basic_causal_tool import get_skeleton_from_stable_pc, residual_by_linreg, fisher_hsic
from utils.data_processing import get_Adj_set, check_vector, check_list, subset


class MLCLiNGAM():
    def __init__(self, alpha=0.05):
        """
        References
        ----------
        Chen W, Cai R, Zhang K, et al.
        Causal discovery in linear non-gaussian acyclic model with multiple latent confounders
        [J]. IEEE Transactions on Neural Networks and Learning Systems, 2021.
        """
        self._alpha = alpha
        self._skeleton = None
        self._adjacency_matrix = None
        self._parents_set = {}
        self._U_res = []
        self._cliques = []
        self._stage1_time = 0
        self._stage2_time = 0
        self._stage3_time = 0


    def fit(self, X):
        # Check parameter
        d = X.shape[1]
        X = check_array(X)

        # Initialize parent set
        for i in range(d):
            self._parents_set[i] = set()

        # Stage1: causal skeleton reconstruction(PC-stable algorithm)
        self._stage_1_learning(X)

        # Stage2: partial causal orders identification
        self._stage_2_learning(X)

        # Stage3: latent confounders' detection
        self._stage_3_learning(X)

        return self


    def _stage_1_learning(self, X):
        skeleton, running_time = get_skeleton_from_stable_pc(X, return_time=True)

        self._skeleton = copy.copy(skeleton)
        self._adjacency_matrix = copy.copy(skeleton)
        self._stage1_time = running_time

        return self


    def _stage_2_learning(self, X):
        start = time.perf_counter()

        Adj_set = get_Adj_set(self._skeleton) # Quarry by Adj_set[variable] = {adjacent variable set}
        d = X.shape[1]
        X_ = copy.copy(X)
        U = np.arange(d)

        # Identify exogenous variable.
        repeat = True
        while repeat:
            if len(U) == 1:
                break
            repeat = False

            for i in U:
                is_exo = True
                i_adj = Adj_set[i] & set(U)

                _ = i_adj.copy()
                for j in _:
                    if self._check_identity(i, j):
                        i_adj.remove(j)

                if len(i_adj) == 0:
                    is_exo = False
                    continue

                for j in i_adj:
                    residual = residual_by_linreg(X=X_[:, i], y=X_[:, j])
                    pval = fisher_hsic(check_vector(X_[:, i]), residual)

                    is_exo = True if pval > self._alpha else False

                if is_exo:
                    repeat = True
                    U = U[U != i]
                    for j in i_adj:
                        self._orient_adjacency_matrix(explanatory=i, explain=j)
                        self._parents_set[j].add(i)
                        residual = residual_by_linreg(X=X_[:, i], y=X_[:, j])  # residual replacement
                        X_[:, j] = residual.ravel()

        if len(U) > 2:
            # Identify leaf variable.
            repeat = True
            while repeat:
                if len(U) == 1:
                    break
                repeat = False

                for i in U:
                    i_adj = Adj_set[i] & set(U)

                    _ = i_adj.copy()
                    for j in _:
                        if self._check_identity(i, j):
                            i_adj.remove(j)

                    if len(i_adj) == 0:
                        continue

                    residual = residual_by_linreg(X=X_[:, list(i_adj)], y=X_[:, i])
                    pval = fisher_hsic(check_vector(X_[:, list(i_adj)]), residual)

                    if pval > self._alpha:
                        repeat = True
                        U = U[U != i]
                        self._orient_adjacency_matrix(explanatory=i_adj, explain=i)
                        self._parents_set[i].union(i_adj)

        self._U_res = U.copy()

        end = time.perf_counter()
        self._stage2_time = end - start
        return self


    def _stage_3_learning(self, X):
        start = time.perf_counter()

        if len(self._U_res) > 2:
            maximal_cliques = self._get_maximal_cliques()

            if len(maximal_cliques) == 0:
                end = time.perf_counter()
                self._stage3_time = end - start
                return self

            else:
                for maximal_clique in maximal_cliques:
                    # Remove effect of observed confounder outside clique.
                    X_ = copy.copy(X)
                    for vi in maximal_clique:
                        for vj in maximal_clique[vi:]:
                            if len(self._parents_set[vi] & self._parents_set[vj]) > 0:
                                confounder_set_out = self._parents_set[vi] & self._parents_set[vj]
                                for confounder in confounder_set_out:
                                    X_[:, vi] = (residual_by_linreg(X=X_[:, confounder], y=X_[:, vi])).ravel()
                                    X_[:, vj] = (residual_by_linreg(X=X_[:, confounder], y=X_[:, vj])).ravel()

                    for size in range(len(maximal_clique), (2-1), -1):
                        new_edge_determine = True

                        while new_edge_determine:
                            new_edge_determine = False
                            for maximal_clique_subset in subset(maximal_clique, size=size):
                                # "complete" refers to undirect graph which is the concept of clique.
                                complete = True
                                for vi in list(maximal_clique_subset):
                                    for vj in list(maximal_clique_subset)[vi:]:
                                        if self._check_identity(vi, vj):
                                            complete = False

                                if not complete:
                                    continue

                                U = np.array(list(maximal_clique_subset))

                                # Identify exogenous variable in a clique.
                                repeat = True
                                while repeat:
                                    if len(U) == 1:
                                        break
                                    repeat = False

                                    for i in U:
                                        is_exo = True
                                        i_adj = set(U) - {i}

                                        _ = i_adj.copy()
                                        for j in _:
                                            if self._check_identity(i, j):
                                                i_adj.remove(j)

                                        if len(i_adj) == 0:
                                            is_exo = False
                                            continue

                                        for j in i_adj:
                                            # Remove effect of observed confounder inside clique.
                                            if len(self._parents_set[i] & set(maximal_clique)) > 0:
                                                confounder_set_in = self._parents_set[i] & set(maximal_clique)
                                                explantory = copy.copy(confounder_set_in)
                                                explantory.add(i)
                                                residual = residual_by_linreg(X=X_[:, list(explantory)], y=X_[:, j])
                                            else:
                                                residual = residual_by_linreg(X=X_[:, i], y=X_[:, j])

                                            pval = fisher_hsic(check_vector(X_[:, i]), residual)

                                            is_exo = True if pval > self._alpha else False

                                        if is_exo:
                                            repeat = True
                                            new_edge_determine = True
                                            U = U[U != i]
                                            for j in i_adj:
                                                self._orient_adjacency_matrix(explanatory=i, explain=j)
                                                self._parents_set[j].add(i)

                                if len(U) > 2:
                                    # Identify leaf variable in a clique.
                                    repeat = True
                                    while repeat:
                                        if len(U) == 1:
                                            break
                                        repeat = False

                                        for i in U:
                                            i_adj = set(U) - {i}

                                            _ = i_adj.copy()
                                            for j in _:
                                                if self._check_identity(i, j):
                                                    i_adj.remove(j)

                                            if len(i_adj) == 0:
                                                continue

                                            # Remove effect of observed confounder inside clique.
                                            if len(self._parents_set[i] & set(maximal_clique)) > 0:
                                                confounder_set_in = self._parents_set[i] & set(maximal_clique)
                                                explantory = i_adj | confounder_set_in
                                                residual = residual_by_linreg(X=X_[:, list(explantory)], y=X_[:, i])
                                            else:
                                                residual = residual_by_linreg(X=X_[:, list(i_adj)], y=X_[:, i])

                                            pval = fisher_hsic(check_vector(X_[:, list(i_adj)]), residual)

                                            if pval > self._alpha:
                                                repeat = True
                                                new_edge_determine = True
                                                U = U[U != i]
                                                self._orient_adjacency_matrix(explanatory=i_adj, explain=i)
                                                self._parents_set[i].union(i_adj)

            for maximal_clique in maximal_cliques:
                complete = True
                undirect_exist = False
                for vi in maximal_clique:
                    for vj in maximal_clique[vi:]:
                        if vi != vj:
                            if self._check_identity(vi, vj):
                                complete = False
                            else:
                                undirect_exist = True

                if (not complete and undirect_exist):
                    self._cliques.append(maximal_clique)
                elif complete:
                    self._cliques.append(maximal_clique)
                else:
                    continue

            end = time.perf_counter()
            self._stage3_time = end - start
            return self


    def _check_identity(self, i, j):
        return True if ((i in self._parents_set[j]) or (j in self._parents_set[i])) else False


    def _orient_adjacency_matrix(self, explanatory, explain):
        i_adj = explanatory
        i = explain

        if type(i_adj) is int or type(i_adj) is np.int32:
            i_adj = check_list(i_adj)

        for j in i_adj:
            self._adjacency_matrix[i][j] = 1
            self._adjacency_matrix[j][i] = 0

        return self


    def _get_maximal_cliques(self):
        undirect_graph_nx = nx.from_numpy_array(self._skeleton)
        iter = nx.find_cliques(undirect_graph_nx)
        temp = [clique for clique in iter]

        maximal_cliques = []

        for item in temp:
            if len(item) > 1:
                complete = True # "complete" refers to undirect graph which is the concept of clique.
                for vi in item:
                    for vj in item[vi:]:
                        if self._check_identity(vi, vj):
                            complete = False
                if complete:
                    maximal_cliques.append(item)

        return maximal_cliques


    @property
    def skeleton_(self):
        return self._skeleton


    @property
    def adjacency_matrix_(self):
        return self._adjacency_matrix


    @property
    def cliques_(self):
        return self._cliques


    @property
    def parents_set_(self):
        return self._parents_set