# import metis
import torch
import random
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
# from cdlib import algorithms, viz
# با کتابخانه بالا در کولب مشکل دارم!
from karateclub.community_detection.overlapping import DANMF
from karateclub import SymmNMF #NNSED #EgoNetSplitter

class ClusteringMachine(object):
    """
    Clustering the graph, feature set and target.
    """
    def __init__(self, args, graph, features, target):
        """
        :param args: Arguments object with parameters.
        :param graph: Networkx Graph.
        :param features: Feature matrix (ndarray).
        :param target: Target vector (ndarray).
        """
        self.args = args
        self.graph = graph
        self.features = features
        self.target = target
        self._set_sizes()

    def _set_sizes(self):
        """
        Setting the feature and class count.
        """
        self.feature_count = self.features.shape[1] 
        self.class_count = np.max(self.target)+1

    def decompose(self):
        """
        Decomposing the graph, partitioning the features and target, creating Torch arrays.
        """
        if self.args.clustering_method == "metis":
            print("\nMetis graph clustering started.\n")
            self.metis_clustering()
        elif self.args.clustering_method == "random":
            print("\nRandom graph clustering started.\n")
            self.random_clustering()
        elif self.args.clustering_method == "danmf":
            # print("\nDANMF clustering started.\n")
            self.danmf_clustering()
        elif self.args.clustering_method == "graph":
            print("\ngraph clustering started.\n")
            self.graph_clustering()
        
        # print('clusters, len',self.clusters, len(self.clusters))
        # self.cluster_lens = np.zeros(len(self.clusters))
        # for i in range(len(self.cluster_membership)):
        #     if isinstance(self.cluster_membership[i], int):
        #         self.cluster_lens[self.cluster_membership[i]] += 1
        #     else:
        #         for j in self.cluster_membership[i]:
        #             self.cluster_lens[j] +=1 

        # print(self.cluster_lens)        
        # print('Clusters info: Min, Max, Sum element numbers:',\
        #  np.min(self.cluster_lens), np.max(self.cluster_lens), np.sum(self.cluster_lens))

        self.general_data_partitioning()
        self.transfer_edges_and_nodes()


    def random_clustering(self):
        """
        Random clustering the nodes.
        """
        self.clusters = [cluster for cluster in range(self.args.cluster_number)]
        self.cluster_membership = {node: random.choice(self.clusters) for node in self.graph.nodes()}

    def metis_clustering(self):
        """
        Clustering the graph with Metis. For details see:
        """
        (st, parts) = metis.part_graph(self.graph, self.args.cluster_number)
        self.clusters = list(set(parts))
        self.cluster_membership = {node: membership for node, membership in enumerate(parts)}

    def danmf_clustering(self):
        """
        Clustering the graph with DANMF. For details see:
        """
        num_labels = {'CiteSeer':6, 'Cora':7, 'PubMed':3, 'WikiCS':10}

        model = DANMF(layers=[32,2*num_labels[self.args.dataset_name]], pre_iterations = 500, iterations = 200)
        # model = EgoNetSplitter(1.0) # ماتریس احتمال بر نمی‌گرداند
        # model = NNSED() # این هم همچنین
        # model = SymmNMF() # در شبکه خطا میده!
        model.fit(self.graph)

        values = model.get_memberships().values()
        # print('values', values)
        values_list = list(values)
        # print('values_list==11', 11 in values_list)

        if self.args.clustering_overlap == False:
            near_clusters = values
        else:
            # نرم دوی هر سطر ماتریس برابر یک می شود
            # DANMF ->P, SymmNMF->W
            # P = normalize(model._W, axis=1)
            P = normalize(model._P, axis=1)
            # print('P.shape', P.shape)
            near_clusters = []
            for i in range(P.shape[0]):
                row = P[i]
                max_in_row = np.max(row)
                
                npw = np.where(row >= (max_in_row*self.args.membership_closeness))
                tmp = npw[0].tolist()
                # برای تعداد زیادی از سطرها همه مقادیر صفر است.
                if max_in_row == 0:
                    cluster_indices = [tmp[0]]    
                    # print('max =0:', cluster_indices)
                else:
                # نگهداری فقط خوشه‌هایی که در لیست اولیه بوده اند
                    cluster_indices = [x for x in tmp if x in values_list]
                    # print('max !=0:', cluster_indices)
                # print(type(row), row, "max in row", max_in_row, 'npw', npw, 'tmp', tmp)
                near_clusters.append(cluster_indices)

        # باید خوشه‌هایی که نیستند حذف شده و سایر اندیس ها اصلاح شوند

        # print("\n near_clusters", type(near_clusters), near_clusters)
        self.clusters = list(set(values_list)) # وقتی یک خوشه خالی باشه گیر داره
        # مشکل باید از جای دیگری باشه. شماره ۱۱ در لیست اصلی نیست اما در دومی هست در کورا
        # self.clusters = list(np.arange(0,np.max(values_list)+1))
        self.cluster_membership = {node: membership for node, membership in enumerate(near_clusters)}
        
    def graph_clustering(self):
        """
        Clustering the graph with other graph clustering algorithms
        """
        # print('principled_clustering clustering')
        # coms = algorithms.kclique(G, k=4)
        # coms = algorithms.louvain(G)
        # coms = algorithms.girvan_newman(G, level=4)
        # coms = algorithms.aslpaw(G)
        # coms = algorithms.mnmf(self.graph, clusters=self.args.cluster_number)
        # coms = algorithms.umstmo(G)
        # coms = algorithms.principled_clustering(self.graph, cluster_count=self.args.cluster_number)
        # coms = algorithms.em(self.graph, k=self.args.cluster_number)
        coms = algorithms.conga(self.graph, number_communities=self.args.cluster_number)
        
        
        count = sum( [ len(listElem) for listElem in coms.communities])
        print('Number of nodes in clustering:', count)
        parts = [0] * len(self.graph.nodes()) #count
        for i in range(len(coms.communities)):
            for x in coms.communities[i]:
                parts[x] = i
        self.clusters = list(set(parts))
        self.cluster_membership = {node: membership for node, membership in enumerate(parts)}

    def general_data_partitioning(self):
        """
        Creating data partitions and train-test splits.
        """
        self.sg_nodes = {}
        self.sg_edges = {}
        self.sg_train_nodes = {}
        self.sg_test_nodes = {}
        self.sg_features = {}
        self.sg_targets = {}
        # print('\nNum Clusters:', len(self.clusters))
        self.ClusterNodes = []
        for cluster in self.clusters:
            # M.Amintoosi
            # subgraph = self.graph.subgraph([node for node in sorted(self.graph.nodes()) if cluster in self.cluster_membership[node]])
            
            if self.args.clustering_overlap == True:
                subgraph = self.graph.subgraph([node for node in sorted(self.graph.nodes()) if cluster in self.cluster_membership[node]])
            else:
                subgraph = self.graph.subgraph([node for node in sorted(self.graph.nodes()) if self.cluster_membership[node] == cluster])
            # print('len subgraph', len(subgraph.nodes()))
            self.ClusterNodes.append(len(subgraph.nodes()))
            self.sg_nodes[cluster] = [node for node in sorted(subgraph.nodes())]
            mapper = {node: i for i, node in enumerate(sorted(self.sg_nodes[cluster]))}
            self.sg_edges[cluster] = [[mapper[edge[0]], mapper[edge[1]]] for edge in subgraph.edges()] +  [[mapper[edge[1]], mapper[edge[0]]] for edge in subgraph.edges()]
            self.sg_train_nodes[cluster], self.sg_test_nodes[cluster] = train_test_split(list(mapper.values()), test_size = self.args.test_ratio)
            self.sg_test_nodes[cluster] = sorted(self.sg_test_nodes[cluster])
            self.sg_train_nodes[cluster] = sorted(self.sg_train_nodes[cluster])
            self.sg_features[cluster] = self.features[self.sg_nodes[cluster],:]
            self.sg_targets[cluster] = self.target[self.sg_nodes[cluster],:]
        # print("\nNumber of clusters' nodes:", np.sum(self.ClusterNodes))
    def transfer_edges_and_nodes(self):
        """
        Transfering the data to PyTorch format.
        """
        for cluster in self.clusters:
            self.sg_nodes[cluster] = torch.LongTensor(self.sg_nodes[cluster])
            self.sg_edges[cluster] = torch.LongTensor(self.sg_edges[cluster]).t()
            self.sg_train_nodes[cluster] = torch.LongTensor(self.sg_train_nodes[cluster])
            self.sg_test_nodes[cluster] = torch.LongTensor(self.sg_test_nodes[cluster])
            self.sg_features[cluster] = torch.FloatTensor(self.sg_features[cluster])
            self.sg_targets[cluster] = torch.LongTensor(self.sg_targets[cluster])
