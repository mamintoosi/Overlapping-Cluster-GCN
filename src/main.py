# -*- coding: utf-8 -*-
#
#    Copyright (C) 2021-2029 by
#    Mahmood Amintoosi <m.amintoosi@gmail.com>
#    All rights reserved.
#    BSD license.
#    This repository is based on Cluster-GCN
import torch
from parser import parameter_parser
from clustering import ClusteringMachine
from clustergcn import ClusterGCNTrainer
from utils import tab_printer, dataset_reader
import numpy as np
from tqdm.notebook import tqdm
# graph_reader, feature_reader, target_reader
from pandas import ExcelWriter
import pandas as pd
from pandas import DataFrame 
import time

# def exec_time(start, end):
#    diff_time = end - start
#    m, s = divmod(diff_time, 60)
#    h, m = divmod(m, 60)
#    s,m,h = int(round(s, 0)), int(round(m, 0)), int(round(h, 0))
   
# def main():
    # """
    # Parsing command line parameters, reading data, graph decomposition, fitting a ClusterGCN and scoring the model.
    # """
args = parameter_parser()
torch.manual_seed(args.seed)
# tab_printer(args)
# graph, features, target = dataset_reader(args)
# graph = graph_reader(args.edge_path)
# # print('Number of graph nodes: ',len(graph.nodes()))
# features = feature_reader(args.features_path)
# target = target_reader(args.target_path)
# print(features.shape, target.shape)
Scores = []
# for i in range(args.num_trial):
graph, features, target = dataset_reader(args)
# print(features.shape, target.shape)
clustering_machine = ClusteringMachine(args, graph, features, target)
clustering_machine.decompose()
start = time.time()    
gcn_trainer = ClusterGCNTrainer(args, clustering_machine)
gcn_trainer.train()
score = gcn_trainer.test()
Scores.append(score)
# print("\nF-1 score: {:.2f}".format(score))

# if args.num_trial>1:
#     print("\n\n Mean F-1 score: {:.2f}".format(np.mean(Scores)))

end = time.time()
run_time = end - start

# jadval = pd.DataFrame(
#     {
#      'Winner Membership Closeness': args.membership_closeness,
#      'dataset name': args.dataset_name,
#      'F-1 Score': score,
#      'Overlapped Nodes': np.sum(clustering_machine.ClusterNodes)/len(graph.nodes())-1,
#      'Run Time': run_time
#     }, index=[0])
    #  'All Clusters Nodes': np.sum(clustering_machine.ClusterNodes),
ds_report = [args.membership_closeness, args.dataset_name, score, \
    np.sum(clustering_machine.ClusterNodes)/len(graph.nodes()),\
        run_time ]

# sns.lineplot(x="k", y="apk",
#             hue="method_name",# style="min_freq",
#             data=jadval)

# file_name = 'results/{}_{}_{}_{}_{}_{}.xlsx'.format(working_file_name[:2], GT_file_name[:2], \
# node_objects, edge_objects, str(min_count), str(minFreq))
# writer = ExcelWriter(file_name)
# gf_df_sorted.to_excel(writer, sheet_name='gf_df_sorted')  # , index=False)
# writer.save()        

# if __name__ == "__main__":
#     main()
