import networkx
import numpy as np
import dgl
import torch
G = networkx.DiGraph()
G.add_edge(5,6)
G.add_edge(5,5)
G.add_edge(7,3)
G.add_edge(3,5)
subG:networkx.DiGraph=G.subgraph([3,5]).copy()
print(subG.nodes())
A=np.array(networkx.adjacency_matrix(G).todense())

# A.tolist()
print(A)
# print(G.nodes)
# G0 = G.to_undirected()  #将有向图转化为无向图（确保图为无向图）
# keys,degree =zip(*G0.degree())  #keys是节点标号，degree是对应节点的度(度序列)
# kmin=min(degree)
# kmax=max(degree)
# C = sorted(networkx.connected_components(G0), key=len, reverse=True)   #其中C是所有连通分量的降序排列，C[0]即为极大连通子图
# print(C)
# print(C[0])

# deg=np.array(degree)   #数组形式的度序列
# max_tries = 1000000  #最大尝试次数
# connected=1
# t0=5  #随机实验的次数
# record_ratio1=np.zeros((t0,kmax-kmin+1))
# for i in range(1,t0+1):
#     j=0
#     I = []
#     n = []
#     for k0 in range(kmin,kmax+1):
#         #list_j=[x for x in degree if x>=k0]  #富节点的集合
#         list_j=np.where(deg>=k0)#富节点的集合
#         print(list_j)
#
#
#         ln = len(list_j[0])  # 富节点的个数
#         newA=A[list_j,:]
#         newA=A[:,list_j]
#         n.append(ln)
#         I.append(list_j[0])
#         j+=1

# labels=np.zeros(A.shape[0])
# print(labels)
# labels[[0,2]]=1
# labels_one_hot=np.zeros((A.shape[0],2))
# labels_one_hot[:,0]=1
# labels_one_hot[[0,2],0]=0
# labels_one_hot[[0,2],1]=1
# print(labels_one_hot)
#
# features = torch.nn.Embedding(4,3).weight
# print(features)
#
# preds=torch.Tensor([0,0,1,0,1,1])
# labels=torch.Tensor([1,1,1,0,1,0])
# labels_index_for_1=torch.where(labels==1)[0]
# preds_value_for_1=preds[labels_index_for_1]
# all_num=labels_index_for_1.shape[0]
# recall_num=torch.count_nonzero(preds_value_for_1).item()
# print(recall_num/all_num)
#
# import torch.nn.functional as F
# from scipy import sparse
# np.random.seed(42)
# torch.manual_seed(42)
# def normalize(mx):
#     """Row-normalize sparse matrix"""
#     rowsum = np.array(mx.sum(1))
#     r_inv = np.power(rowsum, -1).flatten()
#     r_inv[np.isinf(r_inv)] = 0.
#     r_mat_inv = sparse.diags(r_inv)
#     mx = r_mat_inv.dot(mx)
#     return mx
#
# features = torch.rand(4,3)
# print(features)
# features1=normalize(features)
# print(torch.tensor(features1))
# features2=F.normalize(features)
# print(features2)


# from utils.mymap import MapDataset
# MapData = MapDataset()
# print(MapData.rid2nids["22352|0"])
a=eval("(1,2,3)")
b=tuple([a])
print(a)
print(b)