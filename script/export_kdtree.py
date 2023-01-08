import networkx
from scipy.spatial.kdtree import KDTree
import os
import pickle
import sys
base_path = os.path.abspath("../")
sys.path.append(base_path)
import numpy as np
import config
from utils.mymap import MapDataset
import pickle

MapData=MapDataset()
s=MapData.nxgraph.nodes
max_id=max(s)
print(max_id)
poss = np.zeros((max_id + 1, 2))
for id in s:
    node_lat = MapData.nxgraph.nodes[id]["lat"]
    node_lng = MapData.nxgraph.nodes[id]["lng"]
    poss[id, :] = node_lng, node_lat
print("保存KDtree对象")
kdtree = KDTree(data=poss)
opt=config.DefaultConfig()
with open(opt.kdtree_path, mode='wb') as f:
    pickle.dump(kdtree, f)
temp=kdtree.data
print("end")