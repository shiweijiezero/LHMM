import math
import pickle
import os
import random
import sys

base_path = os.path.abspath("../")
sys.path.append(base_path)

base_path = os.path.abspath("../")
sys.path.append(base_path)
import pandas
import tqdm
from fastparquet import ParquetFile
import numpy
from config import DefaultConfig
from utils.mymap import MapDataset
from utils import mydb

filename = "../data/transTable.parquet"
pf = ParquetFile(filename)
df = pf.to_pandas()
# df = df.head(3)
opt = DefaultConfig()
pos2dgl_cellular_id_round = {}
mongo_db = mydb.MYDB().db
network_e_table = mongo_db[opt.roadnetwork_e_mongo_table]
match_result_table = mongo_db[opt.match_result_table_path]
match_result_lines = match_result_table.find()
dic_cellular_trajectory = {}
mymap = MapDataset()
with open(opt.pos2dgl_cellular_id_path, mode='rb') as f:
    print("Reading pos2dgl_cellular_id file...")
    pos2dgl_cellular_id: dict = pickle.load(f)

    for pos_tuple, id in pos2dgl_cellular_id.items():
        pos_tuple = round(pos_tuple[0], 5), round(pos_tuple[1], 5)
        pos2dgl_cellular_id_round[pos_tuple] = id

dic_result = {}
for line in tqdm.tqdm(match_result_lines):
    traj_id = line["id"]
    gps_matched_routes_str = line["gpsmatch_routes"]
    gps_matched_routes_lst = gps_matched_routes_str.split(",")
    match_result = set()
    src_nodes_end = None
    for i in range(len(gps_matched_routes_lst)):
        src_road = gps_matched_routes_lst[i]
        src_nodes_start, src_nodes_end = mymap.rid2nids[src_road]
        match_result.add(src_nodes_start)
        match_result.add(src_nodes_end)
    dic_result[traj_id] = match_result
    filtered_traj_mee_string = line["filtered_mee"]
    filtered_traj_mee_lst = filtered_traj_mee_string.split("|")[1].split(",")
    filtered_traj_mee_id_lst = []
    for traj in filtered_traj_mee_lst:
        lng_, lat_ = traj.split()
        lng = round(float(lng_), 5)
        lat = round(float(lat_), 5)
        dgl_id = pos2dgl_cellular_id_round[(lng, lat)]
        filtered_traj_mee_id_lst.append(dgl_id)
    dic_cellular_trajectory[traj_id] = filtered_traj_mee_id_lst  # 得到traj_id：cellular_id序列

def worker(id,trans_str):
    # 解析trans字符串
    item = trans_str.split("~")[:-1]  # 舍弃最后一个空的！！！
    positive_nodes = set()
    negative_nodes = set()
    for infor in item:
        pos = infor.split("^")
        if (pos[5] == "inf" or math.isinf(float(pos[5]))):
            # print("case 0")
            continue
        cellular1_x = float(pos[0])
        cellular1_y = float(pos[1])
        cellular2_x = float(pos[2])
        cellular2_y = float(pos[3])

        cellular_dis = float(pos[4])
        path_dis = float(pos[5])
        path = pos[6].split("*")
        if (math.isinf(cellular_dis) or math.isinf(path_dis)):
            # print("case 1")
            continue
        if (path[0] == ""):
            # print("case 2")
            continue

        for path_iter in path:
            nid1, nid2 = mymap.rid2nids[path_iter]
            lon_1, lat_1 = mymap.get_pos_by_node_id(nid1)
            lon_2, lat_2 = mymap.get_pos_by_node_id(nid2)
            dis_1 = mymap.get_point2line(lon_1,lat_1, cellular1_x, cellular1_y, cellular2_x, cellular2_y)
            dis_2 = mymap.get_point2line(lon_2,lat_2, cellular1_x, cellular1_y, cellular2_x, cellular2_y)
            if (nid1 in dic_result[id]):
                positive_nodes.add((nid1, dis_1))
            else:
                negative_nodes.add((nid1, dis_1))
            if (nid2 in dic_result[id]):
                positive_nodes.add((nid2, dis_2))
            else:
                negative_nodes.add((nid2, dis_2))
    train_data=[]
    positive_nodes=list(positive_nodes)
    negative_nodes=list(negative_nodes)
    for i in range(25):
        # print(len(negative_nodes))
        # print(len(positive_nodes))
        if(len(positive_nodes)>=len(negative_nodes)):
            print(f"len(positive_nodes)>len(negative_nodes):{len(positive_nodes)},{len(negative_nodes)}")
            pos_sample=random.sample(positive_nodes, len(negative_nodes))
            train_data.append([pos_sample, negative_nodes])
        else:
            neg_sample = random.sample(negative_nodes, len(positive_nodes))
            train_data.append([positive_nodes, neg_sample])
    return id,train_data


from multiprocessing import Pool
t = tqdm.tqdm(total=len(df))
trans_dic = {}
po = Pool(100)
all_temp_data=[]
for row in df.itertuples():
    t.update(1)
    id = getattr(row, 'id')
    trans_str = getattr(row, 'transTable')
    all_temp_data.append(po.apply_async(worker,(id,trans_str,)))


po.close()    # 关闭进程池，关闭后po不再接受新的请求
po.join()     # 等待po中的所有子进程执行完成，必须放在close语句之后
t.close()
print("-----end-----")
for i in tqdm.tqdm(range(len(all_temp_data))):
    id,train_data=all_temp_data[i].get()
    trans_dic[id]=train_data

trans=list(trans_dic.items())
# print("保存trans_train字典")
# with open(opt.trans_train_path, mode='wb') as f:
#     pickle.dump(trans_dic, f)
trans1=dict(trans[:int(len(trans)*0.1)])
with open(opt.trans_train_path1, mode='wb') as f1:
    print("保存trans_train1字典")
    pickle.dump(trans1, f1)
trans2=dict(trans[int(len(trans)*0.1):int(len(trans)*0.2)])
with open(opt.trans_train_path2, mode='wb') as f1:
    print("保存trans_train2字典")
    pickle.dump(trans2, f1)
trans3=dict(trans[int(len(trans)*0.2):int(len(trans)*0.3)])
with open(opt.trans_train_path3, mode='wb') as f1:
    print("保存trans_train3字典")
    pickle.dump(trans3, f1)
trans4=dict(trans[int(len(trans)*0.3):int(len(trans)*0.4)])
with open(opt.trans_train_path4, mode='wb') as f1:
    print("保存trans_train4字典")
    pickle.dump(trans4, f1)
trans5=dict(trans[int(len(trans)*0.4):int(len(trans)*0.5)])
with open(opt.trans_train_path5, mode='wb') as f1:
    print("保存trans_train5字典")
    pickle.dump(trans5, f1)
trans6=dict(trans[int(len(trans)*0.5):int(len(trans)*0.6)])
with open(opt.trans_train_path6, mode='wb') as f1:
    print("保存trans_train6字典")
    pickle.dump(trans6, f1)
trans7=dict(trans[int(len(trans)*0.6):int(len(trans)*0.7)])
with open(opt.trans_train_path7, mode='wb') as f1:
    print("保存trans_train7字典")
    pickle.dump(trans7, f1)
trans8=dict(trans[int(len(trans)*0.7):int(len(trans)*0.8)])
with open(opt.trans_train_path8, mode='wb') as f1:
    print("保存trans_train8字典")
    pickle.dump(trans8, f1)
trans9=dict(trans[int(len(trans)*0.8):int(len(trans)*0.9)])
with open(opt.trans_train_path9, mode='wb') as f1:
    print("保存trans_train9字典")
    pickle.dump(trans9, f1)
trans10=dict(trans[int(len(trans)*0.9):])
with open(opt.trans_train_path10, mode='wb') as f1:
    print("保存trans_train10字典")
    pickle.dump(trans10, f1)
print("保存dic_cellular_trajectory字典")
with open(opt.dic_cellular_trajectory_path, mode='wb') as f:
    pickle.dump(dic_cellular_trajectory, f)


