import pickle
import os
import sys

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
dic_result = {}
dic_cellular_trajectory={}
mymap = MapDataset()
with open(opt.pos2dgl_cellular_id_path, mode='rb') as f:
    print("Reading pos2dgl_cellular_id file...")
    pos2dgl_cellular_id: dict = pickle.load(f)

    for pos_tuple, id in pos2dgl_cellular_id.items():
        pos_tuple = round(pos_tuple[0], 5), round(pos_tuple[1], 5)
        pos2dgl_cellular_id_round[pos_tuple] = id

for line in tqdm.tqdm(match_result_lines):
    traj_id = line["id"]
    gps_matched_routes_str = line["gpsmatch_routes"]
    gps_matched_routes_lst = gps_matched_routes_str.split(",")
    match_result = []
    src_nodes_end = None
    for i in range(len(gps_matched_routes_lst)):
        src_road = gps_matched_routes_lst[i]
        src_nodes_start, src_nodes_end = mymap.rid2nids[src_road]
        match_result.append(src_nodes_start)
    if (src_nodes_end != None):
        match_result.append(src_nodes_end)
    dic_result[traj_id] = tuple(match_result)
    filtered_traj_mee_string = line["filtered_mee"]
    filtered_traj_mee_lst = filtered_traj_mee_string.split("|")[1].split(",")
    filtered_traj_mee_id_lst=[]
    for traj in filtered_traj_mee_lst:
        lng_, lat_ = traj.split()
        lng = round(float(lng_), 5)
        lat = round(float(lat_), 5)
        dgl_id = pos2dgl_cellular_id_round[(lng,lat)]
        filtered_traj_mee_id_lst.append(dgl_id)
    dic_cellular_trajectory[traj_id]=filtered_traj_mee_id_lst # 得到traj_id：cellular_id序列



t = tqdm.tqdm(total=len(df))
trans_dic = {}
for row in df.itertuples():
    t.update(1)
    id = getattr(row, 'id')
    trans_str = getattr(row, 'transTable')

    # 解析trans字符串
    item = trans_str.split("~")[:-1] # 舍弃最后一个空的！！！
    train_data_lst = []
    for infor in item:
        pos = infor.split("^")
        if (pos[5] == "inf"):
            print("inf-continue")
            continue
        cellular1_x = round(float(pos[0]), 5)
        cellular1_y = round(float(pos[1]), 5)
        cellular2_x = round(float(pos[2]), 5)
        cellular2_y = round(float(pos[3]), 5)
        cellular_dis = float(pos[4])
        path_dis = float(pos[5])
        path = pos[6].split("*")
        if (len(path) <= 2):
            continue
        path_node = []
        nids = None
        for path_iter in path:
            nids = mymap.rid2nids[path_iter]
            path_node.append(nids[0])
        if (nids != None):
            path_node.append(nids[1])
        pre_cellular_id = pos2dgl_cellular_id_round[(cellular1_x, cellular1_y)]
        cur_cellular_id = pos2dgl_cellular_id_round[(cellular2_x, cellular2_y)]

        if (trans_dic.get((id, pre_cellular_id, cur_cellular_id)) == None):
            # 第一个元素放截取的ground-truth
            ground_path_node = dic_result[id]

            cellular1_distance_lst = [mymap.get_distance((cellular1_y, cellular1_x), (mymap.get_pos_by_node_id(node)[1], mymap.get_pos_by_node_id(node)[0])) for node in
                                      ground_path_node]
            cellular2_distance_lst = [mymap.get_distance((cellular2_y, cellular2_x), (mymap.get_pos_by_node_id(node)[1], mymap.get_pos_by_node_id(node)[0])) for node in
                                      ground_path_node]
            index_1=numpy.argmin(cellular1_distance_lst)
            index_2=numpy.argmin(cellular2_distance_lst)
            if(index_2<=index_1):
                continue
            truncated_ground_path_node=tuple(ground_path_node[index_1:index_2+1])
            trans_dic[(id, pre_cellular_id, cur_cellular_id)] = [truncated_ground_path_node]
        set_ground=set(trans_dic[(id, pre_cellular_id, cur_cellular_id)][0])
        set_path=set(path_node)
        match_num=len(set_path&set_ground)
        all_num=len(trans_dic[(id, pre_cellular_id, cur_cellular_id)][0])+len(path_node)
        trans_dic[(id, pre_cellular_id, cur_cellular_id)].append((cellular_dis, path_dis, tuple(path_node),match_num/all_num))

t.close()

print("保存trans_train字典")
with open(opt.trans_train_path, mode='wb') as f:
    pickle.dump(trans_dic, f)

print("保存dic_cellular_trajectory字典")
with open(opt.dic_cellular_trajectory_path, mode='wb') as f:
    pickle.dump(dic_cellular_trajectory, f)
