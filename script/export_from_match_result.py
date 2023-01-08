import os
import pickle
import sys
base_path = os.path.abspath("../")
sys.path.append(base_path)
from config import DefaultConfig
from utils import mydb
import networkx
import pyprind
import pickle
from scipy.spatial.kdtree import KDTree
import numpy as np

opt = DefaultConfig()


def export_from_match_result():
    match_result_dict={}
    filtered_mee_dict={}
    filtered_gps_dict={}
    mongo_db = mydb.MYDB().db
    match_result_table = mongo_db[opt.match_result_table_path]
    match_result_lines = match_result_table.find()

    num_data=match_result_table.count()
    print(num_data)
    bar = pyprind.ProgBar(iterations=num_data)
    counter=0
    for line in match_result_lines:
        bar.update()
        print(counter)
        counter+=1
        traj_id=line["id"]
        filtered_traj_gps_string:str=line["filtered_gps"]
        filtered_traj_gps_lst=filtered_traj_gps_string.split("|")[1].split(",")
        traj_result=[]
        for traj in filtered_traj_gps_lst:
            lng,lat=traj.split()
            traj_result.append((float(lng),float(lat),0)) # 第三个参数代表时间戳，后续如果不用这里设为0
        filtered_gps_dict[traj_id]=traj_result

        filtered_traj_mee_string=line["filtered_mee"]
        filtered_traj_mee_lst=filtered_traj_mee_string.split("|")[1].split(",")
        traj_result=[]
        for traj in filtered_traj_mee_lst:
            lng,lat=traj.split()
            traj_result.append((float(lng),float(lat),0))
        filtered_mee_dict[traj_id]=traj_result

        match_result_string=line["gpsmatch_routes"]
        match_result_lst=match_result_string.split(",")
        match_result_dict[traj_id]=match_result_lst

    print("保存filtered_gps_traj字典")
    with open(opt.filtered_gps_traj_path, mode='wb') as f:
        pickle.dump(filtered_gps_dict, f)
    print("保存filtered_mee_traj字典")
    with open(opt.filtered_mee_traj_path, mode='wb') as f:
        pickle.dump(filtered_mee_dict, f)


if __name__ == "__main__":
    export_from_match_result()