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


def export_gps_and_mee():
    mee_dict={}
    gps_dict={}
    mongo_db = mydb.MYDB().db
    pairedData_table = mongo_db[opt.pairedData_mongo_table]
    paired_lines = pairedData_table.find()

    num_data=pairedData_table.count()
    print(num_data)
    bar = pyprind.ProgBar(iterations=num_data)
    counter=0
    for line in paired_lines:
        bar.update()
        print(counter)
        counter+=1
        traj_id=line["id"]
        traj_gps_string:str=line["gpsString"]
        traj_gps_lst=traj_gps_string.split(";")
        traj_result=[]
        for traj in traj_gps_lst:
            lng,lat,timestamp=traj.split()
            traj_result.append((float(lng),float(lat),int(timestamp)))
        gps_dict[traj_id]=traj_result

        traj_mee_string=line["meeString"]
        traj_mee_lst=traj_mee_string.split(";")
        traj_result=[]
        for traj in traj_mee_lst:
            lng,lat,timestamp=traj.split()
            traj_result.append((float(lng),float(lat),int(timestamp)))
        mee_dict[traj_id]=traj_result

    print("保存gps_traj字典")
    with open(opt.gps_traj_path, mode='wb') as f:
        pickle.dump(gps_dict, f)
    print("保存mee_traj字典")
    with open(opt.mee_traj_path, mode='wb') as f:
        pickle.dump(mee_dict, f)

if __name__ == "__main__":
    export_gps_and_mee()