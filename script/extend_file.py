import pickle
import os
import sys

base_path = os.path.abspath("../")
sys.path.append(base_path)
from config import DefaultConfig
opt = DefaultConfig()

with open(opt.trans_train_path, mode='rb') as f:
    print("Reading trans_train_dic file...")
    trans_train_dic = pickle.load(f)
    trans=list(trans_train_dic.items())

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


    # for i in range(len(trans)):
    #     key, value = trans[i]
    #     traj_id,pre_cellular_id,cur_cellular_id=key
    #     print(f"traj_id:{traj_id}")
    #     print(f"pre_cellular_id:{pre_cellular_id}")
    #     print(f"cur_cellular_id:{cur_cellular_id}")
    #     for i in range(1,len(value)): # 因为0是ground-truth path node序列
    #         item=value[i]
    #         print(f"item[0]:{item[0]}")
    #         print(f"item[1]:{item[1]}")
    #         print(f"item[3]:{item[3]}")

