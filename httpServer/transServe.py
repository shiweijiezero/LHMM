import math
import os
import pickle
import sys
import torch

base_path = os.path.abspath("../")
sys.path.append(base_path)

from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.options import options
from tornado.web import Application, RequestHandler

from config import DefaultConfig
import numpy as np
from utils.mymap import MapDataset
from trans.trans_model import TransModel
from trans.trans_utils import load_base_data

opt = DefaultConfig()
mymap = MapDataset()
model = torch.load(DefaultConfig.trans_use_model,map_location=DefaultConfig.DEVICE)  # 导入保存的模型
# model.eval()
with open(opt.pos2dgl_cellular_id_path, mode='rb') as f:
    print("Reading pos2dgl_cellular_id file...")
    pos2dgl_cellular_id: dict = pickle.load(f)
dgl_graph, \
dgl_cellular_id2pos, \
dic_cellular_traj \
    = load_base_data()


class transHandles(RequestHandler):
    def post(self):
        # print("------------------------------------------------")
        # print("enter")
        preSampleX = round(float(self.get_argument("preSampleX")),5)
        preSampleY = round(float(self.get_argument("preSampleY")),5)
        curSampleX = round(float(self.get_argument("curSampleX")),5)
        curSampleY = round(float(self.get_argument("curSampleY")),5)
        # remoteInfo = self.get_argument("remoteInfo")
        traj_id = self.get_argument("traj_id")
        dgl_cellular_id_lst = torch.tensor(dic_cellular_traj[traj_id]).to(DefaultConfig.DEVICE)

        remoteInfoPreId = self.get_argument("remoteInfoPreId")
        remoteInfoCurId = self.get_argument("remoteInfoCurId")
        remoteInfoNodeId = self.get_argument("remoteInfoNodeId")
        remoteInfoP2SDis = self.get_argument("remoteInfoP2SDis")
        pre_id_lst=remoteInfoPreId.split("*")
        cur_id_lst=remoteInfoCurId.split("*")

        node_id_lst_lst=list(map(lambda e:[int(i) for i in e.split(",")],remoteInfoNodeId.split("*")))
        node_dis_lst_lst=list(map(lambda e:[float(i) for i in e.split(",")],remoteInfoP2SDis.split("*")))
        node_pair_lst_lst = list(map(lambda a, b: list(zip(a, b)), node_id_lst_lst, node_dis_lst_lst))

        node_pair_lst =  list(set([x for j in node_pair_lst_lst for x in j]))
        record={value:index for index,value in enumerate(node_pair_lst)}
        all_node_lst=[pair[0] for pair in node_pair_lst]
        all_dis_lst=[pair[1] for pair in node_pair_lst]

        label_nids = torch.tensor(all_node_lst,dtype=torch.long).to(DefaultConfig.DEVICE)
        node_dis = torch.tensor(all_dis_lst).to(DefaultConfig.DEVICE)

        output = model(
            dgl_cellular_id_lst,
            label_nids,
            dgl_graph,
            node_dis
        )
        output = output[:,1].reshape(output.shape[0])  # n

        respose_lst=[]
        counter=0
        for i in range(len(node_id_lst_lst)):
            length=len(node_id_lst_lst[i])
            index_lst=torch.tensor([record[(node_id_lst_lst[i][j],node_dis_lst_lst[i][j])]
                                    for j in range(len(node_id_lst_lst[i]))])
            score=torch.mean(output[index_lst]).item()
            counter+=length
            respose_lst.append(f"{pre_id_lst[i]},{cur_id_lst[i]},{score}")
        # if(counter!=len(all_node_lst)):
        #     raise Exception("數量不等！@")
        post_candidate_str=";".join(respose_lst)
        self.write(post_candidate_str)
        # print(post_candidate_str)
        # print("success")
        # print("------------------------------------------------")


# class transHandles(RequestHandler):
#     def post(self):
#         # print("------------------------------------------------")
#         print("enter")
#         preSampleX = round(float(self.get_argument("preSampleX")),5)
#         preSampleY = round(float(self.get_argument("preSampleY")),5)
#         curSampleX = round(float(self.get_argument("curSampleX")),5)
#         curSampleY = round(float(self.get_argument("curSampleY")),5)
#         remoteInfo = self.get_argument("remoteInfo")
#         traj_id = self.get_argument("traj_id")
#         dgl_cellular_id_lst = torch.tensor(dic_cellular_traj[traj_id]).to(DefaultConfig.DEVICE)
#         # pre_cellular_id = pos2dgl_cellular_id[(preSampleX, preSampleY)]
#         # cur_cellular_id = pos2dgl_cellular_id[(curSampleX, curSampleY)]
#         item = remoteInfo.split("~")[:-1]  # 舍弃最后一个空的！！！
#         data_dic = {}
#         record_dic={}
#         all_path_node = set()
#
#         for infor in item:
#             # 解析
#             pos = infor.split("^")
#             pre_road_id = pos[0]
#             cur_road_id = pos[1]
#             if (pos[3] == "inf" or pos[2]=="inf"):
#                 data_dic[(pre_road_id, cur_road_id)] = 0
#                 continue
#             cellular_dis = round(float(pos[2]), 5)
#             path_dis = round(float(pos[3]), 5)
#             path = pos[4].split("*")
#             if (math.isinf(cellular_dis) or math.isinf(path_dis)):
#                 # print("case 1")
#                 data_dic[(pre_road_id, cur_road_id)] = 0
#                 continue
#             if (path[0] == ""):
#                 # print("case 2")
#                 data_dic[(pre_road_id, cur_road_id)] = 0
#                 continue
#             for path_iter in path:
#                 nid1,nid2 = mymap.rid2nids[path_iter]
#                 lng_1,lat_1=mymap.get_pos_by_node_id(nid1)
#                 lng_2,lat_2=mymap.get_pos_by_node_id(nid2)
#                 dis1=mymap.get_point2line(lng_1,lat_1,preSampleX,preSampleY,curSampleX,curSampleY)
#                 dis2=mymap.get_point2line(lng_2,lat_2,preSampleX,preSampleY,curSampleX,curSampleY)
#                 if(record_dic.get((pre_road_id, cur_road_id))==None):
#                     record_dic[(pre_road_id, cur_road_id)]=set()
#                 record_dic[(pre_road_id, cur_road_id)].add((nid1,dis1))
#                 record_dic[(pre_road_id, cur_road_id)].add((nid2,dis2))
#                 all_path_node.add((nid1,dis1))
#                 all_path_node.add((nid2,dis2))
#         all_path_node_lst=list(all_path_node)
#         node_dic=dict()
#         dgl_cellular_id_lst = torch.tensor(dic_cellular_traj[traj_id]).to(DefaultConfig.DEVICE)
#         nids = [pair[0] for pair in all_path_node_lst]
#         dis = [pair[1] for pair in all_path_node_lst]
#         label_nids = torch.tensor(nids,dtype=torch.long).to(DefaultConfig.DEVICE)
#         node_dis = torch.tensor(dis).to(DefaultConfig.DEVICE)
#
#         output = model(
#             dgl_cellular_id_lst,
#             label_nids,
#             dgl_graph,
#             node_dis
#         )
#         output = output[:,1]  # n*1
#
#         for i in range(output.shape[0]):
#             node_dic[(nids[i],dis[i])] = output[i].item()
#             # print(output[i].item())
#
#         post_candidate_str = ";".join([
#             key[0] + "," +
#             key[1] + "," +
#             str(np.mean([node_dic[item] for item in value]))
#             for key,value in record_dic.items()
#         ])
#         self.write(post_candidate_str)
#         print(post_candidate_str)
#         print("success")
#         # print("------------------------------------------------")

if __name__ == "__main__":
    options.parse_command_line()
    options.define("listen_port", default=12111, help='监听服务的端口号')
    app = Application(
        [
            (r"/trans", transHandles)
        ],
    )
    http_server = HTTPServer(app)
    http_server.listen(options.listen_port)
    http_server.start(1)
    IOLoop.current().start()
