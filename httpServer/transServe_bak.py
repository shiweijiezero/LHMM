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
from trans_bak.trans_model import TransModel
from trans_bak.trans_utils import load_base_data

opt = DefaultConfig()
mymap = MapDataset()
model = torch.load(DefaultConfig.trans_use_model).to(DefaultConfig.DEVICE)  # 导入保存的模型
model.eval()
with open(opt.pos2dgl_cellular_id_path, mode='rb') as f:
    print("Reading pos2dgl_cellular_id file...")
    pos2dgl_cellular_id: dict = pickle.load(f)
road_features, \
cellular_features, \
dgl_graph, \
dgl_cellular_id2pos, \
dic_cellular_traj \
    = load_base_data()


class transHandles(RequestHandler):
    def post(self):
        # print("------------------------------------------------")
        preSampleX = round(float(self.get_argument("preSampleX")),5)
        preSampleY = round(float(self.get_argument("preSampleY")),5)
        curSampleX = round(float(self.get_argument("curSampleX")),5)
        curSampleY = round(float(self.get_argument("curSampleY")),5)
        remoteInfo = self.get_argument("remoteInfo")
        traj_id = self.get_argument("traj_id")
        timeDiff = float(self.get_argument("timeDiff"))

        pre_cellular_id = pos2dgl_cellular_id[(preSampleX, preSampleY)]
        cur_cellular_id = pos2dgl_cellular_id[(curSampleX, curSampleY)]
        item = remoteInfo.split("~")[:-1]  # 舍弃最后一个空的！！！
        data_dic = {}
        all_batch_cellular_dis = []
        all_batch_path_dis = []
        all_batch_path_node = []
        all_batch_name_pair = []
        for infor in item:
            # 解析
            pos = infor.split("^")
            pre_road_id = pos[0]
            cur_road_id = pos[1]
            if (pos[3] == "inf" or pos[2]=="inf"):
                data_dic[(pre_road_id, cur_road_id)] = 0
                continue
            cellular_dis = round(float(pos[2]), 5)
            path_dis = round(float(pos[3]), 5)
            path = pos[4].split("*")
            if (math.isinf(cellular_dis) or math.isinf(path_dis)):
                data_dic[(pre_road_id, cur_road_id)] = 0
                continue
            if (len(path) <= 2):
                beta = 0.008
                normalizedTransitionMetric=(abs(cellular_dis - path_dis))/(timeDiff*timeDiff)
                # print(cellular_dis,path_dis,timeDiff)
                # print(f"normalizedTransitionMetric为：{normalizedTransitionMetric}")
                # print(f"设为：{1 / beta * math.exp(-normalizedTransitionMetric / beta)}")
                data_dic[(pre_road_id, cur_road_id)] = max(1 / beta * math.exp(-normalizedTransitionMetric / beta),0.9)
                continue
            path_node = []
            nids = None
            for path_iter in path:
                nids = mymap.rid2nids[path_iter]
                path_node.append(nids[0])
            if (nids != None):
                path_node.append(nids[1])

            all_batch_name_pair.append((pre_road_id, cur_road_id))
            all_batch_cellular_dis.append(cellular_dis)
            all_batch_path_dis.append(path_dis)
            all_batch_path_node.append(path_node)

        for i in range(0, len(all_batch_cellular_dis), DefaultConfig.trans_mini_batch):  # 训练mini batch最大为128
            batch_cellular_dis = all_batch_cellular_dis[i:i + DefaultConfig.trans_mini_batch]
            batch_path_dis = all_batch_path_dis[i:i + DefaultConfig.trans_mini_batch]
            batch_path_node = all_batch_path_node[i:i + DefaultConfig.trans_mini_batch]
            batch_name_pair = all_batch_name_pair[i:i + DefaultConfig.trans_mini_batch]

            batch_cellular_dis = torch.tensor(batch_cellular_dis).reshape((len(batch_cellular_dis), 1))
            batch_path_dis = torch.tensor(batch_path_dis).reshape((len(batch_path_dis), 1))
            output = model(
                road_features,
                cellular_features,
                torch.tensor(dic_cellular_traj[traj_id]),
                pre_cellular_id,
                cur_cellular_id,
                batch_path_node,
                dgl_graph,
                batch_cellular_dis,
                batch_path_dis
            )
            output = output.reshape(output.shape[0])  # n*1
            for i in range(len(batch_name_pair)):
                data_dic[batch_name_pair[i]] = output[i].item()
                print(output[i].item())
        post_candidate_str = ";".join([
            key[0] + "," +
            key[1] + "," +
            str(value)
            for key, value in data_dic.items()
        ])
        self.write(post_candidate_str)
        # print("success")
        # print("------------------------------------------------")

if __name__ == "__main__":
    options.define("listen_port", default=10086, help='监听服务的端口号')
    app = Application(
        [
            (r"/trans", transHandles)
        ],
    )
    http_server = HTTPServer(app)
    http_server.listen(options.listen_port)
    http_server.start(1)
    IOLoop.current().start()
