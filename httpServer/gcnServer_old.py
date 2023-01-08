import math
import os
import sys


base_path = os.path.abspath("../")
sys.path.append(base_path)

from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.options import options
from tornado.web import Application, RequestHandler

from utils.mymap import MapDataset
from utils.mygcn_graph import GCN_graph

base_path = os.path.abspath("../")
sys.path.append(base_path)
from pygcn.gcn_eval import *
import numpy as np

model_obj = gcn_eval()
gcn_graph = GCN_graph()
MapData = MapDataset()


class gcnHandles(RequestHandler):
    def post(self):
        # print("------------------------------------------------")
        # print("start_recive")
        sample_x = float(self.get_argument("getSampleX"))
        sample_y = float(self.get_argument("getSampleY"))
        candidateNUM = int(self.get_argument("currentCandidateNum"))
        # term_x = float(self.get_argument("endSampleX"))
        # term_y = float(self.get_argument("endSampleY"))
        meePositionList_str = self.get_argument("allSample")[11:]  # 11 代表跳过arrayBuffer
        # print(meePositionList_str)
        meePositionList = list(eval(meePositionList_str))
        # print(meePositionList)
        # print("ok")

        # 先大范围取就近的，然后再取TopNumber的候选节点
        candidate_nids = list(MapData.kdtree.query(np.array([sample_x, sample_y]), k=int(800))[1])
        # print(f"candidate_nids:{candidate_nids}")
        # print(f"{MapData.nid2rids[candidate_nids[0]]}")

        # 将这些节点换成路段，路段分数为两节点的均值
        candidate_roads = set()
        for rids_set in [MapData.nid2rids[nid] for nid in candidate_nids]:
            candidate_roads = candidate_roads.union(rids_set)
        candidate_roads = list(candidate_roads)  # 获得所有的候选路段
        candidate_nids_tuple_lst = [MapData.rid2nids[rid] for rid in candidate_roads]  # [(src,dst),(src,dst)]
        # print(f"候选路段节点数量与路段数量：{candidateNUM},{len(candidate_nids)},{len(candidate_roads)},{len(candidate_nids_tuple_lst)}")
        # print(f"candidate_nids_tuple_lst:{candidate_nids_tuple_lst}")

        # 运行模型，获得模型得到各个路段的概率值
        candidate_nids_set = set()
        for nids in candidate_nids_tuple_lst:
            candidate_nids_set.add(nids[0])
            candidate_nids_set.add(nids[1])
        candidate_nids_lst = list(candidate_nids_set)
        candidate_nids2idx_dict = {nid: idx for idx, nid in enumerate(candidate_nids_lst)}

        output = model_obj.eval(meePositionList, candidate_nids_lst).detach().cpu().numpy()
        output = output[:, 1]  # 取出正例的output值
        # print(f"output:{output}")

        # 根据nid转为index得到路段两个model score取均值
        candidate_roads_model_score = [
            (output[candidate_nids2idx_dict[nids[0]]] + output[candidate_nids2idx_dict[nids[1]]]) / 2
            for nids in candidate_nids_tuple_lst
        ]

        candidate_roads_distance = [
            max(
                distance(sample_x, sample_y,
                         MapData.get_pos_by_node_id(nids[0])[0],
                         MapData.get_pos_by_node_id(nids[0])[1],
                         MapData.get_pos_by_node_id(nids[1])[0],
                         MapData.get_pos_by_node_id(nids[1])[1])
                , 1)  # 加max避免除以0
            for nids in candidate_nids_tuple_lst
        ]
        sigma = 600
        candidate_roads_distance_score = [
            1.0 / (math.sqrt(2.0 * math.pi) * sigma) * math.e ** (-0.5 * math.pow(dis_ / sigma, 2))
            for dis_ in candidate_roads_distance
        ]
        # print(f"原始candidate_roads_model_score[0]:{candidate_roads_model_score[0]}")
        # print(f"原始candidate_roads_model_score_range:{max(candidate_roads_model_score),min(candidate_roads_model_score)}")
        # print(f"原始candidate_roads_distance_score[0]:{candidate_roads_distance_score[0]}")
        # print(f"原始candidate_roads_distance_score_range:{max(candidate_roads_distance_score),min(candidate_roads_distance_score)}")
        # 将model score与distance score进行标准归一化
        # candidate_roads_model_score = standardization(candidate_roads_model_score)
        # candidate_roads_distance_score = standardization(candidate_roads_distance_score)
        # print(f"标准化candidate_roads_model_score[0]:{candidate_roads_model_score[0]}")
        # print(f"标准化candidate_roads_model_score_range:{max(candidate_roads_model_score),min(candidate_roads_model_score)}")
        # print(f"标准化candidate_roads_distance_score[0]:{candidate_roads_distance_score[0]}")
        # print(f"标准化candidate_roads_distance_score_range:{max(candidate_roads_distance_score),min(candidate_roads_distance_score)}")

        # candidate_roads_model_score = normalization(candidate_roads_model_score)
        # candidate_roads_distance_score = normalization(candidate_roads_distance_score)
        # print(f"归一化candidate_roads_model_score[0]:{candidate_roads_model_score[0]}")
        # print(f"归一化candidate_roads_model_score_range:{max(candidate_roads_model_score),min(candidate_roads_model_score)}")
        # print(f"归一化candidate_roads_distance_score[0]:{candidate_roads_distance_score[0]}")
        # print(f"归一化candidate_roads_distance_score_range:{max(candidate_roads_distance_score),min(candidate_roads_distance_score)}")

        # candidate_roads_model_score = scaling(candidate_roads_model_score,min(candidate_roads_distance_score),max(candidate_roads_distance_score))
        # candidate_roads_distance_score = candidate_roads_distance_score
        # print(f"放缩candidate_roads_model_score[0]:{candidate_roads_model_score[0]}")
        # print(f"放缩candidate_roads_model_score_range:{max(candidate_roads_model_score),min(candidate_roads_model_score)}")
        # print(f"放缩candidate_roads_distance_score[0]:{candidate_roads_distance_score[0]}")
        # print(f"放缩candidate_roads_distance_score_range:{max(candidate_roads_distance_score),min(candidate_roads_distance_score)}")


        # 得到发射概率
        # if (
        #         len(candidate_roads_model_score) != len(candidate_roads_distance_score)
        #         or
        #         len(candidate_roads_model_score) != len(candidate_nids_tuple_lst)
        # ):
        #     print("candidate数量不一致！")

        # 调整模型评分与距离评分各自占比
        model_factor = 1
        distance_factor = 1
        candidate_emit_score = [
            (
                # 路名
                MapData.nxgraph[candidate_nids_tuple_lst[i][0]][candidate_nids_tuple_lst[i][1]]["relationship"],
                # 首尾的经纬坐标
                MapData.get_pos_by_node_id(candidate_nids_tuple_lst[i][0])[0],
                MapData.get_pos_by_node_id(candidate_nids_tuple_lst[i][0])[1],
                MapData.get_pos_by_node_id(candidate_nids_tuple_lst[i][1])[0],
                MapData.get_pos_by_node_id(candidate_nids_tuple_lst[i][1])[1],
                # 发射概率项，保证非负性
                # max(
                #     0,
                #     model_factor * candidate_roads_model_score[i]
                #     + distance_factor * candidate_roads_distance_score[i]
                # )
                max(
                    0,
                    model_factor * candidate_roads_model_score[i]
                    * distance_factor * candidate_roads_distance_score[i]
                )
            )
            for i in range(len(candidate_roads_model_score))
        ]
        # 排序并获得前num个候选路段
        result = sorted(candidate_emit_score, key=lambda x: x[5], reverse=True)[:candidateNUM]
        # print(result)
        # print(f"result:{result}")
        # print("数量为:"+str(len(result)))
        # 构建返回字符串
        post_candidate_str = ";".join([
            str(value[0]) + "," +
            str(value[1]) + "," +
            str(value[2]) + "," +
            str(value[3]) + "," +
            str(value[4]) + "," +
            str(value[5])
            for value in result
        ])
        # print(f"post_candidate_str:{post_candidate_str}")
        # print("success")
        # print("------------------------------------------------")
        self.write(post_candidate_str)


def scaling(data,lower_range,upper_range):
    data = np.array(data)
    temp = ((data - np.min(data))*((upper_range-lower_range)/(np.max(data) - np.min(data))))+lower_range
    return temp.tolist()

def distance(x, y, x1, y1, x2, y2):  # 点到路段
    di = get_distance(x1, y1, x2, y2)
    if (get_distance(x, y, x1, y1) >= get_distance(x, y, x2, y2)):
        duan = get_distance(x, y, x2, y2)
        chang = get_distance(x, y, x1, y1)
    else:
        duan = get_distance(x, y, x1, y1)
        chang = get_distance(x, y, x2, y2)
    if (duan ** 2 + di ** 2 <= chang ** 2):
        return duan
    else:
        p = (di + duan + chang) / 2
        square = math.sqrt(p * (p - di) * (p - duan) * (p - chang))
        gao = 2 * square / di
        return gao


def get_distance(lng1, lat1, lng2, lat2):  # 两点间
    lng1 = float(lng1)
    lat1 = float(lat1)
    lng2 = float(lng2)
    lat2 = float(lat2)
    radLat1 = lat1 * math.pi / 180.0
    radLat2 = lat2 * math.pi / 180.0
    a = radLat1 - radLat2
    b = lng1 * math.pi / 180.0 - lng2 * math.pi / 180.0
    s = 2 * math.asin(
        math.sqrt(math.pow(math.sin(a / 2), 2) + math.cos(radLat1) * math.cos(radLat2) * math.pow(math.sin(b / 2), 2)))
    s = s * 6378137
    return s


def softmax(x):
    max = np.max(x, axis=1, keepdims=True)  # returns max of each row and keeps same dims
    e_x = np.exp(x - max)  # subtracts each row with its max value
    sum = np.sum(e_x, axis=1, keepdims=True)  # returns sum of each row and keeps same dims
    f_x = e_x / sum
    return f_x


def normalization(data):
    data = np.array(data)
    _range = np.max(data) - np.min(data)
    temp = (data - np.min(data)) / _range
    return temp.tolist()


def standardization(data):
    data = np.array(data)
    mu = np.mean(data)
    sigma = np.std(data)
    temp = (data - mu) / sigma
    return temp.tolist()


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')
    options.parse_command_line()
    options.define("listen_port", default=9292, help='监听服务的端口号')
    app = Application(
        [
            (r"/gcn", gcnHandles)
        ],
        # debug=True   #调试模式
    )

    http_server = HTTPServer(app)
    # http_server.start(8)
    http_server.listen(options.listen_port)
    # 开启的并发进程数量
    http_server.start(1)  # 8
    IOLoop.current().start()
    # print(standardization([1,2,3]))
