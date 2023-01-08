import os
import sys
import copy

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch

base_path = os.path.abspath("../")
sys.path.append(base_path)

from utils.mymap import MapDataset

base_path = os.path.abspath("../")
sys.path.append(base_path)
from trans_bak.trans_utils import *
from trans_bak.trans_model import *
import time
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import os
from tqdm import tqdm
import numpy as np


class trans_train:
    def __init__(self):
        random.seed(DefaultConfig.seed)
        np.random.seed(DefaultConfig.seed)
        torch.manual_seed(DefaultConfig.seed)
        if (DefaultConfig.DEVICE != "cpu"):
            torch.cuda.manual_seed(DefaultConfig.seed)
        self.time_name = "trans-{}-{}-{}-{}-{}-{}".format(
            time.strftime("%Y-%m-%d-%H-%M"),
            DefaultConfig.gcn_nfeat,
            DefaultConfig.hidden,
            DefaultConfig.lr,
            DefaultConfig.dropout,
            DefaultConfig.train_set_rate
        )
        self.writer = SummaryWriter('../data/log/{}'.format(self.time_name))
        # Load data
        self.mymap = MapDataset()
        self.dgl_graph, \
        self.dgl_cellular_id2pos, \
        self.dic_cellular_traj \
            = load_base_data()

        print("数据加载完成！")
        # Model and optimizer
        if (DefaultConfig.trans_use_model != ""):
            self.model = torch.load(DefaultConfig.trans_use_model)  # 导入保存的模型
        else:
            self.model = TransModel(
                nhid=DefaultConfig.hidden,
                dropout=DefaultConfig.dropout,
                rel_names=self.dgl_graph.etypes,
                road_num_nodes=self.dgl_graph.num_nodes('road'),
                cellular_num_nodes=self.dgl_graph.num_nodes('cellular')
            )
        self.model = self.model.to(DefaultConfig.DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=DefaultConfig.lr, weight_decay=DefaultConfig.weight_decay)
        self.loss = nn.CrossEntropyLoss(torch.tensor(
            [DefaultConfig.trans_negetive_weight,DefaultConfig.trans_positive_weight])).to(DefaultConfig.DEVICE)  # 交叉熵损失
        # self.loss = nn.MSELoss().to(DefaultConfig.DEVICE)  # 均方误差损失
        self.train_counter = 0
        self.val_counter = 0

    def train(self):
        # summary(self.model)
        for epoch in range(DefaultConfig.trans_EPOCHS):
            for flag in range(3, 11):
                self.train_data, self.val_data = load_data(flag)
                self.train_epoch(epoch, flag)
                if (
                not os.path.exists("/home/weijie0311/Project/mapmatching-cikm/data/model/{}".format(self.time_name))):
                    os.makedirs("/home/weijie0311/Project/mapmatching-cikm/data/model/{}".format(self.time_name))
                torch.save(self.model,
                           "/home/weijie0311/Project/mapmatching-cikm/data/model/{}/mlpl-{}.pkl".format(
                               self.time_name, flag))  # 保存整个模型
                print(f"保存模型至" + "/home/weijie0311/Project/mapmatching-cikm/data/model/{}/mlpl-{}.pkl".format(
                    self.time_name, flag))

        print("Optimization Finished!")
        self.writer.close()

    def train_epoch(self, epoch, flag):
        print(f"flag:{flag}")
        t = time.time()
        self.model.train()  # 设置为训练模式
        for i in tqdm(range(len(self.train_data))):
            key, value = self.train_data[i]
            traj_id, pre_cellular_id, cur_cellular_id = key
            all_batch_cellular_dis = []
            all_batch_path_dis = []
            all_batch_path_node = []
            all_batch_label = []
            for i in range(1, len(value)):  # 因为0是ground-truth path node序列
                item = value[i]
                all_batch_cellular_dis.append(item[0])
                all_batch_path_dis.append(item[1])
                all_batch_path_node.append(list(item[2]))
                all_batch_label.append(item[3])

            all_batch_label = self.scaling(all_batch_label)  # 在一对cellular之间的转移放缩到0.1~1之间
            for i in range(0, len(all_batch_cellular_dis), DefaultConfig.trans_mini_batch):  # 训练mini batch最大为128
                batch_cellular_dis = all_batch_cellular_dis[i:i + DefaultConfig.trans_mini_batch]
                batch_path_dis = all_batch_path_dis[i:i + DefaultConfig.trans_mini_batch]
                batch_path_node = all_batch_path_node[i:i + DefaultConfig.trans_mini_batch]
                batch_label = all_batch_label[i:i + DefaultConfig.trans_mini_batch]

                batch_cellular_dis = torch.tensor(batch_cellular_dis).reshape((len(batch_cellular_dis), 1))
                batch_path_dis = torch.tensor(batch_path_dis).reshape((len(batch_path_dis), 1))
                batch_label = torch.tensor(batch_label).reshape((len(batch_label), 1)).to(DefaultConfig.DEVICE)
                neg_batch_label = torch.ones_like(batch_label).to(DefaultConfig.DEVICE)
                neg_batch_label = neg_batch_label - batch_label
                label = torch.cat((neg_batch_label, batch_label), dim=1)

                output = self.model(
                    torch.tensor(self.dic_cellular_traj[traj_id]),
                    pre_cellular_id,
                    cur_cellular_id,
                    batch_path_node,
                    self.dgl_graph,
                    batch_cellular_dis,
                    batch_path_dis
                )
                # 得到损失值，精度，反向传播，并对参数更新
                loss_train = self.loss(output, label)
                acc_train = torch.mean(torch.abs(output[:, 1] - label[:, 1]).reshape(output.shape[0])).item()

                # temp = torch.randint(output.shape[0], (5,))
                # print(f"output:{output[temp]}")
                # print(f"label:{label[temp]}")
                # print(f"acc:{acc_train},loss:{loss_train.item()}")

                self.optimizer.zero_grad()  # 计算梯度重置
                loss_train.backward()
                self.optimizer.step()

                self.writer.add_scalar("loss_train", loss_train, self.train_counter)
                self.writer.add_scalar("acc_train", acc_train, self.train_counter)
                self.train_counter += 1

        acc_lst = []
        self.model.eval()  # 设置为eval模式
        for i in range(len(self.val_data)):
            key, value = self.val_data[i]
            traj_id, pre_cellular_id, cur_cellular_id = key
            all_batch_cellular_dis = []
            all_batch_path_dis = []
            all_batch_path_node = []
            all_batch_label = []
            for i in range(1, len(value)):  # 因为0是ground-truth path node序列
                item = value[i]
                all_batch_cellular_dis.append(item[0])
                all_batch_path_dis.append(item[1])
                all_batch_path_node.append(list(item[2]))
                all_batch_label.append(item[3])

            all_batch_label = self.scaling(all_batch_label)  # 在一对cellular之间的转移放缩到0.1~1之间
            for i in range(0, len(all_batch_cellular_dis), DefaultConfig.trans_mini_batch):  # 训练mini batch最大为128
                batch_cellular_dis = all_batch_cellular_dis[i:i + DefaultConfig.trans_mini_batch]
                batch_path_dis = all_batch_path_dis[i:i + DefaultConfig.trans_mini_batch]
                batch_path_node = all_batch_path_node[i:i + DefaultConfig.trans_mini_batch]
                batch_label = all_batch_label[i:i + DefaultConfig.trans_mini_batch]

                batch_cellular_dis = torch.tensor(batch_cellular_dis).reshape((len(batch_cellular_dis), 1))
                batch_path_dis = torch.tensor(batch_path_dis).reshape((len(batch_path_dis), 1))
                batch_label = torch.tensor(batch_label).reshape((len(batch_label), 1)).to(DefaultConfig.DEVICE)
                neg_batch_label = torch.ones_like(batch_label).to(DefaultConfig.DEVICE)
                neg_batch_label = neg_batch_label - batch_label
                label = torch.cat((neg_batch_label, batch_label), dim=1)

                output = self.model(
                    torch.tensor(self.dic_cellular_traj[traj_id]),
                    pre_cellular_id,
                    cur_cellular_id,
                    batch_path_node,
                    self.dgl_graph,
                    batch_cellular_dis,
                    batch_path_dis
                )
                # 得到损失值，精度
                acc_val = torch.mean(torch.abs(output[:, 1] - label[:, 1]).reshape(output.shape[0])).item()
                acc_lst.append(acc_val)
                self.writer.add_scalar("acc_val", acc_val, self.val_counter)
                self.val_counter += 1
        acc_val = sum(acc_lst) / len(acc_lst)
        print('Epoch: {:0d}'.format(epoch),
              'acc_val: {:.15f}'.format(acc_val),
              'time: {:.15f}s'.format(time.time() - t))

    def scaling(self, data, lower_range=0.05, upper_range=1):
        data = np.array(data)
        data += np.random.random(data.shape) * lower_range
        if (np.max(data) - np.min(data) == 0):
            data = np.random.random(data.shape) * lower_range
        temp = ((data - np.min(data)) * ((upper_range - lower_range) / (np.max(data) - np.min(data)))) + lower_range
        return temp.tolist()


if __name__ == "__main__":
    train_obj = trans_train()
    # print(DefaultConfig.DEVICE)
    train_obj.train()
