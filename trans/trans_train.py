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
from trans.trans_utils import *
from trans.trans_model import *
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
            self.model = torch.load(DefaultConfig.trans_use_model,map_location=DefaultConfig.DEVICE)  # 导入保存的模型
        else:
            self.model = TransModel(
                nfeat=DefaultConfig.gcn_nfeat,
                nhid=DefaultConfig.hidden,
                nclass=DefaultConfig.nclass,
                dropout=DefaultConfig.dropout,
                rel_names=self.dgl_graph.etypes,
                road_num_nodes=self.dgl_graph.num_nodes('road'),
                cellular_num_nodes=self.dgl_graph.num_nodes('cellular')
            )

        self.model = self.model.to(DefaultConfig.DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=DefaultConfig.lr, weight_decay=DefaultConfig.weight_decay)
        self.loss = nn.CrossEntropyLoss(label_smoothing=0.1).to(DefaultConfig.DEVICE)  # 交叉熵损失
        self.train_counter = 0
        self.val_counter = 0

    def train(self):
        # summary(self.model)
        for epoch in range(DefaultConfig.trans_EPOCHS):
            for flag in range(1, 11):
                self.train_data, self.val_data = load_data(flag)
                #
                self.train_data = self.val_data
                #
                self.train_epoch(epoch, flag)
                if (
                        not os.path.exists(
                            "/home/weijie0311/Project/mapmatching-cikm/data/model/{}".format(self.time_name))):
                    os.makedirs("/home/weijie0311/Project/mapmatching-cikm/data/model/{}".format(self.time_name))
                torch.save(self.model,
                           "/home/weijie0311/Project/mapmatching-cikm/data/model/{}/mlpl-{}.pkl".format(
                               self.time_name, epoch*10+flag))  # 保存整个模型
                print(f"保存模型至" + "/home/weijie0311/Project/mapmatching-cikm/data/model/{}/mlpl-{}.pkl".format(
                    self.time_name, epoch*10+flag))

        print("Optimization Finished!")
        self.writer.close()

    def train_epoch(self, epoch, flag):
        print(f"flag:{flag}")
        t = time.time()
        self.model.train()  # 设置为训练模式
        acc_lst = []
        recall_lst = []
        for i in tqdm(range(len(self.train_data))):
            traj_id, value = self.train_data[i]
            dgl_cellular_id_lst = torch.tensor(self.dic_cellular_traj[traj_id]).to(DefaultConfig.DEVICE)
            for train_data in value:
                positive_nids = [pair[0] for pair in train_data[0]]
                positive_dis = [pair[1] for pair in train_data[0]]
                negative_nids = [pair[0] for pair in train_data[1]]
                negative_dis = [pair[1] for pair in train_data[1]]

                label_nids = torch.zeros(len(positive_nids) + len(negative_nids),dtype=torch.long)
                label_nids[:len(positive_nids)] = torch.tensor(positive_nids,dtype=torch.long)
                label_nids[len(positive_nids):] = torch.tensor(negative_nids,dtype=torch.long)
                label_nids = label_nids.to(DefaultConfig.DEVICE)

                node_dis = torch.zeros(len(positive_dis) + len(negative_dis))
                node_dis[:len(positive_dis)] = torch.tensor(positive_dis)
                node_dis[len(positive_dis):] = torch.tensor(negative_dis)
                node_dis = node_dis.to(DefaultConfig.DEVICE)

                output = self.model(
                    dgl_cellular_id_lst,
                    label_nids,
                    self.dgl_graph,
                    node_dis
                )
                # 得到损失值，精度，反向传播，并对参数更新
                label_tensor = torch.zeros(output.shape[0], dtype=torch.long)
                label_tensor[:len(positive_nids)] = torch.ones(len(positive_nids), dtype=torch.long)
                label_tensor = label_tensor.to(DefaultConfig.DEVICE)

                loss_train = self.loss(output, label_tensor)
                acc_train = accuracy(output, label_tensor).item()
                recall_train = recall_percent(output, label_tensor)

                acc_lst.append(acc_train)
                recall_lst.append(recall_train)

                # temp = torch.randint(output.shape[0], (5,))
                # print(f"output:{output[temp]}")
                # print(f"label:{label[temp]}")
                # print(f"acc:{acc_train},loss:{loss_train.item()}")

                self.optimizer.zero_grad()  # 计算梯度重置
                loss_train.backward()
                self.optimizer.step()

                self.writer.add_scalar("loss_train", loss_train, self.train_counter)
                self.writer.add_scalar("acc_train", acc_train, self.train_counter)
                self.writer.add_scalar("recall_train", recall_train, self.train_counter)
                self.train_counter += 1

        acc_train = np.mean(acc_lst)
        recall_train = np.mean(recall_lst)
        self.writer.add_scalar("acc_train_epoch", acc_train, epoch * 10 + flag)
        self.writer.add_scalar("recall_train_epoch", recall_train, epoch * 10 + flag)

        acc_lst = []
        recall_lst = []
        self.model.eval()  # 设置为eval模式
        for i in range(len(self.val_data)):
            traj_id, value = self.val_data[i]
            dgl_cellular_id_lst = torch.tensor(self.dic_cellular_traj[traj_id]).to(DefaultConfig.DEVICE)
            for train_data in value:
                positive_nids = [pair[0] for pair in train_data[0]]
                positive_dis = [pair[1] for pair in train_data[0]]
                negative_nids = [pair[0] for pair in train_data[1]]
                negative_dis = [pair[1] for pair in train_data[1]]

                label_nids = torch.zeros(len(positive_nids) + len(negative_nids),dtype=torch.long)
                label_nids[:len(positive_nids)] = torch.tensor(positive_nids,dtype=torch.long)
                label_nids[len(positive_nids):] = torch.tensor(negative_nids,dtype=torch.long)
                label_nids = label_nids.to(DefaultConfig.DEVICE)

                node_dis = torch.zeros(len(positive_dis) + len(negative_dis))
                node_dis[:len(positive_dis)] = torch.tensor(positive_dis)
                node_dis[len(positive_dis):] = torch.tensor(negative_dis)
                node_dis = node_dis.to(DefaultConfig.DEVICE)

                output = self.model(
                    dgl_cellular_id_lst,
                    label_nids,
                    self.dgl_graph,
                    node_dis
                )
                # 得到损失值，精度，反向传播，并对参数更新
                label_tensor = torch.zeros(output.shape[0], dtype=torch.long)
                label_tensor[:len(positive_nids)] = torch.ones(len(positive_nids), dtype=torch.long)
                label_tensor = label_tensor.to(DefaultConfig.DEVICE)

                loss_val = self.loss(output, label_tensor)
                acc_val = accuracy(output, label_tensor).item()
                recall_val = recall_percent(output, label_tensor)

                acc_lst.append(acc_val)
                recall_lst.append(recall_val)

                self.writer.add_scalar("loss_val", loss_val, self.val_counter)
                self.writer.add_scalar("acc_val", acc_val, self.val_counter)
                self.writer.add_scalar("recall_val", recall_val, self.val_counter)
                self.val_counter += 1

        acc_val = np.mean(acc_lst)
        recall_val = np.mean(recall_lst)
        self.writer.add_scalar("acc_val_epoch", acc_val, epoch * 10 + flag)
        self.writer.add_scalar("recall_val_epoch", recall_val, epoch * 10 + flag)

        print('Epoch: {:0d}'.format(epoch),
              'acc_train: {:.15f}'.format(acc_train),
              'recall_train: {:.15f}'.format(recall_train),
              'acc_val: {:.15f}'.format(acc_val),
              'recall_val: {:.15f}'.format(recall_val),
              'time: {:.15f}s'.format(time.time() - t))


if __name__ == "__main__":
    train_obj = trans_train()
    # print(DefaultConfig.DEVICE)
    train_obj.train()
