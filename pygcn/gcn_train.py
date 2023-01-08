import os
import sys
import copy

base_path = os.path.abspath("../")
sys.path.append(base_path)

from utils.mymap import MapDataset

base_path = os.path.abspath("../")
sys.path.append(base_path)
from pygcn.gcn_utils import *
from pygcn.gcn_model import *
import time
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import os
from tqdm import tqdm
from pygcn.facal_loss import FocalLoss
import numpy as np

with open("../data/roadID2relationship.obj", mode='rb') as f:
    roadID2relationship = pickle.load(f)


class gcn_train:
    def __init__(self):
        random.seed(DefaultConfig.seed)
        np.random.seed(DefaultConfig.seed)
        torch.manual_seed(DefaultConfig.seed)
        if (DefaultConfig.DEVICE != "cpu"):
            torch.cuda.manual_seed(DefaultConfig.seed)
        self.time_name = "{}-{}-{}-{}-{}-{}-{}-{}".format(
            time.strftime("%Y-%m-%d-%H-%M"),
            DefaultConfig.gcn_nfeat,
            DefaultConfig.hidden,
            DefaultConfig.negetive_weight,
            DefaultConfig.positive_weight,
            DefaultConfig.lr,
            DefaultConfig.dropout,
            DefaultConfig.train_set_rate
        )
        self.writer = SummaryWriter('../data/log/{}'.format(self.time_name))
        # Load data
        self.mymap = MapDataset()
        self.train_data, \
        self.val_data, \
        self.dgl_graph, \
        self.dgl_cellular_id2pos \
            = load_data()
        print("数据加载完成！")
        # Model and optimizer
        print(self.dgl_graph.num_nodes('road'), )
        print(self.dgl_graph.num_nodes('cellular'), )
        if (DefaultConfig.use_model != ""):
            self.model = torch.load(DefaultConfig.use_model, map_location=DefaultConfig.DEVICE)  # 导入保存的模型
        else:
            self.model = GCN(nfeat=DefaultConfig.gcn_nfeat,
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

        # self.loss = FocalLoss(
        #     weight=torch.Tensor([DefaultConfig.negetive_weight, DefaultConfig.positive_weight]),
        #     gamma=2).to(DefaultConfig.DEVICE)  # focal交叉熵损失
        # self.loss = nn.NLLLoss(torch.Tensor([DefaultConfig.negetive_weight,DefaultConfig.positive_weight])).to(DefaultConfig.DEVICE) # 负对数似然损失
        # self.loss = nn.BCELoss(torch.Tensor([DefaultConfig.negetive_weight,DefaultConfig.positive_weight])).to(DefaultConfig.DEVICE) # 二分类交叉熵损失
        self.train_counter = 0
        self.val_counter = 0

    def train(self):
        # summary(self.model)
        for epoch in range(DefaultConfig.EPOCHS):
            self.train_epoch(epoch)
            self.model.get_embedding(self.dgl_graph)
            if (not os.path.exists("/home/weijie0311/Project/mapmatching-cikm/data/model/{}".format(self.time_name))):
                os.makedirs("/home/weijie0311/Project/mapmatching-cikm/data/model/{}".format(self.time_name))
            torch.save(self.model,
                       "/home/weijie0311/Project/mapmatching-cikm/data/model/{}/mlpl-{}.pkl".format(
                           self.time_name, epoch))  # 保存整个模型
            print(f"保存模型至" + "/home/weijie0311/Project/mapmatching-cikm/data/model/{}/mlpl-{}.pkl".format(
                self.time_name, epoch))
        self.writer.close()

    def train_epoch(self, epoch):
        # 训练一个epoch
        # 暂时先不设置 batch 训练
        t = time.time()
        acc_lst = []
        recall_lst = []
        self.model.train()  # 设置为训练模式
        for i in tqdm(range(len(self.train_data))):
            traj_id, train_item = self.train_data[i]
            dgl_cellular_id_lst, positive_nids, negetive_nids = train_item
            label_nids = copy.deepcopy(positive_nids)
            label_nids.extend(negetive_nids)

            cellular_pos_lst = [self.dgl_cellular_id2pos[id] for id in dgl_cellular_id_lst]
            node_pos_lst = [
                self.mymap.get_pos_by_node_id(self.mymap.rid2nids[roadID2relationship[nid]][0])
                for nid in label_nids
            ]

            node_distance = [min([self.mymap.get_distance(cell_pos, node_pos) for cell_pos in cellular_pos_lst]) for
                             node_pos in node_pos_lst]

            dgl_cellular_id_lst = torch.tensor(dgl_cellular_id_lst).to(DefaultConfig.DEVICE)
            label_nids = torch.tensor(label_nids).to(DefaultConfig.DEVICE)
            node_distance = torch.tensor(node_distance).to(DefaultConfig.DEVICE)

            output = self.model(
                dgl_cellular_id_lst,
                label_nids,
                self.dgl_graph,
                node_distance
            )
            # 得到损失值，精度，反向传播，并对参数更新
            label_tensor = torch.zeros(output.shape[0], dtype=torch.long)
            label_tensor[:len(positive_nids)] = torch.ones(len(positive_nids), dtype=torch.long)
            label_tensor = label_tensor.to(DefaultConfig.DEVICE)

            loss_train = self.loss(output, label_tensor)
            acc_train = accuracy(output, label_tensor).item()
            recall_train = recall_percent(output, label_tensor)

            # temp = torch.randint(output.shape[0], (5,))
            # print(f"output:{output[temp]}")
            # print(f"label_tensor:{label_tensor[temp]}")
            # print(f"acc:{acc_train},recall:{recall_train}")

            acc_lst.append(acc_train)
            recall_lst.append(recall_train)

            self.optimizer.zero_grad()  # 计算梯度重置
            loss_train.backward()
            self.optimizer.step()

            self.writer.add_scalar("loss_train", loss_train, self.train_counter)
            self.writer.add_scalar("acc_train", acc_train, self.train_counter)
            self.writer.add_scalar("recall_train", recall_train, self.train_counter)
            self.train_counter += 1

        acc_train = np.mean(acc_lst)
        recall_train = np.mean(recall_lst)
        self.writer.add_scalar("acc_train_epoch", acc_train, epoch)
        self.writer.add_scalar("recall_train_epoch", recall_train, epoch)

        # acc_lst = []
        # recall_lst = []
        # self.model.eval()  # 设置为eval模式
        # for i in range(len(self.val_data)):
        #     traj_id, train_item = self.val_data[i]
        #     dgl_cellular_id_lst, positive_nids, negetive_nids = train_item
        #     label_nids = copy.deepcopy(positive_nids)
        #     label_nids.extend(negetive_nids)
        #
        #     cellular_pos_lst = [self.dgl_cellular_id2pos[id] for id in dgl_cellular_id_lst]
        #     node_pos_lst = [self.mymap.get_pos_by_node_id(nid) for nid in label_nids]
        #     node_distance = [min([self.mymap.get_distance(cell_pos, node_pos) for cell_pos in cellular_pos_lst]) for
        #                      node_pos in node_pos_lst]
        #
        #     dgl_cellular_id_lst = torch.tensor(dgl_cellular_id_lst).to(DefaultConfig.DEVICE)
        #     label_nids = torch.tensor(label_nids).to(DefaultConfig.DEVICE)
        #     node_distance = torch.tensor(node_distance).to(DefaultConfig.DEVICE)
        #
        #     output = self.model(
        #         dgl_cellular_id_lst,
        #         label_nids,
        #         self.dgl_graph,
        #         node_distance
        #     )
        #     # 得到损失值，精度
        #     label_tensor = torch.zeros(output.shape[0], dtype=torch.long)
        #     label_tensor[:len(positive_nids)] = torch.ones(len(positive_nids), dtype=torch.long)
        #     label_tensor = label_tensor.to(DefaultConfig.DEVICE)
        #     loss_val = self.loss(output, label_tensor)
        #     acc_val = accuracy(output, label_tensor).item()
        #     recall_val = recall_percent(output, label_tensor)
        #     acc_lst.append(acc_val)
        #     recall_lst.append(recall_val)
        #     self.writer.add_scalar("loss_val", loss_val.item(), self.val_counter)
        #     self.writer.add_scalar("acc_val", acc_val, self.val_counter)
        #     self.writer.add_scalar("recall_val", recall_val, self.val_counter)
        #     self.val_counter += 1
        # acc_val = sum(acc_lst) / len(acc_lst)
        # recall_val = sum(recall_lst) / len(recall_lst)
        # self.writer.add_scalar("acc_val_epoch", acc_val, epoch)
        # self.writer.add_scalar("recall_val_epoch", recall_val, epoch)

        print('Epoch: {:0d}'.format(epoch),
              'acc_train: {:.15f}'.format(acc_train),
              'recall_train: {:.15f}'.format(recall_train),
              # 'acc_val: {:.15f}'.format(acc_val),
              # 'recall_val: {:.15f}'.format(recall_val),
              'time: {:.15f}s'.format(time.time() - t))


if __name__ == "__main__":
    train_obj = gcn_train()
    # print(DefaultConfig.DEVICE)
    train_obj.train()
