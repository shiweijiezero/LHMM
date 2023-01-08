import os
import sys

base_path = os.path.abspath("../")
sys.path.append(base_path)

from utils.mymap import MapDataset

base_path = os.path.abspath("../")
sys.path.append(base_path)
from pygcn.gcn_utils import *
from pygcn.gcn_model import *


class gcn_eval:
    def __init__(self):
        np.random.seed(DefaultConfig.seed)
        torch.manual_seed(DefaultConfig.seed)
        if (DefaultConfig.DEVICE != "cpu"):
            torch.cuda.manual_seed(DefaultConfig.seed)
        # Load data
        self.mymap = MapDataset()
        self.train_data, \
        self.val_data, \
        self.dgl_graph, \
        self.dgl_cellular_id2pos \
            = load_data()
        print("数据加载完成！")
        self.gcn_graph = GCN_graph()
        # Model and optimizer
        if (DefaultConfig.use_model != ""):
            self.model = torch.load(DefaultConfig.use_model,map_location=DefaultConfig.DEVICE)  # 导入保存的模型
            print("导入保存的模型!", DefaultConfig.use_model)
        else:
            self.model = GCN(
                nfeat=DefaultConfig.gcn_nfeat,
                nhid=DefaultConfig.hidden,
                nclass=DefaultConfig.nclass,
                dropout=DefaultConfig.dropout,
                rel_names=self.dgl_graph.etypes,
                road_num_nodes=self.dgl_graph.num_nodes('road'),
                cellular_num_nodes=self.dgl_graph.num_nodes('cellular')
            )
        self.model = self.model.to(DefaultConfig.DEVICE)
        self.pos2dgl_cellular_id_5 = {}  # 为了与scala端对接
        for pos, cellular_id in self.gcn_graph.pos2dgl_cellular_id.items():
            x = round(pos[0], 5)
            y = round(pos[1], 5)
            self.pos2dgl_cellular_id_5[(x, y)] = cellular_id

    def eval(self, meePositionList, candidate_nids_lst):
        self.model.eval()  # 设置为eval模式
        # self.model.train()
        # 将cellular坐标转换为cellular_id
        mee_idx_lst = [self.pos2dgl_cellular_id_5[(round(pos[0], 5), round(pos[1], 5))] for pos in meePositionList]
        # print(f"mee_idx_lst:{mee_idx_lst}")

        cellular_pos_lst = [self.dgl_cellular_id2pos[id] for id in mee_idx_lst]
        node_pos_lst = [self.mymap.get_pos_by_node_id(nid) for nid in candidate_nids_lst]
        node_distance = [min([self.mymap.get_distance(cell_pos, node_pos) for cell_pos in cellular_pos_lst]) for
                         node_pos in node_pos_lst]

        mee_idx_lst = torch.tensor(mee_idx_lst, dtype=torch.long).to(DefaultConfig.DEVICE)
        node_distance = torch.tensor(node_distance).to(DefaultConfig.DEVICE)
        res_lst=torch.zeros((len(candidate_nids_lst),2)) # n*2
        # drop_times=10
        # for i in range(drop_times):
        output = self.model(
            mee_idx_lst,
            candidate_nids_lst,
            self.dgl_graph,
            node_distance
        )
            # res_lst+=output*(1/drop_times)
        return output

# if __name__ == "__main__":
# model_obj = gcn_eval()
# model_obj.eval()
