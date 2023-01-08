import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.nn.init as init

# 导入数据
import networkx as nx
G = nx.karate_club_graph()
print(G.number_of_nodes()) # 34
print(G.number_of_edges()) # 78
#
LEARNING_RATE = 0.1
WEIGHT_DACAY = 5e-4
EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def norm(adj):
    # 增加自循环且根据度进行拉普拉斯归一化
    adj += np.eye(adj.shape[0]) # 为每个结点增加自环
    degree = np.array(adj.sum(1)) # 为每个结点计算度
    degree = np.diag(np.power(degree, -0.5))
    return degree.dot(adj).dot(degree)

class GraphConvolution(nn.Module):
    # 卷积层
    def __init__(self, input_size, output_size):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, adj, features):
        out = torch.mm(adj, features)
        out = self.linear(out)
        return out

class GCN(nn.Module):
    def __init__(self, input_size=34, hidden_size=5):
        super(GCN, self).__init__()
        self.gcn1 = GraphConvolution(input_size, hidden_size)
        self.gcn2 = GraphConvolution(hidden_size, 2)

    def forward(self, adj, features):
        out = F.relu(self.gcn1(adj, features))
        out = self.gcn2(adj, out)
        return out

# 训练数据准备
features = np.eye(34, dtype=np.float)

y = np.zeros(G.number_of_nodes())
for i in range(G.number_of_nodes()):
    if G.nodes[i]['club'] == 'Mr. Hi':
        y[i] = 0
    else:
        y[i] = 1

adj = np.zeros((34, 34)) # 邻阶矩阵
for k, v in G.adj.items():
    for item in v.keys():
        adj[k][item] = 1
adj = norm(adj)


features = torch.tensor(features, dtype=torch.float).to(DEVICE)
y = torch.tensor(y, dtype=torch.long).to(DEVICE)
adj = torch.tensor(adj, dtype=torch.float).to(DEVICE)


# 训练
net = GCN().to(DEVICE)
loss = nn.CrossEntropyLoss().to(DEVICE)
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DACAY)


for epoch in range(EPOCHS):
    out = net(adj, features)
    mask = [False if x != 0 and x != 33 else True for x in range(34)] # 只选择管理员和教练进行训练
    l = loss(out[mask], y[mask])
    optimizer.zero_grad()
    l.backward()
    optimizer.step()

    print(f"epoch: {epoch}, loss: {l.item()}")


# 可视化
r = net(adj, features).cpu().detach().numpy()
import matplotlib.pyplot as plt
fig = plt.figure()
for i in range(34):
    plt.scatter(r[i][0], r[i][1], color="r" if y[i] == 0 else 'b')
plt.show()
