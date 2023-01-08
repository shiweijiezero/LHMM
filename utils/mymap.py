# from utils.mybase import Dataset, split_dataframe
# from .coord_transform import wgs84_to_bd09
import os
import pickle
from numpy import array
from config import DefaultConfig
from scipy.spatial.kdtree import KDTree
import networkx
import sys
from haversine import haversine, Unit


# wgs84
class MapDataset():
    def __init__(self, opt=DefaultConfig()):
        self.opt = opt
        self.nid2rids = {}
        self.pos2nid = {}
        self.rid2nids = {}
        # self.rid2roadinfo = {}

        print("Reading kdtree file...")
        if os.path.exists(opt.kdtree_path):
            with open(opt.kdtree_path, mode='rb') as f:
                self.kdtree: KDTree = pickle.load(f)
        else:
            print("no kdtress")

        print("Reading nxgraph file...")
        if os.path.exists(opt.nxgraph_path):
            with open(opt.nxgraph_path, mode='rb') as f:
                self.nxgraph: networkx.DiGraph = pickle.load(f)
        else:
            print("no nxgraph")

        print("Reading pos2nid file...")
        if os.path.exists(opt.pos2nid_path):
            with open(opt.pos2nid_path, mode='rb') as f:
                self.pos2nid: dict = pickle.load(f)
        else:
            print("no pos2nid")

        print("Reading rid2nids file...")
        if os.path.exists(opt.rid2nids_path):
            with open(opt.rid2nids_path, mode='rb') as f:
                self.rid2nids: dict = pickle.load(f)
        else:
            print("no rid2nids")

        print("Reading nid2rids file...")
        if os.path.exists(opt.nid2rids_path):
            with open(opt.nid2rids_path, mode='rb') as f:
                self.nid2rids: dict = pickle.load(f)
        else:
            print("no nid2rids")

        print("Finish all parsing.")

    def get_pos_by_node_id(self, nid):
        pos_lat = self.nxgraph.nodes[nid]["lat"]
        pos_lng = self.nxgraph.nodes[nid]["lng"]
        return pos_lng, pos_lat

    def get_batch_pos_by_node_id(self, nids):
        res = []
        for nid in nids:
            pos_lat = self.nxgraph.nodes[nid]["lat"]
            pos_lng = self.nxgraph.nodes[nid]["lng"]
            res.append((pos_lng, pos_lat))
        return res

    def get_nid_by_pos(self, pos):
        return self.pos2nid[pos]

    def get_other_way_nodes_pos_by_road_id(self, road_id):
        road_src, road_dst = self.rid2nids[road_id]
        road_way = self.nxgraph[road_src][road_dst]["way"]
        nodesinfo = road_way.split(';')
        i = 0  # !!
        othernodes = []
        while (i < (len(nodesinfo) / 2)):
            lat = float(nodesinfo[2 * i])
            lng = float(nodesinfo[2 * i + 1])
            i += 1
            othernodes.append((lng, lat))

        # # 如果解开这个注释就返回所有的node，否则返回中间node
        # all_nodes = [self.get_node_by_id(beginid)] + othernodes + [self.get_node_by_id(endid)]
        return othernodes

    def query_ball_points(self, pos, range=100):
        nids = self.kdtree.query(array([pos]), k=range)[1].tolist()[0]
        # nids = self.kdtree.query_ball_point(pos,range)

        return nids

    def get_distance(self, pos1, pos2):
        # pos1:longitude,latitude
        pos1 = (pos1[1], pos1[0])
        pos2 = (pos2[1], pos2[0])
        return haversine(pos1, pos2, unit=Unit.KILOMETERS)

    def get_point2line(self,x,y, sx1, sy1, sx2, sy2):
        ppx,ppy=self.getClosestPoint(x,y, sx1, sy1, sx2, sy2)
        return self.get_distance((x,y),(ppx,ppy))

    def getClosestPoint(self, x, y, sx1, sy1, sx2, sy2):
        ppx, ppy = self.getProjection(x, y, sx1, sy1, sx2, sy2)

        if (sx1 < sx2):
            if (ppx <= sx1):
                ppx = sx1
                ppy = sy1
            elif (ppx >= sx2):
                ppx = sx2
                ppy = sy2
        elif (sx1 > sx2):
            if (ppx <= sx2):
                ppx = sx2
                ppy = sy2
            elif (ppx >= sx1):
                ppx = sx1
                ppy = sy1
        else:  # sx1 == sx2
            if (ppx != sx1):
                raise Exception(
                    "The point projection is not on the road: " +
                    ppx + "," + ppy + "," +
                    sx1 + "," + sy1 + "," + sx2 + "," + sy2
                )
        return ppx, ppy

    def getProjection(self, x, y, sx1, sy1, sx2, sy2):
        xDelta = sx2 - sx1
        yDelta = sy2 - sy1
        a = sy2 - sy1
        b = sx1 - sx2
        c = sx2 * sy1 - sx1 * sy2
        if (xDelta == 0):
            ppx = sx1
            ppy = y
            return ppx, ppy
        if (yDelta == 0):
            ppx = x
            ppy = sy1
            return ppx, ppy
        ppx = (b * b * x - a * b * y - a * c) / (a * a + b * b)
        ppy = (-a * b * x + a * a * y - b * c) / (a * a + b * b)
        return ppx, ppy


if __name__ == "__main__":
    base_path = os.path.abspath("../")
    sys.path.append(base_path)
    mymap = MapDataset()
    a = mymap.get_pos_by_node_id(1)
    b = mymap.get_nid_by_pos((120.17188042786451, 30.28081369282634))
    c = mymap.get_other_way_nodes_pos_by_road_id("-89440|0")
    print(a)
    print(b)
    print(c)
