import pickle
from haversine import haversine, Unit
import numpy as np
import math


def split_dataframe(df, n):
    chunk_size = int(df.shape[0] / n)
    chunks = [df.iloc[df.index[i:i + chunk_size]] for i in range(0, df.shape[0], chunk_size)]
    return chunks


class TrajPoint():
    def __init__(self, lng, lat, ts, has_guest=0):
        self.lng = float(lng)
        self.lat = float(lat)
        self.ts = ts
        self.has_guest = has_guest
        pass

    def __repr__(self):
        return "<%f,%f,%s>" % (self.lng, self.lat, self.ts)
        # return f"{{lng:{self.lng},lat:{self.lat},ts:{self.ts},has_guest:{self.has_guest}}}"

    def slotted_to_dict(self):
        return {s: getattr(self, s) for s in ['lng', 'lat', 'ts', 'has_guest']}

    @property
    def pos(self):
        return (self.lng, self.lat)


class TrajLine():
    vid = 0
    slice_index = 0, 0
    gps_line = []  # Traj obj with ts
    mee_line = []

    @property
    def tid(self):
        return "%d[%d:%d]" % (self.vid, self.slice_index[0], self.slice_index[1])

    def __repr__(self):
        return "TrajLine%s" % self.tid

    def gps_pos_line(self):
        return [(x.lng, x.lat) for x in self.gps_line]


class Dataset():
    df = None
    opt = None
    obj_fname = ""

    def __getitem__(self, index):
        return self.df.iloc[index, :]

    def __len__(self):
        return len(self.df)

    def save(self):
        with open(self.obj_fname, mode='wb') as f:
            pickle.dump(self, f, protocol=2)
        print("Save %s obj to [%s]" % (self.__class__.__name__, self.obj_fname))


def interval_between(t1: TrajPoint, t2: TrajPoint):
    return (t2.ts - t1.ts).seconds


def dis_between(t1: TrajPoint, t2: TrajPoint):
    # return haversine(t1.pos, t2.pos, unit=Unit.KILOMETERS)
    return dis_between_pos(t1.pos, t2.pos)


EARTH_RADIUS = 6371000


def dis_between_pos(pos1: tuple, pos2: tuple): #KM
    def toRadians(angdeg):
        return angdeg * 0.017453292519943295

    import math
    EARTH_RADIUS = 6371000
    lon1,lat1  = pos1
    lon2,lat2= pos2
    dLat = toRadians(lat2 - lat1)
    dLon = toRadians(lon2 - lon1)

    tmp = math.cos(toRadians((lat1 + lat2) / 2)) * dLon
    # print(tmp)

    normedDist = dLat * dLat + tmp * tmp
    return EARTH_RADIUS * math.sqrt(normedDist) / 1000
    # return haversine(pos1, pos2, unit=Unit.KILOMETERS)


def speed_between(t1: TrajPoint, t2: TrajPoint):
    dis = dis_between(t1, t2)
    interval = interval_between(t1, t2)
    if (interval == 0):
        interval = 0.00001
    speed = dis * 3600 / interval  # km/h
    return speed


def angle_between(t1: TrajPoint, t2: TrajPoint, t3: TrajPoint):
    a = t1.pos
    b = t2.pos
    c = t3.pos

    ang = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    if (ang < 0):
        ang += 360
    if (ang > 180):
        ang = 360 - ang
    return ang

    a = np.array(t1.pos)
    b = np.array(t2.pos)
    c = np.array(t3.pos)
    ba = a - b
    bc = c - b
    if (np.linalg.norm(ba) * np.linalg.norm(bc) == 0):
        return 0
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# def angle3pt(a, b, c):
#     """Counterclockwise angle in degrees by turning from a to c around b
#         Returns a float between 0.0 and 360.0"""
#     ang = math.degrees(
#         math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
#     return ang + 360 if ang < 0 else ang
