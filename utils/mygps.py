import pickle
import config


class GPSDataset():
    def __init__(self, opt=config.DefaultConfig()):
        self.opt = opt
        # with open(opt.gps_traj_path, mode='rb') as f:
        #     self.gps_traj_dict:dict = pickle.load(f)
        with open(opt.filtered_gps_traj_path, mode='rb') as f:
            self.gps_traj_dict:dict=pickle.load(f)

    def get_gps_line_with_ts(self, vid):
        return self.gps_traj_dict[vid]

    def get_gps_line_without_ts(self, vid):
        return [(traj_obj[0], traj_obj[1]) for traj_obj in self.gps_traj_dict[vid]]

    def get_all_id(self):
        return list(self.gps_traj_dict.keys())
