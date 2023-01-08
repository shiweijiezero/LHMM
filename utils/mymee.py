import pickle
import config


class MEEDataset():
    def __init__(self, opt=config.DefaultConfig()):
        self.opt = opt
        with open(opt.filtered_mee_traj_path, mode='rb') as f:
            self.mee_traj_dict = pickle.load(f)

    def get_mee_line_with_ts(self, vid):
        return self.mee_traj_dict[vid]

    def get_mee_line_without_ts(self, vid):
        return [(traj_obj[0], traj_obj[1]) for traj_obj in self.mee_traj_dict[vid]]

    def get_all_id(self):
        return list(self.mee_traj_dict.keys())