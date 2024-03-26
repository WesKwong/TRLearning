import itertools as it

class Config():
    def __init__(self):
        # -------------------- global settings ------------------- #
        self.data_path = 'data/'
        self.results_path = 'results/'
        self.random_seed = 42

        # ------------------ experiment settings ----------------- #
        # name: name of the setting
        # N: N-gram
        # smooth: smoothing method: 'none', 'add1', 'gt'

        self.expt_settings = []
        # default setting
        default_setting = self.get_default_setting()
        self.expt_settings.append(default_setting)

    def get_default_setting(self):
        set = {'name': ['delault']}
        set['N'] = [2, 3]
        set['smooth'] = ['none', 'add1', 'gt']
        return set

    # ------ helper function to get experiment settings ------ #
    def get_expt_settings(self):
        expt_settings = {}
        for setting in self.expt_settings:
            combinations = it.product(*(setting[name] for name in setting))
            expt_setting = [{key : value[i] for i,key in enumerate(setting)}for value in combinations]
            expt_settings[setting['name'][0]] = expt_setting
        return expt_settings

config = Config()