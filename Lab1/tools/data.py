import os

class MyData:
    def __init__(self):
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def get_train_set(self):
        return self.train_set

    def get_val_set(self):
        return self.val_set

    def get_test_set(self):
        return self.test_set

    def get_selected_set(self, train=True, val=True, test=True):
        sets = {"train": None, "val": None, "test": None}
        if train:
            sets["train"] = self.get_train_set()
        if val:
            sets["val"] = self.get_val_set()
        if test:
            sets["test"] = self.get_test_set()
        return sets

class NgramData(MyData):
    def __init__(self):
        super().__init__()

    def load_data(self, path, name, target='train'):
        data_path = os.path.join(path, name)
        with open(data_path, 'r') as f:
            sentences = f.readlines()
        data = []
        for sentence in sentences:
            words = ['<s>']
            words += sentence.strip().split(' ')
            words += ['</s>']
            data.append(words)
        if target == 'train':
            self.train_set = data
        elif target == 'val':
            self.val_set = data
        elif target == 'test':
            self.test_set = data
