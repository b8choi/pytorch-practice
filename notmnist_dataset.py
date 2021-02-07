import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import os
import pickle


class NotMNIST(Dataset):
    config = {
        'small': {
            'url': 'http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz',
            'paths': {
                'raw_file': 'notMNIST_small.tar.gz',
                'raw_dir': 'notMNIST_small',
                'pkl_file': 'notMNIST_small.pkl'
            }
        },
        'large': {
            'url': 'http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz',
            'paths': {
                'raw_file': 'notMNIST_small.tar.gz',
                'raw_dir': 'notMNIST_small',
                'pkl_file': 'notMNIST_small.pkl'
            }
        },
    }


    def __init__(self, size, root, train, transform, download=True):
        self.root = root
        self.transform = transform
        self.paths = self.config[size]['paths']
        self.dataset = []

        pickle_path = os.path.join(root, self.paths['pkl_file'])

        # pkl 파일이 존재하나 transform 함수가 다를 경우 기존 pkl 파일 삭제
        if os.path.isfile(pickle_path):
            with open(pickle_path, 'rb') as f:
                self.dataset = pickle.load(f)

            if self.dataset['transform'] != type(self.transform):
                print('existing dataset\'s transform function not matched!')
                os.remove(pickle_path)

        if not os.path.isfile(pickle_path):
            if download:
                self.get(self.config[size]['url'])
                self.extract()
                self.build()
            else:
                print('dataset not found!')
                return

        with open(pickle_path, 'rb') as f:
            self.dataset = pickle.load(f)

        if train:
            num = int(len(self) * 0.85)
            self.dataset['content'] = self.dataset['content'][:num]
        else:
            num = int(len(self) * 0.85)
            self.dataset['content'] = self.dataset['content'][num:]


    def __len__(self):
        return len(self.dataset['content'])


    def __getitem__(self, idx):
        data = self.dataset['content'][idx]
        x = data['image']
        y = ord(data['label']) - ord('A')

        return (x, y)


    def get(self, url):
        file_path = os.path.join(self.root, self.paths['raw_file'])
        if not os.path.isfile(file_path):
            print('downloading dataset...')

            import requests
            response = requests.get(url)
            with open(file_path, 'wb') as f:
                f.write(response.content)


    def extract(self):
        dir_path = os.path.join(self.root, self.paths['raw_dir'])
        if not os.path.isdir(dir_path):
            print('extracting dataset...')

            import tarfile
            file_path = os.path.join(self.root, self.paths['raw_file'])
            with tarfile.open(file_path) as t:
                t.extractall(self.root)


    def build(self):
        print('building dataset...')

        dataset = dict()
        dataset['transform'] = type(self.transform)
        dataset['content'] = []

        dir_path = os.path.join(self.root, self.paths['raw_dir'])
        dirs = os.listdir(dir_path)
        for d in dirs:
            files = os.listdir(os.path.join(dir_path, d))
            for f in files:
                try:
                    file_path = os.path.join(dir_path, d, f)
                    import PIL
                    img = PIL.Image.open(file_path)
                    img = self.transform(img)
                except OSError:
                    print('  cannot read the image', file_path)
                    continue

                data = {'image': img, 'label': d}
                dataset['content'].append(data)

        pickle_path = os.path.join(self.root, self.paths['pkl_file'])
        with open(pickle_path, 'wb') as f:
            pickle.dump(dataset, f)


if __name__ == '__main__':
    data = NotMNIST(size='small', root='./data/', train=True, transform=transforms.ToTensor(), download=True)
    print(data[0])
