import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import os
import pickle


class MarvelHeroes(Dataset):
    config = {
        'paths': {
            'raw_dir': 'marvel_heroes',
            'pkl_file': 'marvel_heroes.pkl'
        }
    }


    def __init__(self, root='./', train=True):
        self.root = root
        self.paths = self.config['paths']
        self.transform = img = transforms.Compose([transforms.Resize(128), transforms.CenterCrop(128), transforms.ToTensor()])
        self.dataset = []

        pickle_path = os.path.join(root, self.paths['pkl_file'])

        if not os.path.isfile(pickle_path):
            self.build()

        with open(pickle_path, 'rb') as f:
            self.dataset = pickle.load(f)
            import random
            random.shuffle(self.dataset)

        num = int(len(self) * 0.85)
        if train:
            self.dataset = self.dataset[:num]
        else:
            self.dataset = self.dataset[num:]


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        data = self.dataset[idx]
        x = data['image']
        y = data['label']
        return (x, y)


    def build(self):
        print('building dataset...')

        dataset = []

        dir_path = os.path.join(self.root, self.paths['raw_dir'])
        dirs = os.listdir(dir_path)
        for i, d in enumerate(dirs):
            files = os.listdir(os.path.join(dir_path, d))
            for f in files:
                try:
                    file_path = os.path.join(dir_path, d, f)
                    import PIL
                    img = PIL.Image.open(file_path)
                    img = img.convert('RGBA')
                    bg = PIL.Image.new('RGBA', img.size, (255, 255, 255))
                    img = PIL.Image.alpha_composite(bg, img)
                    img = self.transform(img)
                except OSError:
                    print('  cannot read the image', file_path)
                    continue

                data = {'image': img[:3], 'label': i}
                dataset.append(data)

        pickle_path = os.path.join(self.root, self.paths['pkl_file'])
        with open(pickle_path, 'wb') as f:
            pickle.dump(dataset, f)


if __name__ == '__main__':
    data = MarvelHeroes(root='./data/', train=True)
    for i in range(5):
        print(data[i])
