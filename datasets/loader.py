import glob
import cv2
import numpy as np
import torch
import kornia

from torch.utils.data import Dataset


class ReversedISPDataset(Dataset):
    def __init__(self, root, debug=False, test=None):
        self.root = root
        self.test = test
        self.flipper = kornia.augmentation.RandomHorizontalFlip(p=1.0)
        self.rgbs = sorted(glob.glob(self.root + '/test_rgb/*.jpg')) if self.test else sorted(glob.glob(self.root + '/train/*.jpg'))
        self.raws = None if self.test else sorted(glob.glob(self.root + '/train/*.npy'))
            
        self.debug = debug
        if self.debug:
            self.rgbs = self.rgbs[:100]
            self.raws = self.raws[:100] if not self.test else self.raws
        
    def __len__(self):
        return len(self.rgbs)

    def __getitem__(self, idx):
        flip_p = np.random.rand()
        rgb = self.load_img(self.rgbs[idx], norm=True)
        rgb = torch.from_numpy(rgb.transpose((2, 0, 1)))
        
        if self.test:
            return rgb, self.rgbs[idx]
        else:
            raw = self.load_raw(self.raws[idx])
            raw = torch.from_numpy(raw.transpose((2, 0, 1)))
            if flip_p > 0.5:
                return self.flipper(rgb).squeeze(0), self.flipper(raw).squeeze(0) 
            else:
                return rgb, raw
            
        
    def load_img(self, filename, norm=True):
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if norm:   
            img = img / 255.
            img = img.astype(np.float32)
            
        return img
    
    def load_raw(self, raw, max_val=2**10):
        raw = np.load(raw)/ max_val
        return raw.astype(np.float32)
    
    
if __name__ == '__main__':
    dataset = ReversedISPDataset('data/p20')
    for i in range(len(dataset)):
        rgb, raw = dataset[i]
        print(rgb.shape, raw.shape)
    print(len(dataset))
    