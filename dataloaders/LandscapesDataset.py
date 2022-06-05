import random
import torch
from torchvision import transforms as TR
import os
from PIL import Image
import numpy as np
import torch.nn.functional as F

class LandscapesDataset(torch.utils.data.Dataset):
    def __init__(self, opt, for_metrics, test = False):
        if opt.phase == "test" or for_metrics:
            opt.load_size = opt.size
        else:
            opt.load_size = opt.size
        # opt.crop_size = 256
        opt.crop_size = opt.size
        opt.label_nc = 182
        opt.contain_dontcare_label = True
        opt.semantic_nc = 183 # label_nc + unknown
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False
        opt.aspect_ratio = 1.0
        self.test = test

        self.opt = opt
        self.for_metrics = for_metrics
        self.images, self.labels, self.paths = self.list_images()

    def __len__(self,):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.paths[0], self.images[idx])).convert('RGB')
        # label = Image.open(os.path.join(self.paths[1], self.labels[idx]))
        label = np.load(os.path.join(self.paths[1], self.labels[idx]))
        image, label = self.transforms(image, label)
        label = label + 1
        # print(torch.unique(label))
        label[label == 256] = 0 # unknown class should be zero for correct losses
        return {"image": image, "label": label, "name": self.images[idx]}

    def list_images(self):
        mode = "val" if self.opt.phase == "test" or self.for_metrics else "train"
        path_img = os.path.join(self.opt.dataroot, mode+"_img")
        path_lab = os.path.join(self.opt.dataroot, mode+"_label")
        images = sorted(os.listdir(path_img))
        labels = sorted(os.listdir(path_lab))
        assert len(images)  == len(labels), "different len of images and labels %s - %s" % (len(images), len(labels))
        for i in range(len(images)):
            assert os.path.splitext(images[i])[0] == os.path.splitext(labels[i])[0], '%s and %s are not matching' % (images[i], labels[i])
        if self.opt.perctg_train_data != 1. and mode =='train':
            images = images[:int(len(images) * self.opt.perctg_train_data)]
            labels = labels[:int(len(labels) * self.opt.perctg_train_data)]
        return images, labels, (path_img, path_lab)

    def transforms(self, image, label):
        # assert image.size == label.size
        # resize
        new_width, new_height = (self.opt.load_size, self.opt.load_size)
        image = TR.functional.resize(image, (new_width, new_height), Image.BICUBIC)
        # label = TR.functional.resize(label, (new_width, new_height), Image.NEAREST)
        # label = label.resize((new_width, new_height))
        # crop
        # crop_x = random.randint(0, np.maximum(0, new_width -  self.opt.crop_size))
        # crop_y = random.randint(0, np.maximum(0, new_height - self.opt.crop_size))
        # image = image.crop((crop_x, crop_y, crop_x + self.opt.crop_size, crop_y + self.opt.crop_size))
        # label = label.crop((crop_x, crop_y, crop_x + self.opt.crop_size, crop_y + self.opt.crop_size))
        label = TR.functional.to_tensor(label)
        label = label.permute((1,2,0))
        # print(label.size())
        label = F.interpolate(torch.unsqueeze(label.float(),dim=0), size=(new_width, new_height), mode='nearest')
        label = torch.squeeze(label, dim = 0)
        # flip
        if not (self.opt.phase == "test" or self.opt.no_flip or self.for_metrics):
            if random.random() < 0.5:
                image = TR.functional.hflip(image)
                label = TR.functional.hflip(label)
        # to tensor
        image = TR.functional.to_tensor(image)


        # normalize
        image = TR.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return image, label
