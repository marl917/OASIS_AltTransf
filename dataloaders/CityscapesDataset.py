import random
import torch
from torchvision import transforms as TR
import os
from PIL import Image
import numpy as np

class CityscapesDataset(torch.utils.data.Dataset):
    def __init__(self, opt, for_metrics):
        opt.load_size = opt.size
        # opt.crop_size = 512
        opt.crop_size = opt.size
        opt.label_nc = 34
        opt.contain_dontcare_label = True
        opt.semantic_nc = 35 # label_nc + unknown
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False
        opt.aspect_ratio = 2.0

        self.opt = opt
        self.for_metrics = for_metrics
        self.images, self.labels, self.paths = self.list_images()
        # self.labels = ['aachen/aachen_000000_000019_gtFine_labelIds.png', 'aachen/aachen_000001_000019_gtFine_labelIds.png', 'aachen/aachen_000004_000019_gtFine_labelIds.png', 'aachen/aachen_000006_000019_gtFine_labelIds.png', 'aachen/aachen_000011_000019_gtFine_labelIds.png', 'aachen/aachen_000014_000019_gtFine_labelIds.png', 'aachen/aachen_000015_000019_gtFine_labelIds.png', 'aachen/aachen_000039_000019_gtFine_labelIds.png', 'aachen/aachen_000053_000019_gtFine_labelIds.png', 'aachen/aachen_000075_000019_gtFine_labelIds.png', 'aachen/aachen_000078_000019_gtFine_labelIds.png', 'aachen/aachen_000080_000019_gtFine_labelIds.png', 'bochum/bochum_000000_000313_gtFine_labelIds.png', 'bochum/bochum_000000_000600_gtFine_labelIds.png', 'bochum/bochum_000000_001519_gtFine_labelIds.png']
        if not for_metrics and opt.use_subset:
            print('LOADING SUBSET: ', self.opt.subset_nb)
            name_labels = np.load('../subset_city_fewShot.npy', allow_pickle = True)
            self.labels = list(name_labels[self.opt.subset_nb])
            self.images = [self.labels[i].replace("_gtFine_labelIds.png", "_leftImg8bit.png") for i in range(len(self.labels))]
            
    def __len__(self,):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.paths[0], self.images[idx])).convert('RGB')
        label = Image.open(os.path.join(self.paths[1], self.labels[idx]))
        image, label = self.transforms(image, label)
        # print(torch.min(label), torch.max(label))
        label = label * 255
        # print(torch.min(label), torch.max(label))
        return {"image": image, "label": label, "name": self.images[idx]}

    def list_images(self):
        # mode = "val" if self.opt.phase == "test" or self.for_metrics else "train"
        mode = "val" if self.for_metrics else "train"
        images = []
        path_img = os.path.join(self.opt.dataroot, "leftImg8bit", mode)
        for city_folder in sorted(os.listdir(path_img)):
            cur_folder = os.path.join(path_img, city_folder)
            for item in sorted(os.listdir(cur_folder)):
                images.append(os.path.join(city_folder, item))
        labels = []
        path_lab = os.path.join(self.opt.dataroot, "gtFine", mode)
        for city_folder in sorted(os.listdir(path_lab)):
            cur_folder = os.path.join(path_lab, city_folder)
            for item in sorted(os.listdir(cur_folder)):
                if item.find("labelIds") != -1:
                    labels.append(os.path.join(city_folder, item))

        if self.opt.perctg_train_data != 1. and mode == 'train':
            if self.opt.perctg_train_data == 0.2:
                images = images[int(500*(self.opt.split-1)):int(500 * self.opt.split)]
                labels = labels[int(500*(self.opt.split-1)):int(500 * self.opt.split)]
            else:
                images = images[:int(len(images) * self.opt.perctg_train_data)]
                labels = labels[:int(len(labels) * self.opt.perctg_train_data)]

        assert len(images)  == len(labels), "different len of images and labels %s - %s" % (len(images), len(labels))
        for i in range(len(images)):
            assert images[i].replace("_leftImg8bit.png", "") == labels[i].replace("_gtFine_labelIds.png", ""),\
                '%s and %s are not matching' % (images[i], labels[i])
        return images, labels, (path_img, path_lab)

    def transforms(self, image, label):
        assert image.size == label.size
        # resize
        new_width, new_height = (int(self.opt.load_size / self.opt.aspect_ratio), self.opt.load_size)
        image = TR.functional.resize(image, (new_width, new_height), Image.BICUBIC)
        label = TR.functional.resize(label, (new_width, new_height), Image.NEAREST)
        # flip
        if not (self.opt.phase == "test" or self.opt.no_flip or self.for_metrics):
            if random.random() < 0.5:
                image = TR.functional.hflip(image)
                label = TR.functional.hflip(label)
        # to tensor
        image = TR.functional.to_tensor(image)
        label = TR.functional.to_tensor(label)
        # normalize
        image = TR.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return image, label
