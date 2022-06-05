import random
import torch
from torchvision import transforms as TR
import os
from PIL import Image
import numpy as np


class FacadesDataset(torch.utils.data.Dataset):
    def __init__(self, opt, for_metrics):
        if opt.phase == "test" or for_metrics:
            opt.load_size = 256
        else:
            opt.load_size = 286
        # opt.crop_size = 256
        opt.crop_size = opt.size
        opt.label_nc = 12
        opt.contain_dontcare_label = False
        opt.semantic_nc = 12 # label_nc + unknown
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False
        opt.aspect_ratio = 1.0


        self.opt = opt
        self.for_metrics = for_metrics
        self.images, self.labels, self.path_img = self.list_images()

    def __len__(self,):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.path_img, self.images[idx])).convert('RGB')
        label = Image.open(os.path.join(self.path_img, self.labels[idx]))
        image, label = self.transforms(image, label)
        label = label * 256.4102564 - 1
        label = label.to(torch.int)
        # print(torch.unique(label))

        return {"image": image, "label": label, "name": self.images[idx]}

    def list_images(self):
        # self.opt.dataroot = '/checkpoint/marlenec/datasets/facades/facades_cmp'
        # mode = "val" if self.opt.phase == "test" or self.for_metrics else "train"
        # path_img = os.path.join(self.opt.dataroot,  mode)
        # file_list = os.listdir(path_img)
        # img_list = [filename for filename in file_list if ".jpg" in filename]
        # images = sorted(img_list)
        #
        #
        #
        # label_list = [filename for filename in file_list if ".png" in filename ]
        # labels = sorted(label_list)
        #
        # if self.opt.perctg_train_data != 1. and mode =='train':
        #     images = images[:int(len(images) * self.opt.perctg_train_data)]
        #     labels = labels[:int(len(labels) * self.opt.perctg_train_data)]

        self.opt.dataroot = '/checkpoint/marlenec/datasets/facades/facades_cmp/mixvaltrain'
        mode = "val" if self.for_metrics else "train"
        # path_img = os.path.join(self.opt.dataroot, mode)
        path_img = self.opt.dataroot
        file_list = os.listdir(path_img)
        img_list = [filename for filename in file_list if ".jpg" in filename]
        images = sorted(img_list)

        label_list = [filename for filename in file_list if ".png" in filename]
        labels = sorted(label_list)

        # if self.opt.perctg_train_data != 1. and mode == 'train':
        #     images = images[:int(len(images) * self.opt.perctg_train_data)]
        #     labels = labels[:int(len(labels) * self.opt.perctg_train_data)]

        # if self.opt.perctg_train_data != 1. and mode == 'train':
        # if mode == 'train':
        #     images = images[int(100*(self.opt.perctg_train_data-1)):int(100 * self.opt.perctg_train_data)]
        #     labels = labels[int(100*(self.opt.perctg_train_data-1)):int(100 * self.opt.perctg_train_data)]
        # else:
        #     # print('size of all images', len(images))
        #     images = images[:int(100*(self.opt.perctg_train_data-1))] + images[int(100 * self.opt.perctg_train_data):]
        #     labels = labels[:int(100*(self.opt.perctg_train_data-1))] + labels[int(100 * self.opt.perctg_train_data):]
        #     # print('size of val images :', len(images), int(100*(self.opt.perctg_train_data-1)), int(100 * self.opt.perctg_train_data))


        if mode == 'train':
            # images = images[int(100*(self.opt.perctg_train_data-1)):int(100 * self.opt.perctg_train_data)]
            images = images[:int(len(images) * self.opt.perctg_train_data)]
            labels = labels[:int(len(labels) * self.opt.perctg_train_data)]
        else:
            # print('size of all images', len(images))
            images = images[int(len(images) * self.opt.perctg_train_data):]
            labels = labels[int(len(labels) * self.opt.perctg_train_data):]

        return images,labels, path_img

    def transforms(self, image, label):
        # print(image.size, label.size)
        assert image.size == label.size
        # resize
        new_width, new_height = (self.opt.load_size, self.opt.load_size)
        image = TR.functional.resize(image, (new_width, new_height), Image.BICUBIC)
        label = TR.functional.resize(label, (new_width, new_height), Image.NEAREST)
        # crop
        crop_x = random.randint(0, np.maximum(0, new_width -  self.opt.crop_size))
        crop_y = random.randint(0, np.maximum(0, new_height - self.opt.crop_size))
        image = image.crop((crop_x, crop_y, crop_x + self.opt.crop_size, crop_y + self.opt.crop_size))
        label = label.crop((crop_x, crop_y, crop_x + self.opt.crop_size, crop_y + self.opt.crop_size))
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
