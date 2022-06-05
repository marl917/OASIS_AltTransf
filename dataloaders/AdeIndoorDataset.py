import random
import torch
from torchvision import transforms as TR
import os
from PIL import Image
import numpy as np

class AdeIndoorDataset(torch.utils.data.Dataset):
    def __init__(self, args, for_metrics):
        args.label_nc = 95
        args.contain_dontcare_label = True
        args.aspect_ratio = 1
        args.semantic_nc = 95
        self.args = args

        if for_metrics:
            args.load_size = 256
        else:
            args.load_size = 286
        # args.crop_size = args.size
        args.crop_size = args.size

        # opt.crop_size = opt.load_size
        # opt.label_nc = 34
        # opt.contain_dontcare_label = True
        # opt.semantic_nc = 35 # label_nc + unknown
        # opt.cache_filelist_read = False
        # opt.cache_filelist_write = False
        # opt.aspect_ratio = 2.0
        #
        # self.opt = opt

        self.for_metrics = for_metrics


        self.images, self.labels, self.paths = self.list_images()

        print(f'Loading images of size [{int(args.crop_size)}, {args.crop_size * args.aspect_ratio}]')

    def __len__(self,):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.paths[0], self.images[idx])).convert('RGB')
        label = Image.open(os.path.join(self.paths[1], self.labels[idx]))
        image, label = self.transforms(image, label)
        label = label * 255
        return {"image": image, "label": label, "name": self.images[idx]}

    def list_images(self):
        mode = "val" if self.for_metrics else "train"

        rootPath = '/checkpoint/marlenec/ADE_indoor'
        images = []
        path_img = os.path.join(rootPath, "ADE15c_indoor_img", mode)
        for city_folder in sorted(os.listdir(path_img)):
            cur_folder = os.path.join(path_img, city_folder)
            for item in sorted(os.listdir(cur_folder)):
                if '.jpg' in item:
                    images.append(os.path.join(city_folder, item))
        labels = []
        path_lab = os.path.join(rootPath, "ADE15c_indoor_lbl", mode)
        print(f'Directory of label path for {mode} is {path_lab}')
        for city_folder in sorted(os.listdir(path_lab)):
            cur_folder = os.path.join(path_lab, city_folder)
            for item in sorted(os.listdir(cur_folder)):
                if '.png' in item:
                    labels.append(os.path.join(city_folder, item))

        if self.args.perctg_train_data != 1. and mode =='train':
            images = images[:int(len(images) * self.args.perctg_train_data)]
            labels = labels[:int(len(labels) * self.args.perctg_train_data)]



        assert len(images)  == len(labels), "different len of images and labels %s - %s" % (len(images), len(labels))
        for i in range(len(images)):
            assert images[i].replace(".jpg", "") == labels[i].replace(".png", ""),\
                '%s and %s are not matching' % (images[i], labels[i])
        return images, labels, (path_img, path_lab)

    def transforms(self, image, label):
        # print(image.size, label.size)
        # assert image.size == label.size
        # resize
        new_width, new_height = (self.args.load_size, self.args.load_size)
        image = TR.functional.resize(image, (new_width, new_height), Image.BICUBIC)
        label = TR.functional.resize(label, (new_width, new_height), Image.NEAREST)
        # crop
        crop_x = random.randint(0, np.maximum(0, new_width - self.args.crop_size))
        crop_y = random.randint(0, np.maximum(0, new_height - self.args.crop_size))
        image = image.crop((crop_x, crop_y, crop_x + self.args.crop_size, crop_y + self.args.crop_size))
        label = label.crop((crop_x, crop_y, crop_x + self.args.crop_size, crop_y + self.args.crop_size))
        # flip
        if not self.for_metrics:
            if random.random() < 0.5:
                image = TR.functional.hflip(image)
                label = TR.functional.hflip(label)
        # to tensor
        image = TR.functional.to_tensor(image)
        label = TR.functional.to_tensor(label)
        # normalize
        image = TR.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return image, label
