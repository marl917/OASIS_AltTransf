import random
import torch
from torchvision import transforms as TR
import os
from PIL import Image

class CelebAHQDataset(torch.utils.data.Dataset):
    def __init__(self, opt, for_metrics, ref=False):
        opt.load_size = 256
        # opt.crop_size = 256
        opt.crop_size = opt.size
        opt.label_nc = 19
        opt.contain_dontcare_label = False
        opt.semantic_nc = 19 # label_nc + unknown
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False
        opt.aspect_ratio = 1.0
        self.ref =ref
        self.opt = opt
        self.for_metrics = for_metrics

        self.images, self.labels = self.list_images()

    def __len__(self,):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = Image.open(self.labels[idx])
        image, label = self.transforms(image, label)
        label = label * 255
        return {"image": image, "label": label, "name": os.path.basename(os.path.splitext(self.labels[idx])[0])}

    def list_images(self):
        mode = "val" if self.for_metrics or self.opt.phase == "test" else "train"
        images = []
        labels = []
        if mode == 'train':
            print("type of training :", os.path.basename(os.path.splitext(self.opt.dataroot)[0]))
            if os.path.basename(os.path.splitext(self.opt.dataroot)[0])== 'fewShotSegReal':
                path_img_train = os.path.join('/checkpoint/marlenec/CelebAMask-HQ', "train_img")
            else:
                path_img_train = os.path.join(self.opt.dataroot, "train_img")
            # path_val_train = os.path.join(self.opt.dataroot, "val_img")

            for item in sorted(os.listdir(path_img_train)):
                if '.jpg' in item:
                    images.append(os.path.join(path_img_train, item))

            # for item in sorted(os.listdir(path_val_train)):
            #     if '.jpg' in item:
            #         images.append(os.path.join(path_val_train, item))

            path_lab_train = os.path.join(self.opt.dataroot, "train_label")
            # path_lab_eval = os.path.join(self.opt.dataroot, "val_label")

            for item in sorted(os.listdir(path_lab_train)):
                if '.png' in item:
                    labels.append(os.path.join(path_lab_train, item))
            # for item in sorted(os.listdir(path_lab_eval)):
            #     if '.png' in item:
            #         labels.append(os.path.join(path_lab_eval, item))
            #in case use only a subset of training images
            if self.opt.perctg_train_data!=1.:
                images = images[:int(len(images)*self.opt.perctg_train_data)]
                labels = labels[:int(len(labels)*self.opt.perctg_train_data)]
        elif mode == 'val':
            val_dataroot = '/checkpoint/marlenec/CelebAMask-HQ/'   #HARDCODED
            if self.ref :
                path_img_test = os.path.join(val_dataroot, "val_img")
                for item in sorted(os.listdir(path_img_test)):
                    if '.jpg' in item:
                        images.append(os.path.join(path_img_test, item))
                path_lab_test = os.path.join(val_dataroot, "val_label")
                for item in sorted(os.listdir(path_lab_test)):
                    if '.png' in item:
                        labels.append(os.path.join(path_lab_test, item))
                images = images[:2824]
                labels = labels[:2824]
                print('Size of ref img :', len(images))
            else:
                path_img_test = os.path.join(val_dataroot, "test_img")
                for item in sorted(os.listdir(path_img_test)):
                    if '.jpg' in item:
                        images.append(os.path.join(path_img_test, item))
                # images = images[:500]
                path_lab_test = os.path.join(val_dataroot, "test_label")
                for item in sorted(os.listdir(path_lab_test)):
                    if '.png' in item:
                        labels.append(os.path.join(path_lab_test, item))
                print('Size of test img :', len(images))
        # elif mode =='test':
        #     path_img_test = os.path.join(self.opt.dataroot, "test_img")
        #     for item in sorted(os.listdir(path_img_test)):
        #         if '.jpg' in item:
        #             images.append(os.path.join(path_img_test, item))
        #     path_lab_test = os.path.join(self.opt.dataroot, "test_label")
        #     for item in sorted(os.listdir(path_lab_test)):
        #         if '.png' in item:
        #             labels.append(os.path.join(path_lab_test, item))

        assert len(images)  == len(labels), "different len of images and labels %s - %s" % (len(images), len(labels))
        for i in range(len(images)):
            # print('in assert for loop ', os.path.basename(os.path.splitext(images[i])[0]))
            assert os.path.basename(os.path.splitext(images[i])[0]) == os.path.basename(os.path.splitext(labels[i])[0]),\
                '%s and %s are not matching' % (images[i], labels[i])

        print('size of image training dataset :', len(images))
        return images, labels

    def transforms(self, image, label):
        # print(image.size, label.size)
        # assert image.size == label.size
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

        if label.size(0)==3:   # because output from repurpose gan with few shot model load like images...
            label = torch.unsqueeze(label[0,:,:],0)

        return image, label
