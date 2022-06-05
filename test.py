
import models.models as models
import dataloaders.dataloaders as dataloaders
import utils.utils as utils
import config
import os
from torchvision import utils as t_utils
from tqdm import tqdm
import lpips
from PIL import Image


import torch

import numpy as np
from utils.fid_scores import fid_pytorch

from torchvision import transforms as TR


#--- read options ---#
opt, parser = config.read_arguments_submitit(train=False)
opt = config.read_arguments(opt, parser, train=False)

#--- create dataloader ---#
dataloader_train, dataloader_val = dataloaders.get_dataloaders(opt)



#--- create utils ---#
image_saver = utils.results_saver(opt)

img_saverOnebyOne = utils.image_saver(opt)



#--- create models ---#
if opt.disc_out:
    #TO CREATE CONF MAT
    opt.target_semantic_nc = opt.semantic_nc
    #if source dataset is cocostuff semantic_nc = 183, if ade20k semantic_nc = 151
    if 'coco' in opt.name:
        opt.semantic_nc = 183
    elif 'ade20k' in opt.name:
        opt.semantic_nc = 151
    opt.z_dim = 64
# opt.semantic_nc = 151
    
    
    
model = models.OASIS_model(opt)
model = models.put_on_multi_gpus(model, opt)
model.eval()

#--- iterate over validation set ---#
path = os.path.join(opt.results_dir, opt.name, opt.ckpt_iter)
path_image = os.path.join(path, "image")


def preprocess_label(opt, label, semantic_nc=35):
    label = label.long()
    label_map= label.cuda()

    bs, _, h, w = label_map.size()
    nc = semantic_nc
    if opt.gpu_ids != "-1":
        input_label = torch.cuda.FloatTensor(bs, nc, h, w).zero_()
    else:
        input_label = torch.FloatTensor(bs, nc, h, w).zero_()
    # print('before scatter')
    input_semantics = input_label.scatter_(1, label_map, 1.0)
    # print('after scatter')
    return input_semantics



if opt.gen_img:
    for i, data_i in enumerate(tqdm(dataloader_val, mininterval=1)):
        image, label = models.preprocess_input(opt, data_i)
        generated = model(None, label, "generate", None)
        image_saver(label, generated, data_i["name"])


# if not opt.fid:
#
#     global_avg_dist = []
#     loss_fn = lpips.LPIPS(net='alex', version='0.1')
#     loss_fn.cuda()
#
#     for i, data_i in enumerate(tqdm(dataloader_val, mininterval=1)):
#         image, label = models.preprocess_input(opt, data_i)
#         # generated = model(None, label, "generate", None)
#         # image_saver(label, generated, data_i["name"])
#
#         for i in range(len(label)):
#             repeat_l = torch.stack((label[i], label[i]) * 10, dim=0)
#             generated = model(None, repeat_l, "generate", None)
#
#             dists = []
#             for i, img0 in enumerate(generated):
#                 img1_list = generated[i + 1:]
#                 for img1 in img1_list:
#                     dist01 = loss_fn.forward(img0.cuda(), img1.cuda())
#                     dists.append(dist01.item())
#
#             avg_dist = np.mean(np.array(dists))
#             global_avg_dist.append(avg_dist)
#             # print(avg_dist)
#     glob_avg = np.mean(np.array(global_avg_dist))
#     print(glob_avg)
#

        # t_utils.save_image(
        #     generated,
        #     os.path.join(path_image, '0.jpg'),
        #     nrow=8,
        #     normalize=True,
        #     range=(-1, 1),
        # )
        # generated = model(None, label, "generate", None)

        # t_utils.save_image(
        #     generatedw,
        #     os.path.join(path_image, '1.jpg'),
        #     nrow=8,
        #     normalize=True,
        #     range=(-1, 1),
        # )

if opt.fid:
    fid_computer = fid_pytorch(opt, dataloader_val)
    # val_fid = []
    # for i in range(5):
    is_best, cur_fid = fid_computer.update(model, -1)
    # val_fid.append(cur_fid)
    print(cur_fid)
    # print(np.mean(val_fid), np.var(val_fid))

elif opt.kid:
    fid_computer = fid_pytorch(opt, dataloader_val)
    val_kid = []
    for i in range(5):
        is_best, cur_kid = fid_computer.update_kid(model, 10000000)
        val_kid.append(cur_kid)
        print(cur_kid)
    print(np.mean(val_kid), np.var(val_kid))


    # for img, n in zip(generated, data_i['name']):
    #     # print(os.path.basename(n))
    #     t_utils.save_image(
    #         img,
    #         os.path.join(path_image, n.split("/")[-1]).replace('.jpg', '.png'),
    #         nrow=4,
    #         normalize=True,
    #         range=(-1, 1),
    #     )


    # generated_seg = model(image, None, 'disc_otp', None)
    # # print(output_D_max.size(), torch.min(output_D_max), torch.max(output_D_max))
    # # print(torch.histc(output_D_max, bins=36, min=0, max=35))
    # img_saverOnebyOne .save_images(generated_seg[:, 1:, :, :], 'fakeseg.png',  f'off_{i}', is_label= True)
    # img_saverOnebyOne.save_images(label, 'realseg.png', f'off_{i}', is_label=True)
    # img_saverOnebyOne.save_images(image, 'realimg.png', f'off_{i}', is_label=False)
    # # save_images(generated_seg[:, 1:, :, :], i, args.save_dir, is_label=True, semantic_nc=args.semantic_nc)
    # # save_images(label, f'real{i}', args.save_dir, is_label=True, semantic_nc=args.semantic_nc)
    # # save_images(input, f'image_{i}', args.save_dir)

elif opt.creative_gen:
    new_width, new_height = (256, 256)

    label_path = '/checkpoint/marlenec/cocostuff/train_label'
    label = Image.open(os.path.join(label_path, '000000002066.png'))
    label = TR.functional.resize(label, (new_width, new_height), Image.NEAREST)
    label = TR.functional.to_tensor(label) * 255 + 1
    print(torch.unique(label))

    lab_cop = torch.zeros_like(label)
    # print(any(lab_cop == 25.))
    lab_cop[label == 25.] = 0
    print('unique lab cop :', torch.unique(lab_cop))
    lab_cop[label != 25.] = 1
    print('unique lab cop :', torch.unique(lab_cop))

    lab_cop_add = label.detach()
    lab_cop_add[label != 25.] = 0

    label = Image.open(os.path.join(label_path, '000000000382.png'))
    label = TR.functional.resize(label, (new_width, new_height), Image.NEAREST)
    label = TR.functional.to_tensor(label) * 255 + 1

    print(label.size(),lab_cop.size(), lab_cop_add.size())
    print(torch.unique(label), torch.unique(lab_cop), torch.unique(lab_cop_add))

    label = label * lab_cop +lab_cop_add
    label[label == 256] = 0
    label = preprocess_label(opt, torch.unsqueeze(label, dim=0))
    print(label.size())
    #torch.cat((label, label, label, label), dim=0)
    generated = model(None, label, "generate", None)

    t_utils.save_image(
        generated,
        '0.jpg',
        nrow=1,
        normalize=True,
        range=(-1, 1),
    )

elif opt.class_mixing:
    new_width, new_height = (256, 256)
    label_path = '/checkpoint/marlenec/cocostuff/train_label'
    label = Image.open(os.path.join(label_path, '000000002066.png'))
    label = TR.functional.resize(label, (new_width, new_height), Image.NEAREST)
    label = TR.functional.to_tensor(label) * 255 + 1
    label[label == 256] = 0
    print(torch.unique(label))

    lab_cop = torch.zeros_like(label)
    # print(any(lab_cop == 25.))
    lab_cop[label == 25.] = 0
    print('unique lab cop :', torch.unique(lab_cop))
    lab_cop[label != 25.] = 1
    print('unique lab cop :', torch.unique(lab_cop))

    label_coco = preprocess_label(opt, torch.unsqueeze(label, dim=0), 183)
    print('in test.py :', torch.unique(torch.argmax(label_coco, dim=1)))

    label_city_path = '/datasets01/cityscapes/112817/gtFine/train/aachen/aachen_000110_000019_gtFine_labelIds.png'
    label = Image.open(label_city_path)
    label = TR.functional.resize(label, (new_width, new_height), Image.NEAREST)
    label = TR.functional.to_tensor(label) * 255
    print(torch.unique(label))
    label_cityscapes = preprocess_label(opt, torch.unsqueeze(label, dim=0))

    # generated = model(None, label_cityscapes, "generate", None, seg2=[label_coco, lab_cop])
    generated = model(None, label_coco, "generate", None)

    t_utils.save_image(
        generated,
        '0.jpg',
        nrow=1,
        normalize=True,
        range=(-1, 1),
    )



#to create conf mat
elif opt.disc_out:
    print('target semantic nc is: ', opt.target_semantic_nc)
    print('source semantic nc is: ', opt.semantic_nc)
    os.makedirs(f'conf_mat/target_{opt.dataset_mode}_256x512', exist_ok = True)
    for i, data_i in enumerate(tqdm(dataloader_train, mininterval=1)):
        print(i)
        image, label = models.preprocess_input(opt, data_i)
        clas = model(image, label, "disc_out", None)
        class_coco = torch.argmax(clas[:,1:,:,:], dim=1)
        # image_saver(label, generated, data_i["name"])
        class_city = torch.argmax(label, dim=1)
        # print(label.size(), clas.size(), class_city.size(), class_coco.size(),torch.max(class_city), torch.max(class_coco))
        for j in range(opt.target_semantic_nc):
            if i==0:
                if j==0:
                    conf_mat = []
                conf_mat.append(torch.bincount(class_coco[class_city == j].view(-1), minlength = opt.semantic_nc))
            else:
                conf_mat[j] += torch.bincount(class_coco[class_city == j].view(-1), minlength=opt.semantic_nc)
        # break
    for i in range(opt.target_semantic_nc):
        conf_mat[i] = conf_mat[i] / torch.sum(conf_mat[i])
    conf_mat = torch.stack(conf_mat, dim=0)

    label_map = torch.unsqueeze(torch.argmax(conf_mat, dim=1), dim=1)
    input_label = torch.cuda.FloatTensor(opt.target_semantic_nc, opt.semantic_nc).zero_()
    input_semantics = input_label.scatter_(1, label_map, 1.0)
    print(input_semantics.argmax(-1))
    #     torch.save(input_semantics, f'bin_{file}')
    # print('size of confmat: ', conf_mat.size())
    # print(torch.argsort(conf_mat[7]))
    # torch.save(conf_mat, f'conf_mat_sourceAde20K128{opt.dataset_mode}_fewShottest_subset_Nb{opt.subset_nb}.pt')
    # torch.save(conf_mat, f'conf_mat/conf_mat_sourceAde20K128{opt.dataset_mode}_all2975Data.pt')

    torch.save(conf_mat, f'conf_mat/target_{opt.dataset_mode}_256x512/conf_mat_sourceCOCO_subsetNb{opt.subset_nb}.pt')
    torch.save(input_semantics,
               f'conf_mat/target_{opt.dataset_mode}_256x512/bin_conf_mat_sourceCOCO_subsetNb{opt.subset_nb}.pt')












