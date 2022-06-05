from models.sync_batchnorm import DataParallelWithCallback
import models.generator as generators
import models.discriminator as discriminators
import os
import copy
import torch
import torch.nn as nn
from torch.nn import init
import models.losses as losses
# from augment import AdaptiveAugment, augment
from torchvision import transforms, utils

class OASIS_model(nn.Module):
    def __init__(self, opt):
        super(OASIS_model, self).__init__()
        self.opt = opt
        #--- generator and discriminator ---
        if self.opt.conf_mat != None:
            opt.conf_matrix = torch.load(opt.conf_mat, map_location='cpu').cuda()
            opt.conf_matrix.requires_grad = True
            print('is conf mat with required grad true:', opt.conf_matrix.requires_grad)
        self.netG = generators.OASIS_Generator(opt)

        # if opt.phase == "train":
        self.netD = discriminators.OASIS_Discriminator(opt)

        self.print_parameter_count()
        self.init_networks()
        #--- EMA of generator weights ---
        with torch.no_grad():
            self.netEMA = copy.deepcopy(self.netG) if not opt.no_EMA else None

        #--- used in case of transfer learning ---
        if opt.phase == "train":
            if self.opt.train_index_gabeta != '':
                self.index_layers_gabe = self.opt.train_index_gabeta.split(",")
                self.matches = list(map(lambda x: 'body.' + x, self.index_layers_gabe))
            else:
                self.index_layers_gabe = None

        #--- load previous checkpoints if needed ---
        self.load_checkpoints()

        # print('Number of generator param :', sum(p.numel() for p in self.netG.parameters() if p.requires_grad))

        ## transfer alt 2: train mlpshared/conv_s and all disc
        if opt.phase == "train":
            if opt.transfer_l!='':
                if not opt.trainG:  #transfer_alt==0 for training all
                    for name, param in self.netG.named_parameters():
                        if 'mlp_shared' in name or 'fc' in name and opt.transfer_alt_g ==0:# and opt.transfer_alt!=1: #and opt.transfer_alt = 5:
                            param.requires_grad = True
                        # elif 'fc' in name:
                        #     param.requires_grad = True
                        elif 'mat_conf' in name:
                            param.requires_grad = True
                        elif 'linear_miner' in name:
                            param.requires_grad = True
                        elif 'similarity' in name:
                            param.requires_grad = True

                        # elif 'mlp_shared_new' in name or 'embedding' in name:     #################### RETRAIN Batchnorm affine parameters
                        # # if 'mlp' in name or 'fc.weight' in name:
                        #     print(name, 'grad True')
                        #     param.requires_grad = True
                            # print('contain mlp_shared : ', name
                        # elif opt.transfer_alt in [2,3,4,5,6] and 'conv_s' in name:  # train skip connection conv
                        #     param.requires_grad = True
                        # elif opt.transfer_alt == 4 and 'conv_img' in name:
                        #     param.requires_grad = True
                        # elif self.opt.train_index_gabeta!='' and 'mlp' in name and any(x in name for x in self.matches):
                        #     param.requires_grad = True
                            # print('contain mlp : ', name)
                        else:
                            # print('grad=False :', name)
                            param.requires_grad = False
                    # print('Number of generator param after freezing :', sum(p.numel() for p in self.netG.parameters() if p.requires_grad))

                print('Number of gen param :', sum(p.numel() for p in self.netG.parameters() if p.requires_grad))

                if not opt.trainD:
                    for name, param in self.netD.named_parameters():
                        # print('NAME: ', name)
                        if 'layer_up_last' in name:
                            param.requires_grad = True
                        elif 'mat_conf' in name:
                            param.requires_grad = not opt.notConfMatDisc
                        elif opt.transfer_alt_d in [3,5] and int(name.split('.')[1]) > opt.freezeD and name.split('.')[0]=='body_down':
                            param.requires_grad = True
                        elif opt.transfer_alt_d in [3,5] and name.split('.')[0]=='body_up' and int(name.split('.')[1]) >= opt.freezeD-5:
                            param.requires_grad = True
                        elif opt.transfer_alt_d in [3,5] and name.split('.')[0] != 'body_down' and name.split('.')[0] != 'body_up':
                            param.requires_grad = True
                        else:
                            param.requires_grad = False

                # print('Number of disc param after', sum(p.numel() for p in self.netD.parameters() if p.requires_grad))
                print('Layers in disc with param grad = True')
                for name, param in self.netD.named_parameters():
                    if param.requires_grad == True:
                        print(name)
                    # if 'fc' in name or 'first_norm' in name:
                    #     print('name', param.requires_grad)

                print('Layers in gen with param grad = True')
                for name, param in self.netG.named_parameters():
                    if 'mat_conf' in name:
                        print("MAT CONF IS IN PARAM -----------------------: ", param.requires_grad)
                    if param.requires_grad == True:
                        print('grad=True', name)

        #--- perceptual loss ---#
        if opt.phase == "train":
            if opt.add_vgg_loss:
                self.VGG_loss = losses.VGGLoss(self.opt.gpu_ids)
            if opt.augment:
                self.ada_aug_p = opt.ada_aug_p
                self.ada_augment = AdaptiveAugment(0.6, 500 * 1000, 8, 'cuda')

    def forward(self, image, label, mode, losses_computer):
        # Branching is applied to be compatible with DataParallel

        if mode == "losses_G":       
            loss_G = 0
            # print('before netG')
            fake = self.netG(label)
            if self.opt.augment:
                # print('augment')
                fake_a, seg_a, _ = augment(fake, self.ada_aug_p, seg=label)
                dict_data = {'image': image, 'label': torch.unsqueeze(torch.argmax(seg_a, dim=1), dim=1)}
                i, label_a = preprocess_input(self.opt, dict_data)
            else:
                fake_a = fake
                label_a = label
            # if self.opt.transfer_alt == 1:
            #     loss_G += (l1loss + l2loss) * self.opt.lambda_l1loss
            # print('after netG', l2loss)
            output_D = self.netD(fake_a)
            # print('after netD')
            # print('size of fake and outputD:', fake.size(), output_D.size(), label.size())
            loss_G_adv = losses_computer.loss(output_D, label_a, for_real=True)
            loss_G += loss_G_adv
            if self.opt.add_vgg_loss:
                loss_G_vgg = self.opt.lambda_vgg * self.VGG_loss(fake, image)
                loss_G += loss_G_vgg
            else:
                loss_G_vgg = None
            return loss_G, [loss_G_adv, loss_G_vgg]

        if mode == "losses_D":
            if self.opt.augment:
                image_a, seg_a, _ = augment(image, self.ada_aug_p, seg=label)
                dict_data = {'image': image, 'label': torch.unsqueeze(torch.argmax(seg_a, dim=1), dim=1)}
                i, label_a = preprocess_input(self.opt, dict_data)
            else:
                image_a = image
                label_a = label

            # if self.opt.augment:
            #     image_a, _ = augment(image, self.ada_aug_p)
            # else:
            #     image_a = image
            loss_D = 0

            with torch.no_grad():
                fake = self.netG(label)
            if self.opt.augment:
                fake_a, _, _ = augment(fake, self.ada_aug_p)
            else:
                fake_a = fake


            output_D_fake = self.netD(fake_a)
            loss_D_fake = losses_computer.loss(output_D_fake, label_a, for_real=False)
            loss_D += loss_D_fake
            output_D_real = self.netD(image_a)
            loss_D_real = losses_computer.loss(output_D_real, label_a, for_real=True)
            loss_D += loss_D_real
            if not self.opt.no_labelmix:
                mixed_inp, mask = generate_labelmix(label, fake, image)
                output_D_mixed = self.netD(mixed_inp)
                loss_D_lm = self.opt.lambda_labelmix * losses_computer.loss_labelmix(mask, output_D_mixed, output_D_fake,
                                                                                output_D_real)
                loss_D += loss_D_lm
            else:
                loss_D_lm = None
            # if self.opt.augment:
            #     # label_aug = torch.argmax(label, dim=1) + 1
            #     disc_lab = torch.argmax(output_D_real, dim=1)
            #     true_class = (disc_lab!=0).type(torch.int) * 2 -1
            #     # print(torch.min(true_class), torch.max(true_class))
            #     val = torch.sum(true_class, dim=[1,2])
            #     # for_ada_tens = torch.mean(output_D_real[:,1:,:,:]*label, dim=[1,2,3])
            #     # print('for ada tens val :', for_ada_tens)
            #     # if for_ada_tens.device == 'cuda:0':
            #     # print(val.size())
            #     self.ada_aug_p = self.ada_augment.tune(val)
            #     self.r_t_stat = self.ada_augment.r_t_stat
            return loss_D, [loss_D_fake, loss_D_real, loss_D_lm]

        if mode == "generate":
            with torch.no_grad():
                if self.opt.no_EMA:
                    fake = self.netG(label)
                else:
                    # print('before netEMA')
                    fake = self.netEMA(label)
            return fake

        if mode == "disc_out":
            with torch.no_grad():
                seg = self.netD(image)
            return seg

        # print('Layers in disc with param grad = True')
        # for name, param in self.netD.named_parameters():
        #     if param.requires_grad == True:
        #         print(name)

    def load_checkpoints(self):
        if self.opt.phase == "test":
            which_iter = self.opt.ckpt_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            print('loading :', path)
            if self.opt.no_EMA:
                # print('loading G')
                self.netG.load_state_dict(torch.load(path + "G.pth"), strict=False)
            else:
                self.netEMA.load_state_dict(torch.load(path + "EMA.pth"), strict=False)
                print('loading EMA')
            # print('NOT loading disc')
            # if self.opt.ckpt_disc!=None:
            # self.netD.load_state_dict(torch.load(self.opt.ckpt_disc))
            print('LOADING DISC WITH PATH: ', path + "D.pth")
            self.netD.load_state_dict(torch.load(path + "D.pth"))
        elif self.opt.continue_train:
            which_iter = self.opt.which_iter
            print('OPT.CONTINUE TRAIN LOAD LCKPT :', which_iter)
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            a = torch.load(path + "G.pth")
            b = torch.load(path + "EMA.pth")
            if which_iter == 'best':
                for n in a.copy().keys():
                    if 'mlp_shared' in n:
                        print('Deleting mlp')
                        del a[n]
                        del b[n]
                    elif 'fc' in n:
                        del a[n]
                        del b[n]

            self.netG.load_state_dict(a, strict=False)
            self.netD.load_state_dict(torch.load(path + "D.pth"))
            if not self.opt.no_EMA:
                self.netEMA.load_state_dict(b, strict=False)
        elif self.opt.transfer_l != '':
            path = str(self.opt.transfer_l)

            a = torch.load(path + "/best_G.pth")
            b = torch.load(path + "/best_EMA.pth")
            # del a['body.0.norm_0.mlp_shared.0.weight']

            for n in a.copy().keys():
                if 'mlp_shared' in n and (self.opt.conf_mat==None or self.opt.notConfMatGen):
                    print('Deleting mlp')
                    del a[n]
                    del b[n]
                elif 'fc' in n and (self.opt.conf_mat==None or self.opt.notConfMatGen):
                # if 'mlp' in n or 'fc.weight' in n:
                    del a[n]
                    del b[n]
                elif self.index_layers_gabe!=None:
                    if 'mlp' in n and any(x in n for x in self.matches):
                        # print(n)
                        del a[n]
                        del b[n]
            # self.netG.load_state_dict(torch.load(path + "/best_G.pth"), strict = False)
            ################################################################################# Loading pretrained netEMA for G instead of pretrained G
            print('LOADING NET EMA for initialization of G')
            self.netG.load_state_dict(b, strict=False)

            if not self.opt.no_EMA:
                # self.netEMA.load_state_dict(torch.load(path + "/best_EMA.pth"), strict =False)
                self.netEMA.load_state_dict(b, strict=False)
                print('Loaded pretrained EMA')
            del a
            del b

            a = torch.load(path + "/best_D.pth")
            for n in a.copy().keys():
                if 'layer_up_last' in n and (self.opt.conf_mat==None or self.opt.notConfMatGen):
                     del a[n]
            # self.netD.load_state_dict(torch.load(path + "/best_D.pth"), strict = False)
            self.netD.load_state_dict(a, strict=False)
            del a

            print('LOADED pretrained models for transfer learning')

    def print_parameter_count(self):
        if self.opt.phase == "train":
            networks = [self.netG, self.netD]
        else:
            networks = [self.netG]
        for network in networks:
            param_count = 0
            for name, module in network.named_modules():
                if (isinstance(module, nn.Conv2d)
                        or isinstance(module, nn.Linear)
                        or isinstance(module, nn.Embedding)):
                    param_count += sum([p.data.nelement() for p in module.parameters()])
            print('Created', network.__class__.__name__, "with %d parameters" % param_count)

    def print_parameter_count_oneModel(self, net):
        network = net
        param_count = 0
        for name, module in network.named_modules():
            if (isinstance(module, nn.Conv2d)
                    or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                param_count += sum([p.data.nelement() for p in module.parameters()])
        print(network.__class__.__name__, "with %d parameters" % param_count)

    def init_networks(self):
        def init_weights(m, gain=0.02):
            classname = m.__class__.__name__
            # print('in init: ', classname, m.__class__, m)
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                # print('in channels: ', m.in_channels)
                # if m.in_channels==99 and self.opt.transfer_alt==6:   ###################################################################### nb of ipt channels of mlp_shared_target because we init by 0
                #     init.constant_(m.weight.data, 0.0)
                # else:
                init.xavier_normal_(m.weight.data, gain=gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        if self.opt.phase == "train":
            networks = [self.netG, self.netD]
        else:
            networks = [self.netG]
        for net in networks:
            net.apply(init_weights)


def put_on_multi_gpus(model, opt):
    if opt.gpu_ids != "-1":
        gpus = list(map(int, opt.gpu_ids.split(",")))
        model = DataParallelWithCallback(model, device_ids=gpus).cuda()
    else:
        model.module = model
    assert len(opt.gpu_ids.split(",")) == 0 or opt.batch_size % len(opt.gpu_ids.split(",")) == 0
    return model


def preprocess_input(opt, data):
    # print('start preprocessing input')
    data['label'] = data['label'].long()
    if opt.gpu_ids != "-1":
        data['label'] = data['label'].cuda()
        data['image'] = data['image'].cuda()
    label_map = data['label']
    bs, _, h, w = label_map.size()
    nc = opt.semantic_nc
    if opt.gpu_ids != "-1":
        input_label = torch.cuda.FloatTensor(bs, nc, h, w).zero_()
    else:
        input_label = torch.FloatTensor(bs, nc, h, w).zero_()
    # print('before scatter')
    input_semantics = input_label.scatter_(1, label_map, 1.0)
    # print('after scatter')
    return data['image'], input_semantics


def generate_labelmix(label, fake_image, real_image):
    target_map = torch.argmax(label, dim = 1, keepdim = True)
    all_classes = torch.unique(target_map)
    for c in all_classes:
        target_map[target_map == c] = torch.randint(0,2,(1,)).cuda()
    target_map = target_map.float()
    mixed_image = target_map*real_image+(1-target_map)*fake_image
    return mixed_image, target_map
