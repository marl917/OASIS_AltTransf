

import torch
import models.losses as losses
import models.models as models
import dataloaders.dataloaders as dataloaders
import utils.utils as utils
from utils.fid_scores import fid_pytorch
import config
from torchvision import utils as t_utils
import sys

def main(opt, parser):

    #--- read options ---#
    opt = config.read_arguments(opt, parser,train=True)
    if not hasattr(opt, 'diff_lr_layer'):
        opt.diff_lr_layer = False
    if not hasattr(opt, 'cosineLrSched'):
        opt.cosineLrSched = False
    if opt.transfer_alt_d == 3:
        print("VALUE OF FREEZED IS: ", opt.freezeD)

    if opt.debug:
        opt.gpu_ids = '0'
        opt.batch_size = 4
        print("IN DEBUG MODE")

    #--- create utils ---#
    timer = utils.timer(opt)
    visualizer_losses = utils.losses_saver(opt)
    losses_computer = losses.losses_computer(opt)
    dataloader, dataloader_val = dataloaders.get_dataloaders(opt)
    im_saver = utils.image_saver(opt)
    if not opt.debug:
        fid_computer = fid_pytorch(opt, dataloader_val)

    #--- create models ---#
    model = models.OASIS_model(opt)
    model = models.put_on_multi_gpus(model, opt)

    #--- create optimizers ---#
    if not opt.diff_lr_layer:
        optimizerG = torch.optim.Adam(model.module.netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, opt.beta2))
        optimizerD = torch.optim.Adam(model.module.netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, opt.beta2))
    else:
        list_param_g = [
                {"params": model.module.netG.conv_img.parameters(), "lr": opt.lr_g*0.1},
                {"params": model.module.netG.body[:3].parameters(), "lr": opt.lr_g},
                {"params": model.module.netG.body[3:].parameters(), "lr": opt.lr_g*0.1},
                {"params": model.module.netG.fc.parameters(), "lr": opt.lr_g}
            ]
        list_param_d = [
            {"params": model.module.netD.body_down.parameters(), "lr": opt.lr_d * 0.1},
            {"params": model.module.netD.body_up.parameters(), "lr": opt.lr_d},
            {"params": model.module.netD.layer_up_last.parameters(), "lr": opt.lr_d}
        ]
        if opt.useConfMat:
            list_param_g.append({"params": model.module.netG.mat_conf.parameters(), "lr": opt.lr_g})
            list_param_d.append({"params": model.module.netD.mat_conf.parameters(), "lr": opt.lr_d})
        optimizerG = torch.optim.Adam(list_param_g,
            lr=opt.lr_g, betas=(opt.beta1, opt.beta2)
        )
        optimizerD = torch.optim.Adam(list_param_d,
                                      lr=opt.lr_d, betas=(opt.beta1, opt.beta2)
                                      )
        for param_group in optimizerD.param_groups:
            print('optimizerD, param group', param_group['lr'])
    if opt.cosineLrSched:
        print("USING SCHEDULER")
        schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, 700)
        schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, 700)



    #--- the training loop ---#
    already_started = False
    start_epoch, start_iter = utils.get_start_iters(opt.loaded_latest_iter, len(dataloader))
    print('start epoch, start iter ::', start_epoch, start_iter, opt.num_epochs)
    for epoch in range(start_epoch, opt.num_epochs):
        # print(epoch, start_epoch, opt.num_epochs)
        for i, data_i in enumerate(dataloader):
            # print('at step :', i)
            if not already_started and i < start_iter:
                continue
            already_started = True
            cur_iter = epoch*len(dataloader) + i
            image, label = models.preprocess_input(opt, data_i)

            # model.eval()
            # generated = model(image, label, "generate", losses_computer)
            # print(torch.min(generated), torch.max(generated))
            # t_utils.save_image(
            #     generated,
            #     '0.jpg',
            #     nrow=1,
            #     normalize=True,
            #     range=(-1, 1),
            # )
            # t_utils.save_image(
            #     image,
            #     '1.jpg',
            #     nrow=1,
            #     normalize=True,
            #     range=(-1, 1),
            # )

            # model.train()
            # sys.exit()

            #--- generator update ---#
            model.module.netG.zero_grad()
            loss_G, losses_G_list = model(image, label, "losses_G", losses_computer)
            loss_G, losses_G_list = loss_G.mean(), [loss.mean() if loss is not None else None for loss in losses_G_list]
            loss_G.backward()
            optimizerG.step()
            if opt.cosineLrSched:
                schedulerG.step()
            # print('after optim G')

            #--- discriminator update ---#
            model.module.netD.zero_grad()
            loss_D, losses_D_list = model(image, label, "losses_D", losses_computer)
            loss_D, losses_D_list = loss_D.mean(), [loss.mean() if loss is not None else None for loss in losses_D_list]
            loss_D.backward()
            optimizerD.step()
            if opt.cosineLrSched:
                schedulerD.step()
            # print('after optim D')

            #--- stats update ---#
            if not opt.no_EMA:
                utils.update_EMA(model, cur_iter, dataloader, opt)
            if cur_iter % opt.freq_print == 0:
                # print('Value of ada aug :', model.module.ada_aug_p)
                im_saver.visualize_batch(model, image, label, cur_iter)
                timer(epoch, cur_iter)
            if cur_iter % opt.freq_save_ckpt == 0:
                utils.save_networks(opt, cur_iter, model)
            if cur_iter % opt.freq_save_latest == 0:
                utils.save_networks(opt, cur_iter, model, latest=True)
            if cur_iter % opt.freq_fid == 0 and not opt.debug:# and cur_iter > 0:
            # if cur_iter % 1 == 0:
                is_best, cur_fid = fid_computer.update(model, cur_iter)
                if is_best:
                    utils.save_networks(opt, cur_iter, model, best=True)
            visualizer_losses(cur_iter, losses_G_list+losses_D_list)

    #--- after training ---#
    utils.update_EMA(model, cur_iter, dataloader, opt, force_run_stats=True)
    utils.save_networks(opt, cur_iter, model)
    utils.save_networks(opt, cur_iter, model, latest=True)
    is_best = fid_computer.update(model, cur_iter)
    if is_best:
        utils.save_networks(opt, cur_iter, model, best=True)

    print("The training has successfully finished")

