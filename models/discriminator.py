import torch
import torch.nn as nn
import models.norms as norms


class OASIS_Discriminator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        sp_norm = norms.get_spectral_norm(opt)
        output_channel = opt.semantic_nc + 1 # for N+1 loss
        self.channels = [3, 128, 128, 256, 256, 512, 512]
        self.body_up = nn.ModuleList([])
        self.body_down = nn.ModuleList([])
        # encoder part
        for i in range(opt.num_res_blocks):
            self.body_down.append(residual_block_D(self.channels[i], self.channels[i+1], opt, -1, first=(i==0)))
        # decoder part
        self.body_up.append(residual_block_D(self.channels[-1], self.channels[-2], opt, 1))
        for i in range(1, opt.num_res_blocks-1):
            self.body_up.append(residual_block_D(2*self.channels[-1-i], self.channels[-2-i], opt, 1))
        self.body_up.append(residual_block_D(2*self.channels[1], 64, opt, 1))
        

        if opt.conf_mat!=None and not opt.notConfMatDisc:
            print("USING CONF MAT IN DISC")
            self.mat_conf = self.opt.conf_matrix.detach().clone()
            output_channel = self.mat_conf.size(1) + 1
            # print('size of mat conf:', self.mat_conf.size())
            if self.opt.dataset_mode == 'cityscapes':
                # self.mat_conf = torch.cat((self.mat_conf[:-1], torch.zeros_like(self.mat_conf[0:1])),
                #                 dim=0)  # as last class in case of Cityscapes if full of NaN
                self.mat_conf = torch.nn.parameter.Parameter(
                    torch.cat((self.mat_conf[:-1], torch.zeros_like(self.mat_conf[None, -1])), dim=0),
                    requires_grad=True if not opt.notConfMatDisc else False)
            else:
                self.mat_conf = torch.nn.parameter.Parameter(
                    self.mat_conf,
                    requires_grad=True if not opt.notConfMatDisc else False)
        if opt.transfer_alt_d == 6:
            self.layer_up_last_target = nn.Conv2d(64, opt.semantic_nc + 1, 1, 1, 0)

        self.layer_up_last = nn.Conv2d(64, output_channel, 1, 1, 0)

    def forward(self, input):
        x = input
        #encoder
        encoder_res = list()
        for i in range(len(self.body_down)):
            x = self.body_down[i](x)
            encoder_res.append(x)
        #decoder
        x = self.body_up[0](x)
        for i in range(1, len(self.body_down)):
            x = self.body_up[i](torch.cat((encoder_res[-i-1], x), dim=1))
        ans = self.layer_up_last(x)

        if self.opt.transfer_alt_d == 6:
            x_target = self.layer_up_last_target(x)

        if self.opt.conf_mat!=None and not self.opt.notConfMatDisc:
            b, c, h, w = ans[:,1:,:,:].size()
            temp_ans = ans[:,1:,:,:].view(b, c, -1).permute((0, 2, 1))
            mat_conf_m = torch.unsqueeze(torch.transpose(self.mat_conf, 0, 1), dim=0).repeat(ans.size(0), 1, 1).to(
                ans.get_device())
            # print('DISC dimension of mat: ', mat.size(), torch.min(mat), torch.max(mat), temp_ans.size())
            seg = torch.bmm(temp_ans, mat_conf_m).permute(0, 2, 1).view(b, -1, h, w)
            ans = torch.cat((ans[:,0:1,:,:],seg), dim=1)
            if self.opt.transfer_alt_d == 6:
                ans = ans + x_target
            # print('DISC final ans size: ', ans.size())
        return ans



class residual_block_D(nn.Module):
    def __init__(self, fin, fout, opt, up_or_down, first=False):
        super().__init__()
        # Attributes
        self.up_or_down = up_or_down
        self.first = first
        self.learned_shortcut = (fin != fout)
        fmiddle = fout
        norm_layer = norms.get_spectral_norm(opt)
        if first:
            self.conv1 = nn.Sequential(norm_layer(nn.Conv2d(fin, fmiddle, 3, 1, 1)))
        else:
            if self.up_or_down > 0:
                self.conv1 = nn.Sequential(nn.LeakyReLU(0.2, False), nn.Upsample(scale_factor=2), norm_layer(nn.Conv2d(fin, fmiddle, 3, 1, 1)))
            else:
                self.conv1 = nn.Sequential(nn.LeakyReLU(0.2, False), norm_layer(nn.Conv2d(fin, fmiddle, 3, 1, 1)))
        self.conv2 = nn.Sequential(nn.LeakyReLU(0.2, False), norm_layer(nn.Conv2d(fmiddle, fout, 3, 1, 1)))
        if self.learned_shortcut:
            self.conv_s = norm_layer(nn.Conv2d(fin, fout, 1, 1, 0))
        if up_or_down > 0:
            self.sampling = nn.Upsample(scale_factor=2)
        elif up_or_down < 0:
            self.sampling = nn.AvgPool2d(2)
        else:
            self.sampling = nn.Sequential()

    def forward(self, x):
        x_s = self.shortcut(x)
        dx = self.conv1(x)
        dx = self.conv2(dx)
        if self.up_or_down < 0:
            dx = self.sampling(dx)
        out = x_s + dx
        return out

    def shortcut(self, x):
        if self.first:
            if self.up_or_down < 0:
                x = self.sampling(x)
            if self.learned_shortcut:
                x = self.conv_s(x)
            x_s = x
        else:
            if self.up_or_down > 0:
                x = self.sampling(x)
            if self.learned_shortcut:
                x = self.conv_s(x)
            if self.up_or_down < 0:
                x = self.sampling(x)
            x_s = x
        return x_s
