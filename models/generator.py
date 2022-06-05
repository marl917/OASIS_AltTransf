import torch.nn as nn
import models.norms as norms
import torch
import torch.nn.functional as F



class OASIS_Generator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        sp_norm = norms.get_spectral_norm(opt)
        ch = opt.channels_G
        self.channels = [16*ch, 16*ch, 16*ch, 8*ch, 4*ch, 2*ch, 1*ch]
        self.init_W, self.init_H = self.compute_latent_vector_size(opt)
        self.conv_img = nn.Conv2d(self.channels[-1], 3, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)
        self.body = nn.ModuleList([])
        for i in range(len(self.channels)-1):
            self.body.append(ResnetBlock_with_SPADE(self.channels[i], self.channels[i+1], opt))
        if not self.opt.no_3dnoise:
            if opt.conf_mat!=None and not opt.notConfMatGen:
                if 'ade20k' in opt.transfer_l:
                    target_nc_sem = 151
                elif 'coco' in opt.transfer_l:
                    target_nc_sem = 183
                self.fc = nn.Conv2d(target_nc_sem + self.opt.z_dim, 16 * ch, 3, padding=1)
#                   self.fc = nn.Conv2d(self.opt.semantic_nc + self.opt.z_dim, 16 * ch, 3, padding=1)   ####version to make it work at test time
            else:
                self.fc = nn.Conv2d(self.opt.semantic_nc + self.opt.z_dim, 16 * ch, 3, padding=1)
        else:
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * ch, 3, padding=1)
        if self.opt.conf_mat!=None and not opt.notConfMatGen:
            print("USING CONF MAT IN GEN")
            self.mat_conf = self.opt.conf_matrix.detach().clone()
            self.mat_conf = torch.nn.parameter.Parameter(torch.cat((self.mat_conf[:-1], torch.zeros_like(self.mat_conf[None, -1])), dim=0), requires_grad = True)
            # print('mat conf requires grad', self.mat_conf.requires_grad)
        if self.opt.transfer_alt_g == 1:
            self.linear_miner = nn.Linear(self.opt.z_dim, self.opt.z_dim)
    def compute_latent_vector_size(self, opt):
        w = opt.crop_size // (2**(opt.num_res_blocks-1))
        h = round(w / opt.aspect_ratio)
        return h, w

    def forward(self, input, z=None):
        seg = input
        if self.opt.conf_mat!=None and not self.opt.notConfMatGen:
            b, c, h, w = seg.size()
            segmap1 = seg.view(b, c, -1).permute((0, 2, 1))
            dev = segmap1.get_device()
            mat = torch.unsqueeze(self.mat_conf, dim=0).repeat(b, 1, 1).to(dev)
            # print('in main gen, mat is: ', torch.min(mat), torch.max(mat))
            # print('in gen: ', segmap1.size(),mat.size())
            # print('size of seg in main gen is :', seg.size(), torch.unique(seg))
            # if self.opt.transfer_alt == 5:
            seg = torch.bmm(segmap1, mat).permute(0, 2, 1).view(b, -1, h, w)
            # elif self.opt.transfer_alt == 6:
            #     seg_coco = torch.bmm(segmap1, mat).permute(0, 2, 1).view(b, -1, h, w)
        if self.opt.gpu_ids != "-1":
            seg.cuda()
        if not self.opt.no_3dnoise:
            dev = seg.get_device() if self.opt.gpu_ids != "-1" else "cpu"
            z = torch.randn(seg.size(0), self.opt.z_dim, dtype=torch.float32, device=dev)
            if self.opt.transfer_alt_g == 1:
                z = self.linear_miner(z)
                # print("SIZE OF MINER Z: ", z.size())
            z = z.view(z.size(0), self.opt.z_dim, 1, 1)
            z1 = z.expand(z.size(0), self.opt.z_dim, seg.size(2), seg.size(3))
            seg = torch.cat((z1, seg), dim = 1)
            # if self.opt.transfer_alt == 6:
            #     z1 = z.expand(z.size(0), self.opt.z_dim, seg_coco.size(2), seg_coco.size(3))
            #     seg_coco = torch.cat((z1, seg_coco), dim=1)
        x = F.interpolate(seg, size=(self.init_W, self.init_H))
        x = self.fc(x)
        for i in range(self.opt.num_res_blocks):
            x = self.body[i](x, seg)
            if i < self.opt.num_res_blocks-1:
                x = self.up(x)
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)
        return x


class ResnetBlock_with_SPADE(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        self.opt = opt
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        sp_norm = norms.get_spectral_norm(opt)
        self.conv_0 = sp_norm(nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1))
        self.conv_1 = sp_norm(nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1))
        if self.learned_shortcut:
            self.conv_s = sp_norm(nn.Conv2d(fin, fout, kernel_size=1, bias=False))

        spade_conditional_input_dims = opt.semantic_nc


        if not opt.no_3dnoise:
            spade_conditional_input_dims += opt.z_dim

        self.norm_0 = norms.SPADE(opt, fin, spade_conditional_input_dims)
        self.norm_1 = norms.SPADE(opt, fmiddle, spade_conditional_input_dims)
        if self.learned_shortcut:
            self.norm_s = norms.SPADE(opt, fin, spade_conditional_input_dims)
        self.activ = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, seg):
        # l1loss = torch.tensor(0.).cuda()
        # l2loss = torch.tensor(0.).cuda()
        if self.learned_shortcut:
            x_s = self.norm_s(x, seg)
            x_s = self.conv_s(x_s)
            # if self.opt.transfer_alt == 1:
            #     l1loss += l1
            #     l2loss += l2
        else:
            x_s = x
        dx = self.norm_0(x, seg)
        dx = self.conv_0(self.activ(dx))
        dx = self.norm_1(dx, seg)
        dx = self.conv_1(self.activ(dx))

        # if self.opt.transfer_alt ==1:
        #     l1loss = l1loss + l11 + l21
        #     l2loss = l2loss + l12 + l22
        out = x_s + dx
        # if return_l1loss:
        #     return out, l1loss, l2loss
        return out
