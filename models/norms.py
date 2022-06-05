import torch.nn.utils.spectral_norm as spectral_norm
from models.sync_batchnorm import SynchronizedBatchNorm2d
import torch.nn as nn
import torch.nn.functional as F
import torch



class SPADE(nn.Module):
    def __init__(self, opt, norm_nc, label_nc):
        super().__init__()
        self.opt = opt
        self.first_norm = get_norm_layer(opt, norm_nc)
        if self.opt.transfer_alt_g in [3,4,5,6]:
            self.first_norm.eval()
        # self.first_norm = SynchronizedBatchNorm2d(norm_nc, affine=True)
        ks = opt.spade_ks
        nhidden = 128
        pw = ks // 2

        # if self.opt.transfer_alt == 1:
        #     n_class_shared = 183 + self.opt.z_dim
        #     self.embedding = nn.Conv2d(label_nc - self.opt.z_dim, 183, kernel_size=1, padding=0, bias = False)
        #     self.mlp_shared_new = nn.Sequential(
        #         nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
        #         nn.ReLU()
        #     )
        # print("VALUE OF LABEL NC: ", label_nc)
        if self.opt.conf_mat!=None and not self.opt.notConfMatGen:
            if 'ade20k' in opt.transfer_l:
                target_nc_sem = 151
            elif 'coco' in opt.transfer_l:
                target_nc_sem = 183
            n_class_shared = target_nc_sem + self.opt.z_dim
#             n_class_shared = label_nc #version to make it work at test time
        else:
            n_class_shared = label_nc
        # print('Nb of mlp share ipt: ', n_class_shared)

        # if self.opt.transfer_alt ==6:
        #     self.mlp_shared_target = nn.Sequential(
        #         nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
        #         nn.ReLU()
        #     )
        if self.opt.transfer_alt_g == 2:
            self.mlp_shared_target = nn.Sequential(
                nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
                nn.ReLU()
            )
            self.similarity = nn.Sequential(
                nn.Conv2d(label_nc, n_class_shared, kernel_size=1, padding=0, stride = 1),
                nn.ReLU()
            )

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(n_class_shared, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )


        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)



    def forward(self, x, segmap):
        normalized = self.first_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        # if self.opt.transfer_alt==1:
        #     actv_new = self.mlp_shared_new(segmap)
        #     # print(torch.min(self.embedding.weight), torch.max(self.embedding.weight))
        #     # print('dim of embedding before emb: ', self.embedding.weight.size())
        #     # self.embedding.weight = torch.clamp(self.embedding.weight, 0, 1)
        #     # print('dim of embedding after emb: ', self.embedding.weight.size())
        #     # print('after clamping')
        #     # seg_argmax = torch.argmax(segmap[:,self.opt.z_dim:,:,:], dim=1)
        #     seg_map_class = self.embedding(segmap[:,self.opt.z_dim:,:,:])
        #     # print('after embedding')
        #     # seg_map = seg_map.permute(0,3,1,2)
        #     # print('segmap size: ', seg_map.size())
        #     segmap = torch.cat((segmap[:,:self.opt.z_dim,:,:], seg_map_class), dim=1)
        # if self.opt.transfer_alt == 5:
        #     b,c,h,w = segmap[:,self.opt.z_dim:,:,:].size()
        #     segmap1 = segmap[:,self.opt.z_dim:,:,:].view(b,c,-1).permute((0,2,1))
        #     dev = segmap1.get_device()
        #     mat = torch.unsqueeze(self.opt.conf_matrix, dim=0).repeat(b,1,1).to(dev)
        #     # print('in norms: ', segmap1.size(),mat.size())
        #     segmap1 = torch.bmm(segmap1, mat).permute(0,2,1).view(b,-1,h,w)
        #     segmap = torch.cat((segmap[:,:self.opt.z_dim,:,:], segmap1), dim=1)

        # if self.opt.transfer_alt == 6:
        #     seg_coco = F.interpolate(seg_coco, size=x.size()[2:], mode='nearest')
        #     actv = self.mlp_shared(seg_coco)
        #     actv_target = self.mlp_shared_target(segmap)
        #     actv= actv_target + actv
        # else:
        if self.opt.transfer_alt_g == 2:  #similarity matrix that maps target classes to source classes
            actv_t = self.mlp_shared_target(segmap)
            segmap = self.similarity(segmap)


        actv = self.mlp_shared(segmap)

        if self.opt.transfer_alt_g == 2:  # similarity matrix that maps target classes to source classes
            actv+=actv_t



        # if self.opt.transfer_alt == 1:
        #     actv = actv_new + actv

        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        # if self.opt.transfer_alt ==1:
        #     return out, torch.mean(torch.abs(seg_map_class)),  torch.mean(actv_new**2)
        # else:
        return out


def get_spectral_norm(opt):
    if opt.no_spectral_norm:
        return torch.nn.Identity()
    else:
        return spectral_norm


def get_norm_layer(opt, norm_nc):
    if opt.param_free_norm == 'instance':
        return nn.InstanceNorm2d(norm_nc, affine=False)
    if opt.param_free_norm == 'syncbatch':
        return SynchronizedBatchNorm2d(norm_nc, affine=False)
    if opt.param_free_norm == 'batch':
        return nn.BatchNorm2d(norm_nc, affine=False)
    else:
        raise ValueError('%s is not a recognized param-free norm type in SPADE'
                         % opt.param_free_norm)