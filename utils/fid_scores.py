import os
import numpy as np
import torch
import time
from scipy import linalg # For numpy FID
from pathlib import Path
from PIL import Image
import models.models as models
from utils.fid_folder.inception import InceptionV3
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import polynomial_kernel

# --------------------------------------------------------------------------#
# This code is an adapted version of https://github.com/mseitzer/pytorch-fid
# --------------------------------------------------------------------------#

class fid_pytorch():
    def __init__(self, opt, dataloader_val):
        self.opt = opt
        self.dims = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
        self.model_inc = InceptionV3([block_idx])
        if opt.gpu_ids != "-1":
            self.model_inc.cuda()
        self.val_dataloader = dataloader_val
        self.m1, self.s1, self.true_act = self.compute_statistics_of_val_path(dataloader_val)
        if os.path.exists(os.path.join(opt.checkpoints_dir, opt.name, "best_fid.txt")):
            with open(os.path.join(opt.checkpoints_dir, opt.name, "best_fid.txt"), "r") as f:
                self.opt.best_fid = float(f.read())
            print('Loading best itr from checkpoint for FID', self.opt.best_fid)
        else:
            print('Initializing best FID')
            self.opt.best_fid = 99999999
        print('Initial value of self.opt.best_fid: ', self.opt.best_fid)
        # if not hasattr(self.opt,'best_fid'):
        #     self.opt.best_fid = 99999999

        self.path_to_save = os.path.join(self.opt.checkpoints_dir, self.opt.name, "FID")
        Path(self.path_to_save).mkdir(parents=True, exist_ok=True)

    def compute_statistics_of_val_path(self, dataloader_val):
        print("--- Now computing Inception activations for real set ---")
        pool = self.accumulate_inception_activations()
        mu, sigma = torch.mean(pool, 0), torch_cov(pool, rowvar=False)
        print("--- Finished FID stats for real set ---")
        return mu, sigma, pool

    def accumulate_inception_activations(self):
        pool, logits, labels = [], [], []
        self.model_inc.eval()
        with torch.no_grad():
            for i, data_i in enumerate(self.val_dataloader):
                image = data_i["image"]
                if self.opt.gpu_ids != "-1":
                    image = image.cuda()
                image = (image + 1) / 2
                pool_val = self.model_inc(image.float())[0][:, :, 0, 0]
                pool += [pool_val]
        return torch.cat(pool, 0)

    def compute_fid_with_valid_path(self, netG, netEMA):
        pool, logits, labels = [], [], []
        self.model_inc.eval()
        netG.eval()
        if not self.opt.no_EMA:
            netEMA.eval()
        with torch.no_grad():
            for i, data_i in enumerate(self.val_dataloader):
                image, label = models.preprocess_input(self.opt, data_i)
                if self.opt.no_EMA:
                    generated = netG(label)
                else:
                    # print('Generating with netEMA')
                    generated = netEMA(label)
                    # print(generated.size(), torch.min(generated), torch.max(generated))
                generated = (generated + 1) / 2
                pool_val = self.model_inc(generated.float())[0][:, :, 0, 0]
                pool += [pool_val]
            pool = torch.cat(pool, 0)
            mu, sigma = torch.mean(pool, 0), torch_cov(pool, rowvar=False)
            answer = self.numpy_calculate_frechet_distance(self.m1, self.s1, mu, sigma)
        netG.train()
        if not self.opt.no_EMA:
            netEMA.train()
        return answer

    def compute_kid_with_valid_path(self, netG, netEMA):
        pool, logits, labels = [], [], []
        self.model_inc.eval()
        netG.eval()
        if not self.opt.no_EMA:
            netEMA.eval()
        with torch.no_grad():
            for i, data_i in enumerate(self.val_dataloader):
                image, label = models.preprocess_input(self.opt, data_i)
                if self.opt.no_EMA:
                    generated = netG(label)
                else:
                    generated = netEMA(label)
                generated = (generated + 1) / 2
                pool_val = self.model_inc(generated.float())[0][:, :, 0, 0]
                pool += [pool_val]
            pool = torch.cat(pool, 0)
            kid_values = polynomial_mmd_averages(self.true_act, pool, n_subsets=100)

        netG.train()
        if not self.opt.no_EMA:
            netEMA.train()
        return kid_values[0].mean()

    def numpy_calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        Taken from https://github.com/bioinf-jku/TTUR
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representive data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representive data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1, sigma1, mu2, sigma2 = mu1.detach().cpu().numpy(), sigma1.detach().cpu().numpy(), mu2.detach().cpu().numpy(), sigma2.detach().cpu().numpy()

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            #print('wat')
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                #print('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        out = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
        return out

    def update(self, model, cur_iter):
        print("--- Iter %s: computing FID ---" % (cur_iter))
        cur_fid = self.compute_fid_with_valid_path(model.module.netG, model.module.netEMA)
        self.update_logs(cur_fid, cur_iter)
        print("--- FID at Iter %s: " % cur_iter, "{:.2f}".format(cur_fid), self.opt.best_fid)
        # cur_kid = self.update_kid(model, cur_iter)
        if cur_fid < self.opt.best_fid and not cur_iter==-1:
            print(f'Updating best fid as cur fid is {cur_fid}, which is better than {self.opt.best_fid} ')
            self.opt.best_fid = cur_fid
            print('values of opt.best_fid after update', self.opt.best_fid, cur_fid)
            with open(os.path.join(self.opt.checkpoints_dir, self.opt.name, "best_fid.txt"), "w") as f:
                f.write(str(cur_fid))
            is_best = True
        else:
            print(f'NOT Updating best fid as cur fid is {cur_fid}, which is not better than {self.opt.best_fid} ')
            is_best = False
        print('IS BEST :', is_best)
        return is_best, cur_fid

    def update_kid(self, model, cur_iter):
        print("--- Iter %s: computing KID ---" % (cur_iter))
        cur_kid = self.compute_kid_with_valid_path(model.module.netG, model.module.netEMA)
        self.update_logs_kid(cur_kid, cur_iter)
        print("--- KID at Iter %s: " % cur_iter, "{:.2f}".format(cur_kid))
        return cur_kid

    def update_logs_kid(self, cur_kid, epoch):
        try :
            np_file = np.load(self.path_to_save + "/kid_log.npy")
            first = list(np_file[0, :])
            sercon = list(np_file[1, :])
            first.append(epoch)
            sercon.append(cur_kid)
            np_file = [first, sercon]
        except:
            np_file = [[epoch], [cur_kid]]

        np.save(self.path_to_save + "/kid_log.npy", np_file)

        np_file = np.array(np_file)
        plt.figure()
        plt.plot(np_file[0, :], np_file[1, :])
        plt.grid(b=True, which='major', color='#666666', linestyle='--')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)
        plt.savefig(self.path_to_save + "/plot_kid", dpi=600)
        plt.close()

    def update_logs(self, cur_fid, epoch):
        try :
            np_file = np.load(self.path_to_save + "/fid_log.npy")
            first = list(np_file[0, :])
            sercon = list(np_file[1, :])
            first.append(epoch)
            sercon.append(cur_fid)
            np_file = [first, sercon]
        except:
            np_file = [[epoch], [cur_fid]]

        np.save(self.path_to_save + "/fid_log.npy", np_file)

        np_file = np.array(np_file)
        plt.figure()
        plt.plot(np_file[0, :], np_file[1, :])
        plt.grid(b=True, which='major', color='#666666', linestyle='--')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)
        plt.savefig(self.path_to_save + "/plot_fid", dpi=600)
        plt.close()

def _sqn(arr):
    flat = np.ravel(arr)
    return flat.dot(flat)

# def polynomial_mmd_averages(codes_g, codes_r, n_subsets=50, subset_size=1000,
#                             ret_var=True,  **kernel_args):
#     m = min(codes_g.shape[0], codes_r.shape[0])
#     mmds = np.zeros(n_subsets)
#     if ret_var:
#         vars = np.zeros(n_subsets)
#     choice = np.random.choice
#
#     # with tqdm(range(n_subsets), desc='MMD') as bar:
#     for i in range(n_subsets):
#         # for i in bar:
#         g = codes_g[choice(len(codes_g), subset_size, replace=False)]
#         r = codes_r[choice(len(codes_r), subset_size, replace=False)]
#         o = polynomial_mmd(g, r, **kernel_args, var_at_m=m, ret_var=ret_var)
#         if ret_var:
#             mmds[i], vars[i] = o
#         else:
#             mmds[i] = o
#         # bar.set_postfix({'mean': mmds[:i + 1].mean()})
#     return (mmds, vars) if ret_var else mmds

def polynomial_mmd_averages(codes_g, codes_r, n_subsets=50, subset_size=1000,
                            ret_var=True,  **kernel_args):
    m = min(codes_g.shape[0], codes_r.shape[0])
    # mmds = np.zeros(n_subsets)
    # if ret_var:
    #     vars = np.zeros(n_subsets)
    # choice = np.random.choice

    # with tqdm(range(n_subsets), desc='MMD') as bar:
    # for i in range(n_subsets):
    #     # for i in bar:
    #     g = codes_g[choice(len(codes_g), subset_size, replace=False)]
    #     r = codes_r[choice(len(codes_r), subset_size, replace=False)]
    o = polynomial_mmd(codes_g, codes_r, **kernel_args, var_at_m=m, ret_var=ret_var)
    # if ret_var:
    #     mmds[i], vars[i] = o
    # else:
    #     mmds[i] = o

    mmds, vars = o
        # bar.set_postfix({'mean': mmds[:i + 1].mean()})
    # return (mmds, vars) if ret_var else mmds
    return (mmds, vars)


def polynomial_mmd(codes_g, codes_r, degree=3, gamma=None, coef0=1,
                   var_at_m=None, ret_var=True):
    # use  k(x, y) = (gamma <x, y> + coef0)^degree
    # default gamma is 1 / dim
    X = codes_g.detach().cpu().numpy()
    Y = codes_r.detach().cpu().numpy()

    K_XX = polynomial_kernel(X, degree=degree, gamma=gamma, coef0=coef0)
    K_YY = polynomial_kernel(Y, degree=degree, gamma=gamma, coef0=coef0)
    K_XY = polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)

    return _mmd2_and_variance(K_XX, K_XY, K_YY,
                              var_at_m=var_at_m, ret_var=ret_var)

def _mmd2_and_variance(K_XX, K_XY, K_YY, unit_diagonal=False,
                       mmd_est='unbiased', block_size=1024,
                       var_at_m=None, ret_var=True):
    # based on
    # https://github.com/dougalsutherland/opt-mmd/blob/master/two_sample/mmd.py
    # but changed to not compute the full kernel matrix at once
    m = K_XX.shape[0]
    assert K_XX.shape == (m, m)
    assert K_XY.shape == (m, m)
    assert K_YY.shape == (m, m)
    if var_at_m is None:
        var_at_m = m

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if unit_diagonal:
        diag_X = diag_Y = 1
        sum_diag_X = sum_diag_Y = m
        sum_diag2_X = sum_diag2_Y = m
    else:
        diag_X = np.diagonal(K_XX)
        diag_Y = np.diagonal(K_YY)

        sum_diag_X = diag_X.sum()
        sum_diag_Y = diag_Y.sum()

        sum_diag2_X = _sqn(diag_X)
        sum_diag2_Y = _sqn(diag_Y)

    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(axis=0)
    K_XY_sums_1 = K_XY.sum(axis=1)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    if mmd_est == 'biased':
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
                + (Kt_YY_sum + sum_diag_Y) / (m * m)
                - 2 * K_XY_sum / (m * m))
    else:
        assert mmd_est in {'unbiased', 'u-statistic'}
        mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m - 1))
        if mmd_est == 'unbiased':
            mmd2 -= 2 * K_XY_sum / (m * m)
        else:
            mmd2 -= 2 * (K_XY_sum - np.trace(K_XY)) / (m * (m - 1))

    if not ret_var:
        return mmd2

    Kt_XX_2_sum = _sqn(K_XX) - sum_diag2_X
    Kt_YY_2_sum = _sqn(K_YY) - sum_diag2_Y
    K_XY_2_sum = _sqn(K_XY)

    dot_XX_XY = Kt_XX_sums.dot(K_XY_sums_1)
    dot_YY_YX = Kt_YY_sums.dot(K_XY_sums_0)

    m1 = m - 1
    m2 = m - 2
    zeta1_est = (
            1 / (m * m1 * m2) * (
            _sqn(Kt_XX_sums) - Kt_XX_2_sum + _sqn(Kt_YY_sums) - Kt_YY_2_sum)
            - 1 / (m * m1) ** 2 * (Kt_XX_sum ** 2 + Kt_YY_sum ** 2)
            + 1 / (m * m * m1) * (
                    _sqn(K_XY_sums_1) + _sqn(K_XY_sums_0) - 2 * K_XY_2_sum)
            - 2 / m ** 4 * K_XY_sum ** 2
            - 2 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
            + 2 / (m ** 3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
    )
    zeta2_est = (
            1 / (m * m1) * (Kt_XX_2_sum + Kt_YY_2_sum)
            - 1 / (m * m1) ** 2 * (Kt_XX_sum ** 2 + Kt_YY_sum ** 2)
            + 2 / (m * m) * K_XY_2_sum
            - 2 / m ** 4 * K_XY_sum ** 2
            - 4 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
            + 4 / (m ** 3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
    )
    var_est = (4 * (var_at_m - 2) / (var_at_m * (var_at_m - 1)) * zeta1_est
               + 2 / (var_at_m * (var_at_m - 1)) * zeta2_est)

    return mmd2, var_est


def torch_cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.
    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.
    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.
    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()
