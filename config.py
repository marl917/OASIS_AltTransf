import argparse
import pickle
import os
import utils.utils as utils

def read_arguments_submitit(train=True):
    parser = argparse.ArgumentParser()
    parser = add_all_arguments(parser, train)
    parser.add_argument('--phase', type=str, default='train')
    opt = parser.parse_args()

    use_slurm = opt.slurm

    print('use slurm :', use_slurm)
    print(os.path.join(opt.checkpoints_dir, opt.name, "latest_iter.txt"),os.path.join(opt.checkpoints_dir, opt.name, "opt.pkl") )
    if os.path.isfile(os.path.join(opt.checkpoints_dir, opt.name, "latest_iter.txt")) and os.path.isfile(
            os.path.join(opt.checkpoints_dir, opt.name, "opt.pkl")) and os.path.isfile(os.path.join(opt.checkpoints_dir, opt.name, "losses/losses.npy")):
        print('Continue Training')
        opt.continue_train = True
        parser.set_defaults(continue_train=True)

    if train:
        num_epochs = opt.num_epochs
        set_dataset_default_lm(opt, parser)
        if opt.continue_train:
            update_options_from_file(opt, parser)
            opt.slurm = use_slurm
            opt.num_epochs = num_epochs

            opt.continue_train = True
            parser.set_defaults(continue_train=True)
            parser.set_defaults(slurm=use_slurm)
    opt = parser.parse_args()
    return opt, parser

def read_arguments(opt, parser, train=True):
    # parser = argparse.ArgumentParser()
    # parser = add_all_arguments(parser, train)
    # parser.add_argument('--phase', type=str, default='train')
    # opt = parser.parse_args()
    # use_slurm = opt.slurm

    # if os.path.isfile(os.path.join(opt.checkpoints_dir, opt.name, "latest_iter.txt")) and os.path.isfile(
    #         os.path.join(opt.checkpoints_dir, opt.name, "opt.pkl")):
    #     print('Continue Training')
    #     opt.continue_train = True
    #
    # if train:
    #     set_dataset_default_lm(opt, parser)
    #     if opt.continue_train:
    #         update_options_from_file(opt, parser)
    #         # opt.slurm = use_slurm
    # # opt = parser.parse_args()

    if os.path.isfile(os.path.join(opt.checkpoints_dir, opt.name, "latest_iter.txt")) and os.path.isfile(
            os.path.join(opt.checkpoints_dir, opt.name, "opt.pkl")) and os.path.isfile(os.path.join(opt.checkpoints_dir, opt.name, "losses/losses.npy")):
        print('Continue Training')
        opt.continue_train = True
        parser.set_defaults(continue_train=True)


    opt.phase = 'train' if train else 'test'
    if train:
        print("CONTINUE TRAIN:", opt.continue_train)
        opt.loaded_latest_iter = 0 if not opt.continue_train else load_iter(opt)
        print(opt.loaded_latest_iter)
    utils.fix_seed(opt.seed)
    print_options(opt, parser)
    if train:
        save_options(opt, parser)
    return opt


def add_all_arguments(parser, train):
    #--- general options ---
    parser.add_argument('--name', type=str, default='label2coco', help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--size', type=int, default=256, help='size of images')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoints_dir', type=str, default='/checkpoint/marlenec/oasis_training', help='models are saved here')
    parser.add_argument('--no_spectral_norm', action='store_true', help='this option deactivates spectral norm in all layers')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--dataroot', type=str, default='./datasets/cityscapes/', help='path to dataset root')
    parser.add_argument('--dataset_mode', type=str, default='coco', help='this option indicates which dataset should be loaded')
    parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')
    parser.add_argument('--perctg_train_data', type=float, default=1.,
                        help='to train only on a subset of the training dataset')
    parser.add_argument('--split', type=float, default=1.,
                        help='split to train only on a subset of the training dataset, from 1 to ...')
    parser.add_argument('--slurm', action='store_true',
                        help='if specified, use submitit to launch jobs')
    parser.add_argument('--partition', type=str, default='learnlab',
                        help='name of the partition if using slurm')
    parser.add_argument('--submLogFolder', type=str, default='/checkpoint/marlenec/oasis_submitit_logs',
                        help='name of the partition if using slurm')

    # for generator
    parser.add_argument('--num_res_blocks', type=int, default=6, help='number of residual blocks in G and D')
    parser.add_argument('--channels_G', type=int, default=64, help='# of gen filters in first conv layer in generator')
    parser.add_argument('--param_free_norm', type=str, default='syncbatch', help='which norm to use in generator before SPADE')
    parser.add_argument('--spade_ks', type=int, default=3, help='kernel size of convs inside SPADE')
    parser.add_argument('--no_EMA', action='store_true', help='if specified, do *not* compute exponential moving averages')
    parser.add_argument('--EMA_decay', type=float, default=0.9999, help='decay in exponential moving averages')
    parser.add_argument('--no_3dnoise', action='store_true', default=False, help='if specified, do *not* concatenate noise to label maps')
    parser.add_argument('--z_dim', type=int, default=64, help="dimension of the latent z vector")

    parser.add_argument('--transfer_alt_g', type=int, default=0,
                        help='alt transfer g')
    parser.add_argument('--transfer_alt_d', type=int, default=0,
                        help='alt transfer d')
    parser.add_argument('--transfer_l', type=str, default='',
                        help='do transfer learning : specify dir of pretrained networks')

    ## transfer_alt = 1: MineGAN
    ## transfer_alt = 2: Efficient Knowledge Transfer
    ## transfer_alt = 3: FreezeD
    ## transfer_alt = 5: Confusion Matrix
    parser.add_argument('--conf_mat', type=str, default=None,
                        help='trainable gamma beta index layers (from 0 to 5)')
    parser.add_argument('--useConfMat', action='store_true', default=False,
                        help='if specified, do *not* train conf mat in disc')
    parser.add_argument('--notConfMatGen', action='store_true', default=False, help='if specified, do *not* train conf mat in gen')
    parser.add_argument('--notConfMatDisc', action='store_true', default=False,
                        help='if specified, do *not* train conf mat in disc')
    
    parser.add_argument('--subset_nb', type=int, default=0, help="dimension of the latent z vector")
    parser.add_argument('--use_subset', action='store_true',
                        help='if specified, use subset for finetuning')

    parser.add_argument('--diff_lr_layer', action='store_true',
                        help='if specified, use subset for finetuning')
    parser.add_argument('--cosineLrSched', action='store_true',
                        help='if specified, use cosine annealing lr')

    if train:
        parser.add_argument('--debug', action='store_true', default=False, help='nofid')
        parser.add_argument('--freq_print', type=int, default=500, help='frequency of showing training results')
        parser.add_argument('--freq_save_ckpt', type=int, default=20000, help='frequency of saving the checkpoints')
        parser.add_argument('--freq_save_latest', type=int, default=1000, help='frequency of saving the latest model')
        parser.add_argument('--freq_smooth_loss', type=int, default=250, help='smoothing window for loss visualization')
        parser.add_argument('--freq_save_loss', type=int, default=2500, help='frequency of loss plot updates')
        parser.add_argument('--freq_fid', type=int, default=5000, help='frequency of saving the fid score (in training iterations)')
        # parser.add_argument('--continue_train', action='store_true', help='resume previously interrupted training')
        parser.add_argument('--continue_train', type=bool,default=False, help='resume previously interrupted training')
        parser.add_argument('--which_iter', type=str, default='latest', help='which epoch to load when continue_train')



        parser.add_argument('--train_index_gabeta', type=str, default='', help='trainable gamma beta index layers (from 0 to 5)')
        # parser.add_argument('--no_pretrainD',action='store_true', default=False, help='randomly initialize D')
        # parser.add_argument('--no_pretrainG', action='store_true', default=False, help='randomly initialize D')
        parser.add_argument('--trainD', action='store_true', default=False, help='update all weights of D')
        # parser.add_argument('--trainDdecoderSkip', action='store_true', default=False, help='update all weights of D')
        # parser.add_argument('--trainDSkip', action='store_true', default=False, help='update all weights of D')
        parser.add_argument('--trainG', action='store_true', default=False, help='update all weights of G')
        # parser.add_argument('--trainDdecoder', action='store_true', default=False, help='update decoder weights of D')
        parser.add_argument('--freezeD', type=int, default=0,
                            help='nb of param to freeze in Disc')
        parser.add_argument('--freezeD_list', type=str, default=None, help='nb of param to freeze in Disc')

        parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs to train')
        parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        parser.add_argument('--lr_g', type=float, default=0.0001, help='G learning rate, default=0.0001')
        parser.add_argument('--lr_d', type=float, default=0.0004, help='D learning rate, default=0.0004')

        parser.add_argument('--channels_D', type=int, default=64, help='# of discrim filters in first conv layer in discriminator')
        parser.add_argument('--add_vgg_loss', action='store_true', help='if specified, add VGG feature matching loss')
        parser.add_argument('--lambda_vgg', type=float, default=10.0, help='weight for VGG loss')
        parser.add_argument('--no_balancing_inloss', action='store_true', default=False, help='if specified, do *not* use class balancing in the loss function')
        parser.add_argument('--no_labelmix', action='store_true', default=False, help='if specified, do *not* use LabelMix')
        parser.add_argument('--lambda_labelmix', type=float, default=10.0, help='weight for LabelMix regularization')
        parser.add_argument('--augment', action='store_true', help='if specified, add ADA')
        parser.add_argument('--ada_aug_p', type=float, default=0.3, help='weight for LabelMix regularization')
        parser.add_argument('--lambda_l1loss', type=float, default=10.0, help='weight for l1 regularization on class map embedding')
    else:
        parser.add_argument('--results_dir', type=str, default='/checkpoint/marlenec/oasis_training/results', help='saves testing results here.')
        parser.add_argument('--ckpt_iter', type=str, default='best', help='which epoch to load to evaluate a model')
        parser.add_argument('--ckpt_disc', type=str, default=None, help='which epoch to load to evaluate a model')
        parser.add_argument('--fid', action='store_true', default=False, help='compute fid')
        parser.add_argument('--kid', action='store_true', default=False, help='compute kid')
        parser.add_argument('--disc_out', action='store_true', help='compute disc output')
        parser.add_argument('--gen_img', action='store_true', help='generate images')
        parser.add_argument('--creative_gen', action='store_true', help='creative gen by mixing labels')
        parser.add_argument('--class_mixing', action='store_true', help='creative gen by mixing labels')
    return parser


def set_dataset_default_lm(opt, parser):
    if opt.dataset_mode == "ade20k":
        parser.set_defaults(lambda_labelmix=10.0)
        parser.set_defaults(EMA_decay=0.9999)
    if opt.dataset_mode == "cityscapes":
        parser.set_defaults(lr_g=0.0004)
        parser.set_defaults(lambda_labelmix=5.0)
        parser.set_defaults(freq_fid=2500)
        parser.set_defaults(EMA_decay=0.999)
    if opt.dataset_mode == "coco":
        parser.set_defaults(lambda_labelmix=10.0)
        parser.set_defaults(EMA_decay=0.9999)
        parser.set_defaults(num_epochs=100)


def save_options(opt, parser):
    path_name = os.path.join(opt.checkpoints_dir,opt.name)
    os.makedirs(path_name, exist_ok=True)
    with open(path_name + '/opt.txt', 'wt') as opt_file:
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

    with open(path_name + '/opt.pkl', 'wb') as opt_file:
        pickle.dump(opt, opt_file)


def update_options_from_file(opt, parser):
    new_opt = load_options(opt)
    for k, v in sorted(vars(opt).items()):
        if hasattr(new_opt, k) and v != getattr(new_opt, k):
            new_val = getattr(new_opt, k)
            parser.set_defaults(**{k: new_val})
    return parser


def load_options(opt):
    file_name = os.path.join(opt.checkpoints_dir, opt.name, "opt.pkl")
    new_opt = pickle.load(open(file_name, 'rb'))
    return new_opt


def load_iter(opt):
    if opt.which_iter == "latest":
        with open(os.path.join(opt.checkpoints_dir, opt.name, "latest_iter.txt"), "r") as f:
            res = int(f.read())
            return res
    elif opt.which_iter == "best":
        with open(os.path.join(opt.checkpoints_dir, opt.name, "best_iter.txt"), "r") as f:
            res = int(f.read())
            return res
    else:
        return int(opt.which_iter)


def print_options(opt, parser):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)
