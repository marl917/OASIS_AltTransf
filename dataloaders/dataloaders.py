import torch


def get_dataset_name(mode):
    if mode == "ade20k":
        return "Ade20kDataset"
    if mode == "cityscapes":
        return "CityscapesDataset"
    if mode == "coco":
        return "CocoStuffDataset"
    if mode == 'ade_indoor':
        return 'AdeIndoorDataset'
    if mode == 'ade_outdoor':
        return 'AdeOutdoorDataset'
    if mode == 'ade_indoor_subset':
        return 'AdeIndoorSubsetDataset'
    elif mode == 'facades':
        return 'FacadesDataset'
    elif mode == 'landscapes':
        return 'LandscapesDataset'
    else:
        ValueError("There is no such dataset regime as %s" % mode)

def get_dataloaders(opt):
    dataset_name   = get_dataset_name(opt.dataset_mode)

    file = __import__("dataloaders."+dataset_name)
    dataset_train = file.__dict__[dataset_name].__dict__[dataset_name](opt, for_metrics=False)
    dataset_val   = file.__dict__[dataset_name].__dict__[dataset_name](opt, for_metrics=True)
    print("Created %s, size train: %d, size val: %d" % (dataset_name, len(dataset_train), len(dataset_val)))

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size = opt.batch_size, shuffle = True, drop_last=False)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size = opt.batch_size//4, shuffle = False, drop_last=False)

    return dataloader_train, dataloader_val