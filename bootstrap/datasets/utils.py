import torchvision.transforms as transforms


def default_image_tf(scale_size, crop_size,
        mean=[0.485, 0.456, 0.406], # resnet imagnet
        std=[0.229, 0.224, 0.225],
        test=False):
    tfs = []
    tfs.append(transforms.Resize(scale_size))

    if not test:
        tfs.append(transforms.RandomCrop(crop_size))
    else:
        tfs.append(transforms.CenterCrop(crop_size))

    tfs.append(transforms.ToTensor()) # divide by 255 automatically
    tfs.append(transforms.Normalize(mean=mean, std=std))
    transform = transforms.Compose(tfs)
    return transform
