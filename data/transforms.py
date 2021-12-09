import random
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F



class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None, mask=None):
        for t in self.transforms:
            image, target, mask = t(image, target, mask)
        return image, target, mask


class ToTensor(object):
    def __call__(self, image, target=None, mask=None):
        # to rgb
        image = image[..., (2, 1, 0)]
        image = F.to_tensor(image)
        if target is not None:
            target["boxes"] = torch.as_tensor(target["boxes"]).float()
            target["labels"] = torch.as_tensor(target["labels"]).long()
        return image, target, mask


class ColorJitter(object):
    def __init__(self, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5):
        self.transform = T.ColorJitter(brightness=brightness,
                                       contrast=contrast,
                                       saturation=saturation,
                                       hue=hue)

    def __call__(self, image, target=None, mask=None):
        image = self.transform(image)

        return image, target, mask


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target=None, mask=None):
        if random.random() < self.p:
            image = F.hflip(image)
            if target is not None:
                h, w = target["orig_size"]
                if "boxes" in target:
                    boxes = target["boxes"].clone()
                    boxes[..., [0, 2]] = w - boxes[..., [2, 0]]
                    target["boxes"] = boxes

        return image, target, mask


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None, mask=None):
        # normalize image
        image = F.normalize(image, mean=self.mean, std=self.std)

        return image, target, mask


class Resize(object):
    def __init__(self, size=800, random_size=False):
        self.size = size
        self.random_size = random_size

    def __call__(self, image, target=None, mask=None):
        # resize
        if self.random_size:
            size = random.choice([480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800])
        else:
            size = self.size
        image = F.resize(image, size=size, max_size=1333)

        return image, target, mask


class PadImage(object):
    def __init__(self, max_size=1333) -> None:
        self.max_size = max_size

    def __call__(self, image, target=None, mask=None):
        img_h0, img_w0 = target['orig_size']
        img_h, img_w = image.shape[1:]
        pad_image = torch.zeros([image.size(0), self.max_size, self.max_size]).float()
        pad_image[:, :img_h, :img_w] = image

        mask = torch.zeros_like(pad_image[0])
        mask[:img_h, :img_w] = 1.0

        if target is not None:
            # rescale bbox
            boxes_ = target["boxes"].clone()
            boxes_[:, [0, 2]] = boxes_[:, [0, 2]] / img_w0 * img_w
            boxes_[:, [1, 3]] = boxes_[:, [1, 3]] / img_h0 * img_h
            target["boxes"] = boxes_

        return pad_image, target, mask


# TrainTransform
class TrainTransforms(object):
    def __init__(self, size=800, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), random_size=False):
        self.size = size
        self.mean = mean
        self.std = std
        self.transforms = Compose([
            ToTensor(),
            RandomHorizontalFlip(),
            Resize(size, random_size=random_size),
            Normalize(mean, std),
            PadImage(max_size=1333)
        ])

    def __call__(self, image, target, mask=None):
        return self.transforms(image, target, mask)


# ValTransform
class ValTransforms(object):
    def __init__(self, size=800, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.size = size
        self.mean = mean
        self.std = std
        self.transforms = Compose([
            ToTensor(),
            Resize(size),
            Normalize(mean, std)
        ])


    def __call__(self, image, target=None, mask=None):
        return self.transforms(image, target, mask)
