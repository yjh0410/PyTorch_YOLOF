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

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    def __call__(self, image, target=None):
        # to rgb
        image = image[..., (2, 1, 0)]
        image = F.to_tensor(image)
        if target is not None:
            target["boxes"] = torch.as_tensor(target["boxes"]).float()
            target["labels"] = torch.as_tensor(target["labels"]).long()
        return image, target


class ColorJitter(object):
    def __init__(self, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5):
        self.transform = T.ColorJitter(brightness=brightness,
                                       contrast=contrast,
                                       saturation=saturation,
                                       hue=hue)

    def __call__(self, image, target=None):
        image = self.transform(image)

        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target=None):
        if random.random() < self.p:
            image = F.hflip(image)
            if target is not None:
                h, w = target["orig_size"]
                if "boxes" in target:
                    boxes = target["boxes"].clone()
                    boxes[..., [0, 2]] = w - boxes[..., [2, 0]]
                    target["boxes"] = boxes

        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        # normalize image
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is not None:
            h, w = target["orig_size"]
            if "boxes" in target:
                boxes = target["boxes"].clone()
                # normalize bboxes
                boxes  = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
                target["boxes"] = boxes

        return image, target


class Resize(object):
    def __init__(self, size=800, random_size=False):
        self.size = size
        self.random_size = random_size

    def __call__(self, image, target=None):
        # resize
        if self.random_size:
            size = random.choice([480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800])
        else:
            size = self.size
        image = F.resize(image, size=size, max_size=1333)

        return image, target


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
            Normalize(mean, std)
        ])

    def __call__(self, image, target):
        return self.transforms(image, target)


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


    def __call__(self, image, target=None):
        return self.transforms(image, target)
