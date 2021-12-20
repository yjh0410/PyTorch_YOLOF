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


# Convert ndarray to tensor
class ToTensor(object):
    def __call__(self, image, target=None, mask=None):
        # to rgb
        image = image[..., (2, 1, 0)]
        image = F.to_tensor(image)
        if target is not None:
            target["boxes"] = torch.as_tensor(target["boxes"]).float()
            target["labels"] = torch.as_tensor(target["labels"]).long()
        return image, target, mask


# Color Jitter
class ColorJitter(object):
    def __init__(self, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5):
        self.transform = T.ColorJitter(brightness=brightness,
                                       contrast=contrast,
                                       saturation=saturation,
                                       hue=hue)

    def __call__(self, image, target=None, mask=None):
        image = self.transform(image)

        return image, target, mask


# RandomHFlip
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


# RandomShift
class RandomShift(object):
    def __init__(self, p=0.5, max_shift=32):
        self.p = p
        self.max_shift = max_shift

    def __call__(self, image, target=None, mask=None):
        if random.random() < self.p:
            shift_x = random.randint(-self.max_shift, self.max_shift)
            shift_y = random.randint(-self.max_shift, self.max_shift)
            if shift_x < 0:
                new_x = 0
                orig_x = -shift_x
            else:
                new_x = shift_x
                orig_x = 0
            if shift_y < 0:
                new_y = 0
                orig_y = -shift_y
            else:
                new_y = shift_y
                orig_y = 0
            new_image = torch.zeros_like(image)
            img_h, img_w = image.shape[1:]
            new_h = img_h - abs(shift_y)
            new_w = img_w - abs(shift_x)
            new_image[:, new_y:new_y + new_h, new_x:new_x + new_w] = image[:,
                                                                orig_y:orig_y + new_h,
                                                                orig_x:orig_x + new_w]
            boxes_ = target["boxes"].clone()
            boxes_[..., [0, 2]] += shift_x
            boxes_[..., [1, 3]] += shift_y
            boxes_[..., [0, 2]] = boxes_[..., [0, 2]].clamp(0, img_w)
            boxes_[..., [1, 3]] = boxes_[..., [1, 3]].clamp(0, img_h)
            target["boxes"] = boxes_

            return new_image, target, mask

        return image, target, mask


# Normalize tensor image
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None, mask=None):
        # normalize image
        image = F.normalize(image, mean=self.mean, std=self.std)

        return image, target, mask


# Resize tensor image
class Resize(object):
    def __init__(self, min_size=800, max_size=1333, random_size=False):
        self.min_size = min_size
        self.max_size = max_size
        self.random_size = random_size

    def __call__(self, image, target=None, mask=None):
        if self.random_size:
            min_size = random.choice([640, 672, 704, 736, 768, 800])
        else:
            min_size = self.min_size

        # resize
        img_h0, img_w0 = image.shape[1:]
        min_original_size = float(min((img_w0, img_h0)))
        max_original_size = float(max((img_w0, img_h0)))

        if max_original_size / min_original_size * min_size > self.max_size:
            min_size = int(round(min_original_size / max_original_size * self.max_size))

        image = F.resize(image, size=min_size, max_size=self.max_size)

        if target is not None:
            img_h, img_w = image.shape[1:]
            # rescale bbox
            boxes_ = target["boxes"].clone()
            boxes_[:, [0, 2]] = boxes_[:, [0, 2]] / img_w0 * img_w
            boxes_[:, [1, 3]] = boxes_[:, [1, 3]] / img_h0 * img_h
            target["boxes"] = boxes_

        return image, target, mask


# Pad tensor image
class PadImage(object):
    def __init__(self, max_size=1333) -> None:
        self.max_size = max_size

    def __call__(self, image, target=None, mask=None):
        img_h0, img_w0 = image.shape[1:]
        pad_image = torch.zeros([image.size(0), self.max_size, self.max_size]).float()
        pad_image[:, :img_h0, :img_w0] = image

        mask = torch.zeros_like(pad_image[0])
        mask[:img_h0, :img_w0] = 1.0

        return pad_image, target, mask


# TrainTransform
class TrainTransforms(object):
    def __init__(self, min_size=800, max_size=1333, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), random_size=False):
        self.min_size = min_size
        self.mean = mean
        self.std = std
        self.transforms = Compose([
            ToTensor(),
            RandomHorizontalFlip(),
            RandomShift(max_shift=32),
            Resize(min_size=min_size, max_size=max_size, random_size=random_size),
            Normalize(mean, std),
            PadImage(max_size=max_size)
        ])

    def __call__(self, image, target, mask=None):
        return self.transforms(image, target, mask)


# ValTransform
class ValTransforms(object):
    def __init__(self, min_size=800, max_size=1333, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.min_size = min_size
        self.max_size = max_size
        self.mean = mean
        self.std = std
        self.transforms = Compose([
            ToTensor(),
            Resize(min_size=min_size, max_size=max_size),
            Normalize(mean, std)
        ])


    def __call__(self, image, target=None, mask=None):
        return self.transforms(image, target, mask)
