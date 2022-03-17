import random
import cv2
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
    def __init__(self, format='RGB'):
        self.format = format

    def __call__(self, image, target=None, mask=None):
        # to rgb
        if self.format == 'RGB':
            image = image[..., (2, 1, 0)]
        elif self.format == 'BGR':
            image = image
        else:
            print('Unknown color format !!')
            exit()
        image = F.to_tensor(image)
        if target is not None:
            target["boxes"] = torch.as_tensor(target["boxes"]).float()
            target["labels"] = torch.as_tensor(target["labels"]).long()
        return image, target, mask


# DistortTransform
class DistortTransform(object):
    """
    Distort image.
    """
    def __call__(self, image, target=None, mask=None):
        """
        Args:
            img (ndarray): of shape HxWxC. The Tensor is floating point in range[0, 255].

        Returns:
            ndarray: the distorted image(s).
        """
        def _convert(image, alpha=1, beta=0):
            tmp = image.astype(float) * alpha + beta
            tmp[tmp < 0] = 0
            tmp[tmp > 255] = 255
            image[:] = tmp

        image = image.copy()

        _convert(image, beta=random.uniform(-32, 32))
        _convert(image, alpha=random.uniform(0.5, 1.5))
        # BGR -> HSV
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
        tmp %= 180
        image[:, :, 0] = tmp
        _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))
        # HSV -> BGR
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        return image, target, mask


# RandomSizeCrop
class RandomSizeCrop(object):
    def __init__(self, max_size=800, pixel_mean=(0.485, 0.456, 0.406)):
        self.min_size = round(max_size * 0.3)
        self.max_size = max_size
        self.pixel_mean = torch.tensor(pixel_mean).float()[..., None, None]

    def crop(self, image, region, target=None, mask=None):
        oh, ow = image.shape[1:]
        cropped_image = torch.ones_like(image) * self.pixel_mean
        cropped_image_ = F.crop(image, *region)
        ph, pw = cropped_image_.shape[1:]

        dh, dw = oh - ph, ow - pw
        pleft = random.randint(0, dw)
        ptop = random.randint(0, dh)
        cropped_image[:, ptop:ptop+ph, pleft:pleft+pw] = cropped_image_

        target = target.copy()
        i, j, h, w = region

        if "boxes" in target:
            boxes = target["boxes"]
            max_size = torch.as_tensor([w, h], dtype=torch.float32)
            cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
            cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
            cropped_boxes = cropped_boxes.clamp(min=0).reshape(-1, 4)
            # offset
            cropped_boxes[..., [0, 2]] += pleft
            cropped_boxes[..., [1, 3]] += ptop

            # check box
            valid_box = []
            if len(cropped_boxes) == 0:
                valid_box.append([0., 0., 0., 0.])
            else:
                for box in cropped_boxes:
                    x1, y1, x2, y2 = box
                    bw = x2 - x1
                    bh = y2 - y1
                    if bw > 10. and bh > 10.:
                        valid_box.append([x1, y1, x2, y2])

                if len(valid_box) == 0:
                    valid_box.append([0., 0., 0., 0.])

            target["boxes"] = torch.tensor(valid_box).float()

        if mask is not None:
            # FIXME should we update the area here if there are no boxes?
            mask = mask[:, ptop:ptop+ph, pleft:pleft+pw]
            print(mask)

        return cropped_image, target, mask

    def __call__(self, image, target=None, mask=None):
        height, width = image.shape[1:]
        max_size = min(width, self.max_size)
        if max_size > self.min_size:
            w = random.randint(self.min_size, min(width, self.max_size))
        else:
            w = width

        max_size = min(height, self.max_size)
        if max_size > self.min_size:
            h = random.randint(self.min_size, min(height, self.max_size))
        else:
            h = height

        region = T.RandomCrop.get_params(image, [h, w])
        return self.crop(image, region, target, mask)


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
    def __init__(self, pixel_mean, pixel_std):
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std

    def __call__(self, image, target=None, mask=None):
        # normalize image
        image = F.normalize(image, mean=self.pixel_mean, std=self.pixel_std)

        return image, target, mask


# Resize tensor image
class Resize(object):
    def __init__(self, min_size=800, max_size=1333, random_size=None):
        self.min_size = min_size
        self.max_size = max_size
        self.random_size = random_size

    def __call__(self, image, target=None, mask=None):
        if self.random_size:
            min_size = random.choice(self.random_size)
        else:
            min_size = self.min_size

        # resize
        if self.min_size == self.max_size:
            # donot keep aspect ratio
            img_h0, img_w0 = image.shape[1:]
            image = F.resize(image, size=[min_size, min_size])
        else:
            # keep aspect ratio
            img_h0, img_w0 = image.shape[1:]
            min_original_size = float(min((img_w0, img_h0)))
            max_original_size = float(max((img_w0, img_h0)))

            if max_original_size / min_original_size * min_size > self.max_size:
                min_size = int(round(min_original_size / max_original_size * self.max_size))

            image = F.resize(image, size=min_size, max_size=self.max_size)

        # rescale bboxes
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


# BaseTransforms
class BaseTransforms(object):
    def __init__(self, 
                 min_size=800, 
                 max_size=800, 
                 random_size=None, 
                 pixel_mean=(0.485, 0.456, 0.406), 
                 pixel_std=(0.229, 0.224, 0.225),
                 format='RGB'):
        assert min_size == max_size
        self.min_size = min_size
        self.max_size = max_size
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.format = format
        self.random_size =random_size
        self.transforms = Compose([
            DistortTransform(),
            ToTensor(format=format),
            RandomHorizontalFlip(),
            Resize(min_size=min_size, 
                   max_size=max_size,
                   random_size=random_size),
            Normalize(pixel_mean, pixel_std),
            PadImage(max_size=max_size)
        ])


    def __call__(self, image, target, mask=None):
        return self.transforms(image, target, mask)


# TrainTransform
class TrainTransforms(object):
    def __init__(self, 
                 trans_config=None,
                 min_size=800, 
                 max_size=1333, 
                 random_size=None, 
                 pixel_mean=(0.485, 0.456, 0.406), 
                 pixel_std=(0.229, 0.224, 0.225),
                 format='RGB'):
        self.trans_config = trans_config
        self.min_size = min_size
        self.max_size = max_size
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.format = format
        self.random_size =random_size
        self.transforms = Compose(self.build_transforms(trans_config))


    def build_transforms(self, trans_config):
        transform = []
        for t in trans_config:
            if t['name'] == 'DistortTransform':
                transform.append(DistortTransform())
            elif t['name'] == 'ToTensor':
                transform.append(ToTensor(format=self.format))
            elif t['name'] == 'RandomHorizontalFlip':
                transform.append(RandomHorizontalFlip())
            elif t['name'] == 'RandomShift':
                transform.append(RandomShift(max_shift=t['max_shift']))
            elif t['name'] == 'RandomSizeCrop':
                transform.append(RandomSizeCrop(max_size=self.max_size))
            elif t['name'] == 'Resize':
                transform.append(Resize(min_size=self.min_size, 
                                        max_size=self.max_size, 
                                        random_size=self.random_size))
            elif t['name'] == 'Normalize':
                transform.append(Normalize(pixel_mean=self.pixel_mean,
                                           pixel_std=self.pixel_std))
            elif t['name'] == 'PadImage':
                transform.append(PadImage(max_size=self.max_size))
        
        return transform


    def __call__(self, image, target, mask=None):
        return self.transforms(image, target, mask)


# ValTransform
class ValTransforms(object):
    def __init__(self, 
                 min_size=800, 
                 max_size=1333, 
                 pixel_mean=(0.485, 0.456, 0.406), 
                 pixel_std=(0.229, 0.224, 0.225),
                 format='RGB'):
        self.min_size = min_size
        self.max_size = max_size
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.format = format
        self.transforms = Compose([
            ToTensor(),
            Resize(min_size=min_size, max_size=max_size),
            Normalize(pixel_mean, pixel_std)
        ])


    def __call__(self, image, target=None, mask=None):
        return self.transforms(image, target, mask)
