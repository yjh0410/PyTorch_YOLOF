"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
import os.path as osp
import torch
import torch.utils.data as data
import cv2
import numpy as np
import random
import xml.etree.ElementTree as ET


VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')


class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt if i % 2 == 0 else cur_pt
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [x1, y1, x2, y2, label_ind]

        return res  # [[x1, y1, x2, y2, label_ind], ... ]


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, 
                 data_dir=None,
                 image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                 transform=None, 
                 target_transform=VOCAnnotationTransform()):
        self.root = data_dir
        self.image_set = image_sets
        self.target_transform = target_transform
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))
        # augmentation
        self.transform = transform


    def __getitem__(self, index):
        image, target, mask= self.pull_item(index)
        return image, target, mask


    def __len__(self):
        return len(self.ids)


    def load_image_target(self, img_id):
        # load an image
        image = cv2.imread(self._imgpath % img_id)
        height, width, channels = image.shape

        # laod an annotation
        anno = ET.parse(self._annopath % img_id).getroot()
        if self.target_transform is not None:
            anno = self.target_transform(anno)

        # check the annotation
        if len(anno) == 0:
            anno = np.zeros([1, 5])
        else:
            anno = np.array(anno)
        
        # target
        target = {
            "boxes": anno[:, :4],
            "labels": anno[:, 4],
            "orig_size": [height, width]
        }
        
        return image, target


    def pull_item(self, index):
        img_id = self.ids[index]
        image, target = self.load_image_target(img_id)
        # augment
        image, target, mask = self.transform(image, target)
            
        return image, target, mask


    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR), img_id


    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt


if __name__ == "__main__":
    from transforms import TrainTransforms, ValTransforms

    rgb_mean = np.array((0.485, 0.456, 0.406), dtype=np.float32)
    rgb_std = np.array((0.229, 0.224, 0.225), dtype=np.float32)

    img_size = 320
    max_size = int(round(1333 / 800 * img_size))
    dataset = VOCDetection(
                data_dir='d:/datasets/VOCdevkit/',
                transform=TrainTransforms(size=img_size, max_size=max_size, random_size=True))
    
    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(20)]
    print('Data length: ', len(dataset))

    for i in range(1000):
        image, target, mask= dataset.pull_item(i)
        # to numpy
        image = image.permute(1, 2, 0).numpy()
        # denormalize
        image = ((image * rgb_std + rgb_mean)*255).astype(np.uint8)
        # to BGR format
        image = image[:, :, (2, 1, 0)]
        image = image.copy()
        img_h, img_w = image.shape[:2]

        boxes = target["boxes"]
        labels = target["labels"]

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            cls_id = int(label)
            color = class_colors[cls_id]
            # class name
            label = VOC_CLASSES[cls_id]
            image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
            # put the test on the bbox
            cv2.putText(image, label, (int(x1), int(y1 - 5)), 0, 0.5, color, 1, lineType=cv2.LINE_AA)
        cv2.imshow('gt', image)
        # cv2.imwrite(str(i)+'.jpg', img)
        cv2.waitKey(0)

        if mask is not None:
            # to numpy
            mask = mask.cpu().numpy()
            mask = (mask * 255).astype(np.uint8).copy()

            boxes = target["boxes"]
            labels = target["labels"]
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box
                cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

            cv2.imshow('mask', mask)
            cv2.waitKey(0)
