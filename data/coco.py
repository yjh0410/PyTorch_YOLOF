import os
import numpy as np

from torch.utils.data import Dataset
import cv2

try:
    from pycocotools.coco import COCO
except:
    print("It seems that the COCOAPI is not installed.")


coco_class_labels = ('background',
                        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                        'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign',
                        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                        'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella',
                        'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
                        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                        'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk',
                        'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book',
                        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

coco_class_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
                    21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67,
                    70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]


class COCODataset(Dataset):
    """
    COCO dataset class.
    """
    def __init__(self, 
                 data_dir=None, 
                 image_set='train2017',
                 transform=None):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            debug (bool): if True, only one data id is selected from the dataset
        """
        if image_set == 'train2017':
            self.json_file='instances_train2017.json'
        elif image_set == 'val2017':
            self.json_file='instances_val2017.json'
        elif image_set == 'test2017':
            self.json_file='image_info_test-dev2017.json'
        self.image_set = image_set
        self.data_dir = data_dir
        self.coco = COCO(os.path.join(self.data_dir, 'annotations', self.json_file))
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        # augmentation
        self.transform = transform


    def __len__(self):
        return len(self.ids)


    def __getitem__(self, index):
        image, target, mask = self.pull_item(index)
        return image, target, mask


    def load_image_target(self, index):
        anno_ids = self.coco.getAnnIds(imgIds=[int(index)], iscrowd=None)
        annotations = self.coco.loadAnns(anno_ids)

        # load an image
        img_file = os.path.join(self.data_dir, self.image_set,
                                '{:012}'.format(index) + '.jpg')
        image = cv2.imread(img_file)
        
        if self.json_file == 'instances_val5k.json' and image is None:
            img_file = os.path.join(self.data_dir, 'train2017',
                                    '{:012}'.format(index) + '.jpg')
            image = cv2.imread(img_file)

        assert image is not None

        height, width, channels = image.shape
        
        #load a target
        anno = []
        for label in annotations:
            if 'bbox' in label and label['area'] > 0:   
                xmin = np.max((0, label['bbox'][0]))
                ymin = np.max((0, label['bbox'][1]))
                xmax = np.min((width - 1, xmin + np.max((0, label['bbox'][2] - 1))))
                ymax = np.min((height - 1, ymin + np.max((0, label['bbox'][3] - 1))))
                if xmax > xmin and ymax > ymin:
                    label_ind = label['category_id']
                    cls_id = self.class_ids.index(label_ind)

                    anno.append([xmin, ymin, xmax, ymax, cls_id])  # [xmin, ymin, xmax, ymax, label_ind]
            else:
                print('No bbox !!!')

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
        id_ = self.ids[index]
        image, target = self.load_image_target(id_)
        # augment
        image, target, mask = self.transform(image, target)
        
        return image, target, mask


    def pull_image(self, index):
        id_ = self.ids[index]
        img_file = os.path.join(self.data_dir, self.image_set,
                                '{:012}'.format(id_) + '.jpg')
        img = cv2.imread(img_file)

        if self.json_file == 'instances_val5k.json' and img is None:
            img_file = os.path.join(self.data_dir, 'train2017',
                                    '{:012}'.format(id_) + '.jpg')
            img = cv2.imread(img_file)

        return img, id_


    def pull_anno(self, index):
        id_ = self.ids[index]

        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=None)
        annotations = self.coco.loadAnns(anno_ids)
        
        anno = []
        for label in annotations:
            if 'bbox' in label:
                xmin = np.max((0, label['bbox'][0]))
                ymin = np.max((0, label['bbox'][1]))
                xmax = xmin + label['bbox'][2]
                ymax = ymin + label['bbox'][3]
                
                if label['area'] > 0 and xmax >= xmin and ymax >= ymin:
                    label_ind = label['category_id']
                    cls_id = self.class_ids.index(label_ind)

                    anno.append([xmin, ymin, xmax, ymax, cls_id])  # [xmin, ymin, xmax, ymax, label_ind]
            else:
                print('No bbox !!')
        return anno


if __name__ == "__main__":
    from transforms import TrainTransforms, ValTransforms

    rgb_mean = np.array((0.485, 0.456, 0.406), dtype=np.float32)
    rgb_std = np.array((0.229, 0.224, 0.225), dtype=np.float32)

    img_size = 800
    dataset = COCODataset(
                data_dir='/mnt/share/ssd2/dataset/COCO/',
                image_set='train2017',
                transform=TrainTransforms(img_size))
    
    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(80)]
    print('Data length: ', len(dataset))

    for i in range(1000):
        image, target, mask = dataset.pull_item(i)
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
            label = coco_class_labels[coco_class_index[cls_id]]
            # bbox
            x1 *= img_w
            y1 *= img_h
            x2 *= img_w
            y2 *= img_h
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
