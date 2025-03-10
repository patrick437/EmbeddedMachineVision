from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pandas as pd
import torch
import utils

ImageFile.LOAD_TRUNCATED_IMAGES = True


class YOLODataset(Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,    # Directory where images are stored
        label_dir,  # Directory where labels are stored 
        anchors,    # Anchors for the dataset   
        image_size=416,
        S=[13, 26, 52], # Number of grid points
        C=20,          # Number of classes  
        transform=None
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2]) ##for al 3 scales
        self.image_size = image_size
        self.S = S
        self.transform = transform
        self.num_anchors = self.anchors
        self.num_anchores_per_scale = self.num_anchors/3
        self.C = C
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=",", ndim=2), 4, axis=1).tolist()
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path))

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]
                            
        targets = [torch.zeros((self.num_anchors//3, S, S, 6)) for S in self.S]     ##6 for [c, x, y, w, h, 1]

        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False, False, False]

            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx//self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S*y), int(S*x)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 4]

                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = class_label
                    targets[scale_idx][anchor_on_scale, i, j, 1:4] = torch.tensor([x*S - j, y*S - i, width, height])
                    targets[scale_idx][anchor_on_scale, i, j, 4] = 1
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  ##ignore this prediction
                    targets[scale_idx][anchor_on_scale, i, j, 1:4] = torch.tensor([x*S - j, y*S - i, width, height])
                    targets[scale_idx][anchor_on_scale, i, j, 4] = 1