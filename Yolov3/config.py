import albumentations as A
import cv2
import torch

from albumentations.pytorch import ToTensorV2
from utils import seed_everything

DATASET = 'BDD100K'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
#seed_everything(SEED)
NUM_WORKERS = 4
BATCH_SIZE = 32
IMAGE_SIZE = 416
NUM_CLASSES = 3
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 100
CONF_THRESHOLD = 0.6
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45
S = [IMAGE_SIZE//32, IMAGE_SIZE//16, IMAGE_SIZE//8]
PIN_MEMORY = True
LOAD_MODEL = False
SAVE_MODEL = False
CHECKPOINT_FILE = "checkpoint.pth.tar"
IMG_DIR = DATASET + "/images/"
LABEL_DIR = DATASET + "/labels/"

ANCHORS = [[(116, 90), (156, 198), (373, 326)], [(30, 61), (62, 45), (59, 119)], [(10, 13), (16, 30), (33, 23)]]
scale=1
train_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE*scale, 
            min_width=IMAGE_SIZE*scale, 
            border_mode=cv2.BORDER_CONSTANT
            ),
        A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.3),
        A.OneOf(
            [
                A.ShiftScaleRotate(
                    rotate_limit=10, p=0.3, border_mode=cv2.BORDER_CONSTANT
                ),
                A.Affine(shear=15, p=0.5, mode="constant"),
            ],
            p=0.3,
        ),
        A.HorizontalFlip(p=0.5),
        A.Blur(blur_limit=3, p=0.3),
        A.CLAHE(p=0.3),
        A.Posterize(p=0.3),
        A.ToGray(p=0.3),
        A.ChannelShuffle(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], max_pixel_value=255, p=1.0),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),
)
test_transform = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE*scale, 
            min_width=IMAGE_SIZE*scale, 
            border_mode=cv2.BORDER_CONSTANT
            ),
        A.Normalize(mean=[0.485, 0.456, 0.406], max_pixel_value=255, p=1.0),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)


BDD10K_Classes = [
    "person",
    "rider",
    "car"
]