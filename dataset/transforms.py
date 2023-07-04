from pathlib import Path
import json
import cv2

from albumentations.pytorch import ToTensorV2
from albumentations import (
    HorizontalFlip, Normalize, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, CenterCrop, CoarseDropout,
    ToGray, Downscale, Resize, ToFloat, OneOf, Compose, CropAndPad, RandomCrop
)


def get_transforms(cfg: dict, is_train: bool) -> Compose:
    augs_file = Path("config", "augs", f"{cfg.dataset.augmentation}.json")
    with open(augs_file, 'r') as f:
        pars = json.load(f)
        
    input_size = cfg["dataset"]["resize_size"]
    crop_size = cfg["dataset"]["crop_size"]

    def albu_cdo(v):
        return CoarseDropout(max_holes=pars['coarsedropout']['max_holes'],
                             max_height=pars['coarsedropout']['max_height'],
                             max_width=pars['coarsedropout']['max_width'],
                             min_holes=pars['coarsedropout']['min_holes'],
                             min_height=pars['coarsedropout']['min_height'],
                             min_width=pars['coarsedropout']['min_width'],
                             fill_value=v)

    aug_lst = [Resize(input_size, input_size, interpolation=cv2.INTER_AREA)]
    if is_train:
        train_augs = [HorizontalFlip(),
                      ShiftScaleRotate(shift_limit=pars['shift_scale']['shift_limit'],
                         scale_limit=pars['shift_scale']['scale_limit'], interpolation=pars['interpolation'],
                         rotate_limit=pars['rotate']['rotate_limit'], p=pars['shift_scale']['prob']),
        OneOf([
            CLAHE(clip_limit=pars['contrast']['clahe_clip_limit'],
                  tile_grid_size=pars['contrast']['clahe_grid_size']),
            RandomBrightnessContrast(brightness_limit=pars['contrast']['brightness_limit'],
                                     contrast_limit=pars['contrast']['contrast_limit']),
        ], p=pars['contrast']['prob']),
        HueSaturationValue(hue_shift_limit=pars['hue_saturation']['hue_shift_limit'],
                           sat_shift_limit=pars['hue_saturation']['sat_shift_limit'],
                           val_shift_limit=pars['hue_saturation']['val_shift_limit'],
                           p=pars['hue_saturation']['prob']),
        GaussNoise(var_limit=pars['noise']['var_limit'], p=pars['noise']['prob']),
        OneOf([
              MotionBlur(blur_limit=pars['blur']['limit'], always_apply=True),
              MedianBlur(blur_limit=pars['blur']['limit'], always_apply=True),
              Blur(blur_limit=pars['blur']['limit'], always_apply=True),
          ], p=pars['blur']['prob']),
        Downscale(scale_min=pars['downscale']['scale_min'],
                    scale_max=pars['downscale']['scale_max'] if 'scale_max' in pars['downscale'] else 0.75,
                  p=pars['downscale']['prob']),
        OneOf([albu_cdo(v) for v in range(0, 256, 36)], p=pars['coarsedropout']['prob']),
        ToGray(p=pars['togray'])
                      ]

        if 'crop_and_pad' in pars:
            train_augs.append(
                CropAndPad(percent=(0, pars['crop_and_pad']['cut_rate']),
                     keep_size=False,
                    #  sequential_apply=True,
                     always_apply=False,
                     pad_cval=(0, 255), p=pars['crop_and_pad']['prob']))

        aug_lst.extend(train_augs)

        aug_lst.append(RandomCrop(crop_size, crop_size, p=1))
    else:
        aug_lst.append(CenterCrop(crop_size, crop_size))
    aug_lst.append(Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229]))  # BGR
    aug_lst.append(ToTensorV2())
    # aug_lst.append(ToFloat())
    return Compose(aug_lst)