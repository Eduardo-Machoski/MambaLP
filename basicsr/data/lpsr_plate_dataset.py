import random
from pathlib import Path

import cv2
import numpy as np
from torch.utils import data

from basicsr.data.transforms import paired_random_crop, augment
from basicsr.utils import FileClient, img2tensor, imfrombytes
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class LPSRPlateDataset(data.Dataset):
    """
    Dataset para LPSR baseado na estrutura:

    OCR_GT;PATH;SPLIT

    Exemplo:
        XXX9999;./dataset/plate-000001;training
    """

    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self.phase = opt["phase"]

        phase_map = {
            "train": "training",
            "val": "validation",
            "test": "testing"
        }

        self.phase = phase_map.get(opt["phase"], opt["phase"])
        self.scale = opt["scale"]

        self.dataroot = Path(opt["dataroot"])
        self.index_file = self.dataroot / opt["index_file"]

        self.file_client = None
        self.io_backend_opt = opt["io_backend"]

        self.samples = []

        self._load_index()


    ###########################################################################
    # Construção do dataset
    ###########################################################################

    def _load_index(self):

        with open(self.index_file, "r") as f:

            for line in f:

                line = line.strip()

                if len(line) == 0:
                    continue

                ocr_gt, folder, split = line.split(";")

                if split != self.phase:
                    continue

                folder = self.dataroot / folder

                lq_images = sorted(folder.glob("lr-*.png"))
                gt_images = sorted(folder.glob("hr-*.png"))

                if len(lq_images) == 0 or len(gt_images) == 0:
                    continue

                self.samples.append(
                    {
                        "ocr_gt": ocr_gt,
                        "folder": folder,
                        "lq_candidates": [str(x) for x in lq_images],
                        "gt_candidates": [str(x) for x in gt_images],
                    }
                )


    ###########################################################################
    # Utilidades
    ###########################################################################

    def _select_lq(self, sample):
        return random.choice(sample["lq_candidates"])


    def _select_best_gt(self, sample):
        return random.choice(sample["gt_candidates"])

    ###########################################################################
    # Dataset
    ###########################################################################

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        sample = self.samples[index]

        lq_path = self._select_lq(sample)
        gt_path = self._select_best_gt(sample)


        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32., H W 3

        # LQ
        img_bytes = self.file_client.get(lq_path, "lq")
        img_lq = imfrombytes(img_bytes, float32=True)

        # GT
        img_bytes = self.file_client.get(gt_path, "gt")
        img_gt = imfrombytes(img_bytes, float32=True)

        # Tamanho da LQ
        h_lq, w_lq = img_lq.shape[:2]
        scale = self.opt["scale"]  

        # Tamanho esperado da GT 
        expected_h_gt = h_lq * scale
        expected_w_gt = w_lq * scale

        # Redimensiona apenas a GT se necessário
        if img_gt.shape[0] != expected_h_gt or img_gt.shape[1] != expected_w_gt:
            img_gt = cv2.resize(
                img_gt,
                (expected_w_gt, expected_h_gt),
                interpolation=cv2.INTER_CUBIC
            )

        gt_size = self.opt['gt_size']
        lq_patch_size = gt_size // scale

        if h_lq < lq_patch_size or w_lq < lq_patch_size:

            pad_h = max(0, lq_patch_size - h_lq)
            pad_w = max(0, lq_patch_size - w_lq)

            img_lq = cv2.copyMakeBorder(
                img_lq,
                0, pad_h,
                0, pad_w,
                cv2.BORDER_REFLECT_101
            )

            img_gt = cv2.copyMakeBorder(
                img_gt,
                0, pad_h * scale,
                0, pad_w * scale,
                cv2.BORDER_REFLECT_101
            )

        h_lq, w_lq = img_lq.shape[:2]

        # augmentation for training
        if self.phase == 'training':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])


        h_gt, w_gt = img_gt.shape[:2]
        # Validation
        if self.phase != 'training':
            h = min(h_gt, round(h_lq * scale))
            w = min(w_gt, round(w_lq * scale))
            img_gt = img_gt[:h, :w, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)


        return {
            "lq": img_lq,
            "gt": img_gt,
            "label": sample["ocr_gt"],
            "lq_path": lq_path,
            "gt_path": gt_path,
        }

    def __len__(self):
        return len(self.samples)
