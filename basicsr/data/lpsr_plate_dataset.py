import random
from pathlib import Path

import cv2
import numpy as np
from torch.utils import data

from basicsr.data.transforms import augment
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

                lq_images = _find_images(
                            folder,
                            "lr",
                )

                gt_images = _find_images(
                            folder,
                            "hr",
                )

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

    def load_sample(self, sample, lq_path, gt_path = None):

        # Load selected images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32., H W 3

        #LQ
        img_bytes = self.file_client.get(lq_path, "lq")
        img_lq = imfrombytes(img_bytes, float32=True)


        if img_lq is None:
            raise FileNotFoundError(
                f"Não foi possível carregar LQ: {lq_path}"
            )

        # Expected LQ size
        h_lq = self.opt["lq_size"]["h"]
        w_lq = self.opt["lq_size"]["w"]

        # LR resize if needed
        if img_lq.shape[0] != h_lq or img_lq.shape[1] != w_lq:
            img_lq = cv2.resize(
                img_lq,
                (w_lq, h_lq),
                interpolation=cv2.INTER_CUBIC
            )

        # GT
        img_gt = None

        if gt_path is not None:
            img_bytes = self.file_client.get(gt_path, "gt")
            img_gt = imfrombytes(img_bytes, float32=True)

            if img_gt is None:
                raise FileNotFoundError(
                    f"Não foi possível carregar GT: {gt_path}"
                )

            # Escala tem que ser fixa para o modelo
            scale = self.opt["scale"]  

            # Tamanho esperado da GT 
            expected_h_gt = h_lq * scale
            expected_w_gt = w_lq * scale

            if img_gt.shape[0] != expected_h_gt or img_gt.shape[1] != expected_w_gt:
                img_gt = cv2.resize(
                    img_gt,
                    (expected_w_gt, expected_h_gt),
                    interpolation=cv2.INTER_CUBIC
                )

            h_lq, w_lq = img_lq.shape[:2]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)

        if img_gt is not None:
            img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)


        return img_lq, img_gt

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        sample = self.samples[index]

        lq_path = self._select_lq(sample)
        gt_path = self._select_best_gt(sample)


        img_lq, img_gt = self.load_sample(sample, lq_path, gt_path)


        return {
            "lq": img_lq,
            "gt": img_gt,
            "label": sample["ocr_gt"],
            "lq_path": lq_path,
            "gt_path": gt_path,
        }

    def __len__(self):
        return len(self.samples)


def _find_images(
        folder: Path,
        prefix: str,
    ) -> list[Path]:

        extensions = (
            "png",
            "jpg",
            "jpeg",
        )

        images = []

        for extension in extensions:
            images.extend(
                folder.glob(
                    f"{prefix}-*.{extension}"
                )
            )

        return sorted(images)
