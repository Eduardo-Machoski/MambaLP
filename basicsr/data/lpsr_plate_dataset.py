import csv
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from torch.utils import data

from basicsr.data.transforms import augment
from basicsr.utils import FileClient, img2tensor, imfrombytes, get_root_logger
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
        self.require_hr = self.phase != "testing"
        self.single_lr_per_track = bool(opt.get("single_lr_per_track", self.phase == "testing"))
        # The execution phase is set by parse_options; select CSV rows separately.
        self.split = _normalize_split(opt.get("split", self.phase))
        self.scale = opt["scale"]

        self.dataroot = Path(opt["dataroot"]).expanduser()
        self.index_file = self.dataroot / opt["index_file"]

        self.file_client = None
        self.io_backend_opt = opt["io_backend"]

        self.samples = []

        self._load_index()



    ###########################################################################
    # Construção do dataset
    ###########################################################################

    def _load_index(self):
        splits = Counter()
        rejected = Counter()
        tracks_without_hr = 0
        examples = []
        with open(self.index_file, "r", encoding="utf-8-sig", newline="") as f:
            for line_number, row in enumerate(csv.reader(f, delimiter=";"), 1):
                if not row or not any(value.strip() for value in row):
                    continue
                if len(row) != 3:
                    raise ValueError(f"{self.index_file}:{line_number}: expected OCR_GT;PATH;SPLIT, got {len(row)} columns")
                ocr_gt, folder, split = (value.strip() for value in row)
                if [value.upper() for value in (ocr_gt, folder, split)] == ["OCR_GT", "PATH", "SPLIT"]:
                    continue
                split = _normalize_split(split)
                splits[split] += 1
                if split != self.split:
                    continue
                folder = self.dataroot / Path(folder).expanduser()
                if not folder.is_dir():
                    rejected["missing_folder"] += 1
                    if len(examples) < 5:
                        examples.append(f"missing_folder: {folder.resolve()}")
                    continue
                lq_images = _find_images(folder, "lr")
                gt_images = _find_images(folder, "hr")
                if not lq_images or (self.require_hr and not gt_images):
                    if not lq_images:
                        rejected["missing_lr"] += 1
                    if self.require_hr and not gt_images:
                        rejected["missing_hr"] += 1
                    if len(examples) < 5:
                        examples.append(f"LR={len(lq_images)}, HR={len(gt_images)}: {folder.resolve()}")
                    continue
                tracks_without_hr += int(not gt_images)
                self.samples.append({
                    "ocr_gt": ocr_gt,
                    "folder": folder,
                    "lq_candidates": [str(x) for x in lq_images],
                    "gt_candidates": [str(x) for x in gt_images],
                })
        details = (
            f"index={self.index_file.resolve()}; selected_split={self.split!r}; "
            f"rows_by_split={dict(splits)}; selected_rows={splits[self.split]}; "
            f"loaded_tracks={len(self.samples)}; tracks_without_hr={tracks_without_hr}; "
            f"require_hr={self.require_hr}; rejected={dict(rejected)}; "
            f"examples={examples}"
        )
        get_root_logger().info('LPSR index: %s', details)
        if not self.samples:
            raise ValueError(
                "Empty LPSR dataset: " + details + ". "
                "Set datasets.test.split to the intended CSV split. "
                "Relative PATH values are resolved against dataroot. "
                "Each folder needs LR images; HR is required only for train/val "
                "(lr-/lr_ and hr-/hr_; png/jpg/jpeg)."
            )


    ###########################################################################
    # Utilidades
    ###########################################################################

    def _select_lq(self, sample):
        if self.single_lr_per_track:
            return sample["lq_candidates"][0]
        return random.choice(sample["lq_candidates"])


    def _select_best_gt(self, sample):
        if self.single_lr_per_track and sample["gt_candidates"]:
            return sample["gt_candidates"][0]
        return random.choice(sample["gt_candidates"]) if sample["gt_candidates"] else None

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


        result = {
            "lq": img_lq,
            "label": sample["ocr_gt"],
            "lq_path": lq_path,
        }
        # Omit missing values so PyTorch default_collate can build LR-only batches.
        if img_gt is not None:
            result.update(gt=img_gt, gt_path=gt_path)
        return result

    def __len__(self):
        return len(self.samples)


def _normalize_split(value):
    value = str(value).strip().lower()
    return {"train": "training", "val": "validation", "valid": "validation",
            "test": "testing"}.get(value, value)


def _find_images(
        folder: Path,
        prefix: str,
    ) -> list[Path]:

    return sorted(
        image for image in folder.iterdir()
        if image.is_file()
        and image.suffix.lower() in {".png", ".jpg", ".jpeg"}
        and image.name.lower().startswith((prefix.lower() + "-", prefix.lower() + "_"))
    )
