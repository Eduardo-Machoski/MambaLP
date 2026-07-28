"""Differentiable OCR loss for license-plate super-resolution.

The pretrained OCR used by ``test_ocr.py`` returns three values; its second
output contains per-position character logits with shape (N, K, C).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class StrLabelConverter:
    """Encode fixed-length plate strings using index 0 as padding/blank."""

    def __init__(self, alphabet: str, ignore_case: bool = False):
        if ignore_case:
            alphabet = alphabet.lower()
        self.ignore_case = ignore_case
        self.alphabet = "-" + alphabet
        self.char_to_index = {char: i for i, char in enumerate(self.alphabet)}

    def encode_list(
        self,
        texts: Sequence[str | bytes],
        max_length: int,
        device: torch.device,
    ) -> Tensor:
        encoded = torch.zeros(
            (len(texts), max_length), dtype=torch.long, device=device
        )

        for row, value in enumerate(texts):
            text = value.decode("utf-8") if isinstance(value, bytes) else value
            text = text.lower() if self.ignore_case else text

            if len(text) > max_length:
                raise ValueError(
                    f"Label {text!r} has {len(text)} characters, but the OCR "
                    f"predicts only {max_length} positions."
                )

            for column, char in enumerate(text):
                if char not in self.char_to_index:
                    raise ValueError(f"Character {char!r} is not in the OCR alphabet.")
                encoded[row, column] = self.char_to_index[char]

        return encoded


class OCRLoss(nn.Module):
    """Character-level loss through a frozen, pretrained OCR network.

    Freezing OCR parameters does not detach the SR image. Consequently,
    gradients still flow OCR(input) -> SR image -> super-resolution network.
    """

    def __init__(
        self,
        ocr_model: nn.Module,
        alphabet: str,
        loss_weight: float = 1.0,
        input_size: tuple[int, int] | None = None,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        ignore_case: bool = False,
        ignore_padding: bool = True,
    ):
        super().__init__()
        self.ocr_model = ocr_model
        self.converter = StrLabelConverter(alphabet, ignore_case)
        self.loss_weight = loss_weight
        self.input_size = input_size
        self.ignore_padding = ignore_padding

        channels = 0 if mean is None else len(mean)
        if (mean is None) != (std is None):
            raise ValueError("mean and std must be provided together.")
        if std is not None and len(std) != channels:
            raise ValueError("mean and std must have the same length.")

        self.register_buffer(
            "mean",
            None if mean is None else torch.tensor(mean).view(1, channels, 1, 1),
        )
        self.register_buffer(
            "std",
            None if std is None else torch.tensor(std).view(1, channels, 1, 1),
        )

        self.ocr_model.requires_grad_(False)
        self.ocr_model.eval()

    def train(self, mode: bool = True):
        # BasicSR calls train() recursively. The pretrained OCR must remain in
        # evaluation mode so Dropout/BatchNorm do not change during SR training.
        super().train(mode)
        self.ocr_model.eval()
        return self

    def _preprocess(self, image: Tensor) -> Tensor:
        if image.ndim != 4:
            raise ValueError(f"Expected image shaped (N,C,H,W), got {image.shape}.")

        image = image.clamp(0.0, 1.0)
        if self.input_size is not None:
            image = F.interpolate(
                image,
                size=self.input_size,
                mode="bilinear",
                align_corners=False,
            )
        if self.mean is not None:
            image = (image - self.mean) / self.std
        return image

    def predict_logits(self, image: Tensor) -> Tensor:
        """Return OCR logits in canonical ``(N, K, C)`` format.

        This method is used both by the differentiable loss and by validation
        metrics, ensuring identical preprocessing and decoding.
        """
        image = self._preprocess(image)
        output = self.ocr_model(image)
        logits = output[1] if isinstance(output, (tuple, list)) else output

        if logits.ndim != 3:
            raise ValueError(
                f"Expected OCR logits shaped (N,K,C), got {logits.shape}."
            )
        if logits.shape[0] != image.shape[0]:
            if logits.shape[1] == image.shape[0]:
                logits = logits.permute(1, 0, 2)
            else:
                raise ValueError(
                    "Could not identify the batch dimension in OCR logits."
                )
        return logits

    @torch.no_grad()
    def predict(self, image: Tensor) -> list[str]:
        """Decode a batch of images into fixed-position plate strings."""
        logits = self.predict_logits(image)
        indices = logits.argmax(dim=-1)
        alphabet = self.converter.alphabet

        predictions = []
        for row in indices:
            # Keep '-' for blank so character positions are never shifted in
            # the per-position validation statistics.
            predictions.append("".join(alphabet[index] for index in row.tolist()))
        return predictions

    def forward(self, sr_image: Tensor, labels: Sequence[str | bytes]) -> Tensor:
        image = self._preprocess(sr_image)

        # Do not use torch.no_grad() here: only OCR weights are frozen. The
        # graph with respect to sr_image is required to train the SR model.
        # ``image`` is already preprocessed here, therefore execute only the
        # model and normalize its output dimensions locally.
        output = self.ocr_model(image)
        logits = output[1] if isinstance(output, (tuple, list)) else output
        if logits.ndim != 3:
            raise ValueError(
                f"Expected OCR logits shaped (N,K,C), got {logits.shape}."
            )
        if logits.shape[0] != image.shape[0]:
            if logits.shape[1] == image.shape[0]:
                logits = logits.permute(1, 0, 2)
            else:
                raise ValueError("Could not identify the batch dimension in OCR logits.")

        batch_size, positions, classes = logits.shape
        expected_classes = len(self.converter.alphabet)
        if classes != expected_classes:
            raise ValueError(
                f"OCR returned {classes} classes, but alphabet requires "
                f"{expected_classes} including padding/blank."
            )
        if len(labels) != batch_size:
            raise ValueError(
                f"Received {len(labels)} labels for a batch of {batch_size} images."
            )

        targets = self.converter.encode_list(labels, positions, logits.device)
        ignore_index = 0 if self.ignore_padding else -100

        loss = F.cross_entropy(
            logits.reshape(-1, classes),
            targets.reshape(-1),
            ignore_index=ignore_index,
        )
        return self.loss_weight * loss


def load_ocr_loss(
    checkpoint_path: str,
    alphabet: str,
    *,
    device: torch.device,
    loss_weight: float = 0.01,
    input_size: tuple[int, int] | None = None,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> OCRLoss:
    """Load the same checkpoint format used by the original test script."""
    # These are project-local modules from the pretrained OCR repository.
    from basicsr.ocr import models as ocr_models

    checkpoint = torch.load(checkpoint_path, map_location=device)
    ocr_model = ocr_models.make(checkpoint["model"], load_model=True).to(device)

    return OCRLoss(
        ocr_model=ocr_model,
        alphabet=alphabet,
        loss_weight=loss_weight,
        input_size=input_size,
        mean=mean,
        std=std,
    ).to(device)
