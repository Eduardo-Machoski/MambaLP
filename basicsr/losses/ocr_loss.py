"""HR-aware differentiable OCR loss for license-plate super-resolution.

The pretrained OCR used by ``test_ocr.py`` returns three values; its second
output contains per-position character logits with shape (N, K, C). OCR
predictions from SR and HR determine a configurable weight for each position,
while cross-entropy over SR logits supplies the training gradient.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


CASE_NAMES = (
    "both_correct",
    "sr_correct_hr_wrong",
    "hr_correct_sr_wrong",
    "both_wrong",
)

DEFAULT_CASE_WEIGHTS = {
    "both_correct": 0.0,
    "sr_correct_hr_wrong": 1.0,
    "hr_correct_sr_wrong": 2.0,
    "both_wrong": 0.1,
}


def _normalize_case_weights(
    case_weights: Mapping[str | int, float] | Sequence[float] | None,
) -> Tensor:
    """Convert YAML-friendly case weights to a tensor ordered from case 0 to 3."""
    values = dict(DEFAULT_CASE_WEIGHTS)

    if case_weights is not None:
        if isinstance(case_weights, Mapping):
            aliases: dict[str | int, str] = {
                index: name for index, name in enumerate(CASE_NAMES)
            }
            aliases.update({str(index): name for index, name in enumerate(CASE_NAMES)})
            aliases.update(
                {f"case_{index}": name for index, name in enumerate(CASE_NAMES)}
            )
            aliases.update({name: name for name in CASE_NAMES})

            for key, value in case_weights.items():
                if key not in aliases:
                    valid_keys = ", ".join(CASE_NAMES)
                    raise ValueError(
                        f"Unknown OCR-loss case weight {key!r}. Expected one of: "
                        f"{valid_keys}, or case IDs 0 to 3."
                    )
                values[aliases[key]] = float(value)
        elif isinstance(case_weights, Sequence) and not isinstance(
            case_weights, (str, bytes)
        ):
            if len(case_weights) != len(CASE_NAMES):
                raise ValueError(
                    "case_weights must contain exactly four values ordered by "
                    "case IDs 0, 1, 2 and 3."
                )
            values = {
                name: float(value)
                for name, value in zip(CASE_NAMES, case_weights, strict=True)
            }
        else:
            raise TypeError(
                "case_weights must be a mapping or a sequence of 4 values."
            )

    for name, value in values.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"Weight {name!r} must be a finite, non-negative number, got {value}."
            )

    return torch.tensor([values[name] for name in CASE_NAMES], dtype=torch.float32)


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
    """HR-aware character loss through a frozen, pretrained OCR network.

    Freezing OCR parameters does not detach the SR image. Consequently,
    gradients still flow OCR(input) -> SR image -> super-resolution network.

    Each plate position receives one of four configurable weights:

    0. both OCR predictions match the label;
    1. SR matches the label, but HR does not;
    2. HR matches the label, but SR does not;
    3. neither OCR prediction matches the label.
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
        case_weights: Mapping[str | int, float] | Sequence[float] | None = None,
    ):
        super().__init__()
        self.ocr_model = ocr_model
        self.converter = StrLabelConverter(alphabet, ignore_case)
        self.loss_weight = loss_weight
        self.input_size = input_size
        self.ignore_padding = ignore_padding
        self.register_buffer("case_weights", _normalize_case_weights(case_weights))

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

        self.ocr_model.eval()
        for parameter in self.ocr_model.parameters():
            parameter.requires_grad_(False)

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
        ocr_parameter = next(self.ocr_model.parameters())

        image = image.to(
            device=ocr_parameter.device,
            dtype=ocr_parameter.dtype,
            non_blocking=True,
        )

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

    def forward(
        self,
        sr_image: Tensor,
        hr_image: Tensor,
        labels: Sequence[str | bytes],
    ) -> Tensor:
        """Calculate the weighted per-position loss for SR against its HR source."""
        # Do not use torch.no_grad() for SR: only OCR weights are frozen. The
        # graph with respect to sr_image is required to train the SR model.
        sr_logits = self.predict_logits(sr_image)

        # HR is used only to select the reliability case for each position. It
        # is a fixed target image, so retaining this graph would waste memory.
        with torch.no_grad():
            hr_logits = self.predict_logits(hr_image)

        if hr_logits.shape != sr_logits.shape:
            raise ValueError(
                "OCR returned different shapes for SR and HR: "
                f"{sr_logits.shape} versus {hr_logits.shape}."
            )

        batch_size, positions, classes = sr_logits.shape
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

        targets = self.converter.encode_list(labels, positions, sr_logits.device)
        ignore_index = 0 if self.ignore_padding else -100

        position_losses = F.cross_entropy(
            sr_logits.reshape(-1, classes),
            targets.reshape(-1),
            ignore_index=ignore_index,
            reduction="none",
        ).view(batch_size, positions)

        sr_correct = sr_logits.detach().argmax(dim=-1).eq(targets)
        hr_correct = hr_logits.argmax(dim=-1).eq(targets)

        # Case 3 is the fallback; the other masks overwrite it as needed.
        case_ids = torch.full_like(targets, 3)
        case_ids.masked_fill_(sr_correct & hr_correct, 0)
        case_ids.masked_fill_(sr_correct & ~hr_correct, 1)
        case_ids.masked_fill_(~sr_correct & hr_correct, 2)

        position_weights = self.case_weights[case_ids].to(position_losses.dtype)
        if self.ignore_padding:
            valid_positions = targets.ne(0)
        else:
            valid_positions = torch.ones_like(targets, dtype=torch.bool)

        weighted_sum = (
            position_losses * position_weights * valid_positions
        ).sum()
        valid_count = valid_positions.sum().clamp_min(1).to(position_losses.dtype)
        return self.loss_weight * weighted_sum / valid_count

    def cross_entropy_loss(
        self,
        sr_image: torch.Tensor,
        labels,
    ) -> torch.Tensor:
        """Calculate differentiable OCR cross-entropy loss.

        The OCR parameters remain frozen, but gradients flow from the OCR
        logits to the SR image and consequently to the SR network.
        """
        sr_logits = self.predict_logits(sr_image)

        # Canonical shape:
        # sr_logits: (batch_size, num_positions, num_classes)
        batch_size, num_positions, num_classes = sr_logits.shape

        if isinstance(labels, str):
            labels = [labels]

        labels = [
            str(label).strip().upper()
            for label in labels
        ]

        targets = self.converter.encode_list(
            labels,
            K=num_positions,
        ).to(
            device=sr_logits.device,
            dtype=torch.long,
        )

        expected_shape = (batch_size, num_positions)

        if tuple(targets.shape) != expected_shape:
            raise ValueError(
                "OCR target shape does not match logits. "
                f"Expected {expected_shape}, got {tuple(targets.shape)}."
            )

        loss = F.cross_entropy(
            sr_logits.reshape(-1, num_classes),
            targets.reshape(-1),
            ignore_index=0,
            reduction="mean",
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
    case_weights: Mapping[str | int, float] | Sequence[float] | None = None,
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
        case_weights=case_weights,
    ).to(device)
