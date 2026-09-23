"""OCR recognition metrics for fixed-format license plates."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class OCRRecognitionMetrics:
    """Accumulate exact-plate and per-character OCR statistics."""

    total_plates: int = 0
    correct_plates: int = 0
    total_characters: int = 0
    correct_characters_any_position: int = 0
    position_total: Counter = field(default_factory=Counter)
    position_correct: Counter = field(default_factory=Counter)
    char_total: Counter = field(default_factory=Counter)
    char_errors: Counter = field(default_factory=Counter)
    confusions: Counter = field(default_factory=Counter)

    def update(self, targets: Iterable[str], predictions: Iterable[str]) -> None:
        for target, prediction in zip(targets, predictions):
            target = str(target).strip().upper()
            prediction = str(prediction).strip().upper()

            self.total_plates += 1
            self.correct_plates += int(target == prediction)

            # Compare each plate as a multiset: repeated characters count only
            # as often as they occur in both GT and prediction. Keep plates
            # separate so characters from another plate cannot match this one.
            # The longer length counts insertions as errors too.
            self.total_characters += max(len(target), len(prediction))
            self.correct_characters_any_position += sum(
                (Counter(target) & Counter(prediction)).values()
            )

            max_len = max(len(target), len(prediction))
            for position in range(max_len):
                expected = target[position] if position < len(target) else "<PAD>"
                predicted = prediction[position] if position < len(prediction) else "<PAD>"

                # Position accuracy is evaluated only for real GT characters.
                if position < len(target):
                    self.position_total[position] += 1
                    self.char_total[expected] += 1
                    if expected == predicted:
                        self.position_correct[position] += 1
                    else:
                        self.char_errors[expected] += 1
                        self.confusions[(expected, predicted)] += 1


    def majority_vote(self, predictions: Iterable[str]) -> str:
        predictions = [str(prediction).strip().upper() for prediction in predictions]

        if not predictions:
            return ""

        max_len = max(len(prediction) for prediction in predictions)

        result = []

        for position in range(max_len):
            chars = [prediction[position] for prediction in predictions]

            if chars:
                most_common_char = Counter(chars).most_common(1)[0][0]
                result.append(most_common_char)

        return "".join(result)


    @property
    def plate_accuracy(self) -> float:
        if self.total_plates == 0:
            return 0.0
        return self.correct_plates / self.total_plates

    @property
    def character_accuracy_any_position(self) -> float:
        if self.total_characters == 0:
            return 0.0
        return self.correct_characters_any_position / self.total_characters

    def position_accuracies(self) -> dict[int, float]:
        return {
            position: self.position_correct[position] / total
            for position, total in sorted(self.position_total.items())
            if total > 0
        }

    def most_error_prone_chars(self, limit: int = 10) -> list[tuple[str, int, int, float]]:
        rows = []
        for char, errors in self.char_errors.items():
            total = self.char_total[char]
            error_rate = errors / total if total else 0.0
            rows.append((char, errors, total, error_rate))
        return sorted(rows, key=lambda row: (-row[1], -row[3], row[0]))[:limit]

    def most_common_confusions(self, limit: int = 10) -> list[tuple[str, str, int]]:
        return [
            (expected, predicted, count)
            for (expected, predicted), count in self.confusions.most_common(limit)
        ]
