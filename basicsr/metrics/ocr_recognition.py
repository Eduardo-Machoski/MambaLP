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
    char_predicted: Counter = field(default_factory=Counter)
    confusion_matrix_counts: Counter = field(default_factory=Counter)

    def update(self, targets: Iterable[str], predictions: Iterable[str]) -> None:
        for target, prediction in zip(targets, predictions):
            target = str(target).strip().upper()
            prediction = str(prediction).strip().upper()

            self.total_plates += 1
            self.correct_plates += int(target == prediction)
            self.char_predicted.update(prediction)

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
                self.confusion_matrix_counts[(expected, predicted)] += 1

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
            chars = [prediction[position] for prediction in predictions if position < len(prediction)]

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

    def character_precisions(self) -> dict[str, float]:
        """True positives / predicted occurrences, with positional matching."""
        return {
            char: self.confusion_matrix_counts[(char, char)] / total
            for char, total in sorted(self.char_predicted.items()) if total
        }

    def to_dict(self, alphabet="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", limit=10):
        """Serializable accumulated results; matrix rows=GT, columns=OCR."""
        chars = list(dict.fromkeys(alphabet))
        observed = set(self.char_total) | set(self.char_predicted)
        chars.extend(sorted(observed - set(chars)))
        labels = chars + ["<PAD>"]
        matrix = [[self.confusion_matrix_counts[(gt, pred)] for pred in labels] for gt in labels]
        character_rows = []
        for char in chars:
            correct = self.confusion_matrix_counts[(char, char)]
            total = self.char_total[char]
            predicted = self.char_predicted[char]
            character_rows.append({
                'character': char, 'correct': correct, 'target_total': total,
                'predicted_total': predicted,
                'precision': correct / predicted if predicted else 0.0,
                'accuracy': correct / total if total else 0.0,
            })
        return {
            'total_plates': self.total_plates, 'correct_plates': self.correct_plates,
            'plate_accuracy': self.plate_accuracy,
            'character_accuracy_any_position': self.character_accuracy_any_position,
            'characters': character_rows,
            'positions': [
                {'position': pos + 1, 'correct': self.position_correct[pos],
                 'total': self.position_total[pos], 'accuracy': accuracy}
                for pos, accuracy in self.position_accuracies().items()
            ],
            'most_common_errors': [
                {'character': char, 'errors': errors, 'total': total, 'error_rate': rate}
                for char, errors, total, rate in self.most_error_prone_chars(limit)
            ],
            'most_common_confusions': [
                {'expected': gt, 'predicted': pred, 'count': count}
                for (gt, pred), count in sorted(self.confusion_matrix_counts.items(),
                    key=lambda item: (-item[1], item[0])) if gt != pred
            ][:limit],
            'confusion_matrix': {'labels': labels, 'counts': matrix,
                                 'rows': 'ground_truth', 'columns': 'prediction'},
        }
