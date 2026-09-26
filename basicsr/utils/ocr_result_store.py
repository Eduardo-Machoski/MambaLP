"""Portable per-SR review artifacts, grouped by positional OCR matches."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def compare_characters(target, prediction):
    target = str(target).strip().upper()
    prediction = str(prediction).strip().upper()
    positions = []
    for index in range(max(len(target), len(prediction))):
        expected = target[index] if index < len(target) else ''
        predicted = prediction[index] if index < len(prediction) else ''
        positions.append({'position': index + 1, 'expected': expected,
                          'predicted': predicted, 'correct': expected == predicted})
    correct = sum(item['correct'] for item in positions)
    return {'target': target, 'prediction': prediction, 'positions': positions,
            'correct_characters': correct, 'total_positions': len(positions),
            'errors': len(positions) - correct, 'plate_correct': target == prediction}


def find_hr_references(gt_path, candidates=None):
    if candidates:
        return sorted({Path(path).expanduser().resolve() for path in candidates})
    if not gt_path:
        return []
    gt = Path(gt_path).expanduser().resolve()
    paths = [p for p in gt.parent.iterdir() if p.is_file()
             and p.suffix.lower() in IMAGE_SUFFIXES
             and p.stem.lower().startswith(('hr-', 'hr_'))]
    return sorted(set(paths + [gt]))


class OCRResultStore:
    def __init__(self, root, dataset_name):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.dataset_name = dataset_name
        self.counts = Counter()
        self.count = 0
        self.references = {}
        # A previous review archive must not be silently mixed with a new run.
        self.manifest_path = self.root / 'manifest.jsonl'
        self.manifest = self.manifest_path.open('x', encoding='utf-8')

    def _copy_hr(self, path):
        path = Path(path).resolve()
        if path not in self.references:
            folder_key = sha256(str(path.parent).encode()).hexdigest()[:16]
            relative = Path('references') / folder_key / path.name
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            self.references[path] = relative.as_posix()
        return self.references[path]

    def save(self, sr_rgb, lr_rgb, target, prediction, lq_path, gt_path=None, hr_candidates=None):
        comparison = compare_characters(target, prediction)
        lq_path = Path(lq_path).expanduser().resolve()
        key = sha256(str(lq_path).encode()).hexdigest()[:12]
        relative = Path(f"{comparison['correct_characters']}_acertos") / f'{self.count + 1:06d}_{key}'
        output = self.root / relative
        output.mkdir(parents=True, exist_ok=False)
        refs = find_hr_references(gt_path, hr_candidates)
        record = {
            'format': 'ocr_sr_case_v1', 'dataset': self.dataset_name,
            'id': relative.name, **comparison,
            'sr_image': (relative / 'sr.png').as_posix(),
            'lr_image': (relative / 'lr.png').as_posix(),
            'hr_images': [self._copy_hr(p) for p in refs],
            'original_lq_path': str(lq_path),
            'original_gt_path': str(Path(gt_path).resolve()) if gt_path else None,
            'original_hr_paths': [str(p) for p in refs],
            'ocr_scope': 'individual_sr',
            'lr_kind': 'actual_model_input',
        }
        Image.fromarray(np.asarray(sr_rgb, dtype=np.uint8)).save(output / 'sr.png')
        Image.fromarray(np.asarray(lr_rgb, dtype=np.uint8)).save(output / 'lr.png')
        (output / 'result.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        # Only expose complete records to a viewer reading during the test.
        self.manifest.write(json.dumps(record, ensure_ascii=False) + '\n')
        self.manifest.flush()
        self.count += 1
        self.counts[comparison['correct_characters']] += 1
        return record

    def summary(self):
        return {'root': str(self.root), 'manifest': str(self.manifest_path),
                'images_saved': self.count,
                'by_correct_characters': dict(sorted(self.counts.items())),
                'criterion': 'positional matches for each individual SR'}

    def close(self):
        self.manifest.close()
