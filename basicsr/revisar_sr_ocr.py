"""Revisão manual de SR no BasicSR: LR + cinco HR + SR e filtro por OCR.

Coloque este arquivo na raiz do projeto ou dentro de ``basicsr``.
Execute ``python3 basicsr/revisar_sr_ocr.py --opt options/test/mambair/test_MambaIR_SR_REAL_OCR_review.yml``.
O YAML de teste fornece modelo SR, split de treino e OCR.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
import torch
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == 'basicsr' else SCRIPT_DIR
sys.path.insert(0, str(PROJECT_ROOT))

from basicsr.data import build_dataset
from basicsr.models import build_model
from basicsr.utils import get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options


IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff'}
LABEL_KEYS = ('plate', 'plate_text', 'license_plate', 'lp_text', 'label', 'text', 'gt_label', 'gt')


def plate_text(value):
    """Remove separadores e uniformiza a comparação entre OCR e anotação."""
    return ''.join(c for c in str(value).upper() if c.isalnum())


def character_errors(target, prediction):
    """Conta posições erradas como OCRRecognitionMetrics no treinamento."""
    target, prediction = target.strip().upper(), prediction.strip().upper()
    return sum(a != b for a, b in zip(target, prediction)) + abs(len(target) - len(prediction))


def colored_ocr_positions(target, prediction):
    """Texto e acerto em cada posição; ∅ marca caractere ausente na predição."""
    return [
        (prediction[i] if i < len(prediction) else '∅',
         i < len(target) and i < len(prediction) and prediction[i] == target[i])
        for i in range(max(len(target), len(prediction)))
    ]


def valid_sr_label(value, gt_label):
    """O rótulo deve ter o tamanho da placa GT e apenas letras/dígitos ASCII."""
    return bool(gt_label) and len(value) == len(gt_label) and value.isascii() and value.isalnum()


def scalar_string(value):
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return scalar_string(value[0])
    return None


def label_from_json(data, explicit_key=None):
    if explicit_key:
        for part in explicit_key.split('.'):
            if not isinstance(data, dict) or part not in data:
                return None
            data = data[part]
        return scalar_string(data)
    if isinstance(data, dict):
        for key in LABEL_KEYS:
            value = scalar_string(data.get(key))
            if value:
                return value
        for value in data.values():
            if isinstance(value, dict):
                found = label_from_json(value)
                if found:
                    return found
    return None


def get_label(sample, gt_path, label_key):
    keys = (label_key,) if label_key else LABEL_KEYS
    for key in keys:
        value = scalar_string(sample.get(key))
        if value:
            return plate_text(value)
    # Na UFPR-SR-Plates, a anotação pode acompanhar a imagem em JSON.
    json_path = gt_path.with_suffix('.json')
    if json_path.is_file():
        with json_path.open(encoding='utf-8') as stream:
            value = label_from_json(json.load(stream), label_key)
        if value:
            return plate_text(value)
    raise ValueError(
        f'GT não encontrado para {gt_path}. Informe --label-key ou adapte get_label() '
        'para o formato das anotações do seu dataset.'
    )


def track_folder(lq_path, gt_path, levels):
    if levels is not None:
        return gt_path.parents[levels - 1]
    if lq_path.parent == gt_path.parent:
        return gt_path.parent
    if lq_path.parent.parent == gt_path.parent.parent:
        return gt_path.parent.parent
    # Ex.: /dataset/LR/trilha/1.png e /dataset/HR/trilha/1.png.
    return gt_path.parent


def hr_images(track, gt_path, pattern):
    if pattern:
        files = sorted(p for p in track.glob(pattern)
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    else:
        siblings = sorted(p for p in gt_path.parent.iterdir()
                          if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                          and not p.stem.lower().startswith('mamba_sr'))
        if len(siblings) == 5:
            files = siblings
        else:
            files = sorted(p for p in track.rglob('*') if p.is_file()
                           and p.suffix.lower() in IMAGE_EXTENSIONS
                           and 'hr' in p.relative_to(track).as_posix().lower()
                           and not p.stem.lower().startswith('mamba_sr'))
    if len(files) != 5 or gt_path not in files:
        raise ValueError(
            f'Não foi possível identificar as cinco HR de {gt_path} '
            f'(encontradas: {len(files)}). Use --hr-glob para indicar o padrão.'
        )
    return files


def already_saved(track):
    return any(p.is_file() and p.stem.lower() == 'mamba_sr' and
               p.suffix.lower() in IMAGE_EXTENSIONS for p in track.iterdir())


def saved_sr_tracks(records):
    """Conta cada trilha do split uma vez, mesmo com várias LR ou extensões."""
    tracks = {record[3] for record in records}
    return {track for track in tracks if already_saved(track)}


def training_records(dataset, levels, label_key):
    """Expande as LR do split train sem executar transforms de treinamento.

    LPSRPlateDataset.samples guarda os caminhos originais e o rótulo da trilha.
    Ao usar esses caminhos, a SR salva corresponde exatamente à LR mostrada.
    """
    samples = getattr(dataset, 'samples', None)
    if samples is None:
        raise ValueError(
            'O dataset de treino precisa expor samples com lq_candidates, '
            'gt_candidates e ocr_gt para revisão sem recortes aleatórios.'
        )
    records = []
    seen = set()
    for sample in samples:
        lqs = sample.get('lq_candidates')
        gts = sample.get('gt_candidates')
        if not lqs or not gts:
            raise ValueError('Amostra de treino sem lq_candidates ou gt_candidates.')
        gt_paths = sorted({Path(path).expanduser().resolve() for path in gts})
        gt_path = gt_paths[0]
        raw_label = scalar_string(sample.get('ocr_gt'))
        label = plate_text(raw_label) if raw_label and not label_key else get_label(sample, gt_path, label_key)
        for lq in lqs:
            lq_path = Path(lq).expanduser().resolve()
            track = track_folder(lq_path, gt_path, levels)
            declared_folder = sample.get('folder')
            if declared_folder:
                folder_path = Path(declared_folder).expanduser().resolve()
                if folder_path.is_dir() and gt_path.is_relative_to(folder_path):
                    track = folder_path
            key = (lq_path, track)
            if key not in seen:
                seen.add(key)
                records.append((lq_path, gt_path, gt_paths, track, label))
    return records


def load_rgb(path):
    with Image.open(path) as img:
        return img.convert('RGB')


def training_lr(path, lq_size):
    """Repete a leitura BGR, resize bicúbico e conversão RGB do dataset."""
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Não foi possível ler a LR: {path}')
    image = image.astype(np.float32) / 255.0
    if lq_size is not None:
        height, width = int(lq_size['h']), int(lq_size['w'])
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    shown = Image.fromarray(np.uint8(np.round(np.clip(rgb, 0, 1) * 255)))
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0)
    return shown, tensor


def training_hr(path, height, width):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Não foi possível ler a HR: {path}')
    image = image.astype(np.float32) / 255.0
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0)


def tensor_rgb(tensor):
    if isinstance(tensor, (list, tuple)):
        tensor = tensor[0]
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim != 3 or tensor.shape[0] not in (1, 3):
        raise ValueError(f'Tensor de imagem inesperado: {tuple(tensor.shape)}')
    array = tensor.detach().float().cpu().clamp(0, 1).numpy()
    if array.shape[0] == 1:
        array = np.repeat(array, 3, axis=0)
    return (np.transpose(array, (1, 2, 0)) * 255).round().astype(np.uint8)


def load_training_ocr(ocr_opt, device):
    """Reproduz train.ocr_opt e o predict usado por model.validation()."""
    from basicsr.losses.ocr_loss import load_ocr_loss

    input_size = ocr_opt.get('input_size')
    if input_size is not None:
        input_size = tuple(input_size)
    return load_ocr_loss(
        checkpoint_path=os.path.expanduser(ocr_opt['load']),
        alphabet=ocr_opt['alphabet'],
        device=device,
        loss_weight=ocr_opt.get('loss_weight', 0.01),
        input_size=input_size,
        mean=ocr_opt.get('mean'),
        std=ocr_opt.get('std'),
    )


def save_exclusive(image, track, extension):
    """Cria o arquivo sem substituir uma SR existente."""
    if already_saved(track):
        return False
    target = track / f'mamba_sr.{extension}'
    fd, temp_name = tempfile.mkstemp(prefix='.mamba_sr_', suffix=f'.{extension}', dir=track)
    try:
        with os.fdopen(fd, 'wb') as stream:
            image.save(stream, format={'jpg': 'JPEG', 'jpeg': 'JPEG'}.get(extension, 'PNG'))
        os.link(temp_name, target)  # falha se outro processo criou o destino
        return True
    except FileExistsError:
        return False
    finally:
        os.unlink(temp_name)


def save_labeled_sr(image, track, extension, plate_label, gt_label):
    """Salva a SR e seu rótulo revisado no annotations.json da trilha."""
    plate_label = plate_label.strip().upper()
    if not valid_sr_label(plate_label, gt_label):
        raise ValueError(f'Informe uma placa com {len(gt_label)} letras/dígitos, sem espaços ou hífens.')
    annotation_path = track / 'annotations.json'
    with annotation_path.open(encoding='utf-8') as stream:
        annotation = json.load(stream)
    if not isinstance(annotation, dict):
        raise ValueError(f'A anotação deve ser um objeto JSON: {annotation_path}')

    filename = f'mamba_sr.{extension}'
    existing_labels = annotation.get('SR_plate_text', {})
    if not isinstance(existing_labels, dict):
        raise ValueError(f'SR_plate_text precisa ser um objeto JSON: {annotation_path}')
    if filename in existing_labels:
        raise FileExistsError(f'Já existe LABEL para {filename} em {annotation_path}')
    annotation['SR_plate_text'] = {**existing_labels, filename: plate_label}

    fd, temporary_json = tempfile.mkstemp(prefix='.annotations_', suffix='.json', dir=track)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(annotation, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        if not save_exclusive(image, track, extension):
            return False
        try:
            os.replace(temporary_json, annotation_path)
        except OSError:
            (track / filename).unlink()
            raise
        return True
    finally:
        if os.path.exists(temporary_json):
            os.unlink(temporary_json)


class Reviewer:
    def __init__(self, window, records, model, ocr, args, logger, lq_size, scale):
        self.window, self.records, self.model = window, records, model
        self.ocr, self.args, self.logger = ocr, args, logger
        self.lq_size = lq_size
        self.scale = scale
        self.index, self.current = 0, None
        self.counted_tracks = saved_sr_tracks(records)
        self.current_errors = None
        self.current_ocr = self.current_label = None
        self.photos = []
        self.max_errors = tk.IntVar(value=args.max_errors)
        self.manual_plate = tk.StringVar(value='')
        self.status = tk.StringVar(value='Iniciando...')
        self.counter = tk.StringVar(value=f'SR geradas no treino: {len(self.counted_tracks)}')

        window.title('Revisão de super-resolução por OCR')
        window.geometry('1200x650')
        header = ttk.Frame(window, padding=10)
        header.pack(fill='x')
        ttk.Label(header, text='Máximo de caracteres com erro:').pack(side='left')
        ttk.Spinbox(header, from_=0, to=20, textvariable=self.max_errors, width=4).pack(side='left', padx=8)
        ttk.Label(header, textvariable=self.counter).pack(side='right')

        self.info = ttk.Label(window, textvariable=self.status, padding=(10, 0))
        self.info.pack(fill='x')
        self.grid = ttk.Frame(window, padding=10)
        self.grid.pack(fill='both', expand=True)
        for column in range(4):
            self.grid.columnconfigure(column, weight=1)
        for row in range(2):
            self.grid.rowconfigure(row, weight=1)
        self.panels = []
        for number in range(7):
            cell = ttk.Frame(self.grid, padding=4)
            cell.grid(row=number // 4, column=number % 4, sticky='nsew')
            title = ttk.Label(cell, text='', anchor='center')
            title.pack(fill='x')
            if number == 6:
                self.ocr_row = ttk.Frame(cell)
                self.ocr_row.pack(pady=(3, 5))
            picture = ttk.Label(cell, anchor='center')
            picture.pack(fill='both', expand=True)
            self.panels.append((title, picture))

        controls = ttk.Frame(window, padding=10)
        controls.pack(fill='x')
        self.next_button = ttk.Button(controls, text='Próxima imagem', command=self.next_image)
        self.next_button.pack(side='left', padx=5)
        self.accept_button = ttk.Button(controls, text='Aceitar SR', command=self.accept)
        self.accept_button.pack(side='left', padx=5)
        self.add_label_button = ttk.Button(controls, text='Adicionar LABEL', command=self.add_label,
                                           state='disabled')
        self.add_label_button.pack(side='left', padx=5)
        ttk.Label(controls, text='Placa:').pack(side='left', padx=(10, 3))
        self.manual_entry = ttk.Entry(controls, textvariable=self.manual_plate, width=12,
                                      justify='center', state='disabled')
        self.manual_entry.pack(side='left')
        self.manual_plate.trace_add('write', self.on_manual_plate_change)
        self.max_errors.trace_add('write', self.on_threshold_change)
        self.window.after(0, self.scan)

    def mark_saved(self, track):
        self.counted_tracks.add(track)
        self.counter.set(f'SR geradas no treino: {len(self.counted_tracks)}')

    def on_manual_plate_change(self, *_):
        value = self.manual_plate.get()
        uppercase = value.upper()
        if value != uppercase:
            self.manual_plate.set(uppercase)
            return
        self.add_label_button.configure(
            state='normal' if self.current is not None and
            valid_sr_label(value.strip(), self.current_label) else 'disabled'
        )

    def on_threshold_change(self, *_):
        try:
            limit = int(self.max_errors.get())
        except (ValueError, tk.TclError):
            return
        if limit >= 0 and self.current is not None and self.current_errors > limit:
            self.next_image()

    def threshold(self):
        try:
            value = int(self.max_errors.get())
            if value < 0:
                raise ValueError
            return value
        except (ValueError, tk.TclError):
            messagebox.showerror('Limite inválido', 'Informe um inteiro maior ou igual a zero.')
            return None

    def next_image(self):
        if self.threshold() is None:
            return
        self.current = None
        self.current_errors = None
        self.current_ocr = self.current_label = None
        self.manual_plate.set('')
        self.manual_entry.configure(state='disabled')
        for widget in self.ocr_row.winfo_children():
            widget.destroy()
        self.next_button.configure(state='disabled')
        self.accept_button.configure(state='disabled')
        self.add_label_button.configure(state='disabled')
        self.window.after(0, self.scan)

    def scan(self):
        limit = self.threshold()
        if limit is None:
            self.next_button.configure(state='normal')
            return
        if self.index >= len(self.records):
            self.status.set('Fim do split de treino. Nenhuma outra imagem atende ao filtro.')
            self.next_button.configure(state='disabled')
            self.accept_button.configure(state='disabled')
            self.add_label_button.configure(state='disabled')
            return
        i = self.index
        self.index += 1
        try:
            lq_path, gt_path, gt_paths, track, label = self.records[i]
            if already_saved(track):
                self.mark_saved(track)
                self.window.after(0, self.scan)
                return
            hrs = hr_images(track, gt_path, self.args.hr_glob) if self.args.hr_glob or len(gt_paths) != 5 else gt_paths
            if len(hrs) != 5 or any(not path.is_file() for path in hrs):
                raise ValueError(f'A trilha {track} não contém cinco arquivos HR válidos.')

            lr, lq_tensor = training_lr(lq_path, self.lq_size)
            height, width = lq_tensor.shape[-2:]
            gt_tensor = training_hr(gt_path, height * self.scale, width * self.scale)
            self.model.feed_data({
                'lq': lq_tensor, 'gt': gt_tensor,
                'lq_path': [str(lq_path)], 'gt_path': [str(gt_path)],
                'label': [label],
            })
            with torch.inference_mode():
                self.model.test()
            visuals = self.model.get_current_visuals()
            result = visuals.get('result')
            if result is None:
                raise KeyError('get_current_visuals() não retornou a chave "result"')
            sr = Image.fromarray(tensor_rgb(result))
            # predict() aplica o pré-processamento e a decodificação usados
            # na validação; '-' indica uma posição vazia e conta como erro.
            ocr_predictions = self.ocr.predict(result)
            if len(ocr_predictions) != 1:
                raise ValueError(f'O OCR retornou {len(ocr_predictions)} placas para uma LR.')
            ocr = str(ocr_predictions[0]).strip().upper()
            errors = character_errors(label, ocr)
            if errors <= limit:
                self.current = (sr, track)
                self.current_errors = errors
                self.current_ocr, self.current_label = ocr, label
                self.show(lr, hrs, sr, label, ocr, errors, i)
                self.next_button.configure(state='normal')
                self.accept_button.configure(state='normal')
                return
            self.status.set(f'Analisando {self.index}/{len(self.records)} | OCR: {ocr} | GT: {label}')
            self.window.after(0, self.scan)
        except Exception as exc:
            self.logger.exception('Falha na amostra %d', i)
            self.status.set(f'Erro na amostra {i}: {exc}')
            self.next_button.configure(state='normal')
            self.accept_button.configure(state='disabled')
            self.add_label_button.configure(state='disabled')

    def show(self, lr, hrs, sr, label, ocr, errors, i):
        self.manual_entry.configure(state='normal')
        self.manual_plate.set(ocr)
        self.status.set(
            f'Amostra {i + 1}/{len(self.records)} | GT: {label} | '
            f'OCR da SR: {ocr or "(vazio)"} | Erros: {errors} | Pasta: {self.current[1]}'
        )
        self.photos.clear()
        for widget in self.ocr_row.winfo_children():
            widget.destroy()
        ttk.Label(self.ocr_row, text='OCR: ', font=('TkDefaultFont', 12)).pack(side='left')
        positions = colored_ocr_positions(label, ocr)
        if not positions:
            positions = [('∅', False)]
        for char, correct in positions:
            ttk.Label(
                self.ocr_row,
                text=char,
                foreground='#17803d' if correct else '#bd2424',
                font=('TkDefaultFont', 15, 'bold'),
            ).pack(side='left')
        images = [(lr, 'LR usada')] + [(load_rgb(path), f'HR {n}: {path.name}')
                                              for n, path in enumerate(hrs, 1)] + [(sr, 'SR gerada')]
        for (title, picture), (img, label_text) in zip(self.panels, images):
            title.configure(text=label_text)
            thumb = img.copy()
            scale = min(260 / thumb.width, 190 / thumb.height)
            thumb = thumb.resize((max(1, round(thumb.width * scale)),
                                  max(1, round(thumb.height * scale))), Image.Resampling.NEAREST)
            photo = ImageTk.PhotoImage(thumb)
            picture.configure(image=photo)
            self.photos.append(photo)

    def accept(self):
        if self.current is None:
            return
        limit = self.threshold()
        if limit is None:
            return
        if self.current_errors > limit:
            self.next_image()
            return
        sr, track = self.current
        try:
            if save_exclusive(sr, track, self.args.extension):
                self.mark_saved(track)
                self.logger.info('SR aceita: %s', track / f'mamba_sr.{self.args.extension}')
            else:
                self.mark_saved(track)
                messagebox.showinfo('Já existe SR', f'A pasta já contém mamba_sr.*: {track}')
        except OSError as exc:
            messagebox.showerror('Erro ao salvar', str(exc))
            return
        self.next_image()

    def add_label(self):
        if self.current is None:
            return
        limit = self.threshold()
        if limit is None:
            return
        if self.current_errors > limit:
            self.next_image()
            return
        sr, track = self.current
        plate_label = self.manual_plate.get().strip().upper()
        try:
            if save_labeled_sr(sr, track, self.args.extension, plate_label, self.current_label):
                self.mark_saved(track)
                self.logger.info('SR com LABEL aceita: %s | LABEL: %s | OCR: %s',
                                 track / f'mamba_sr.{self.args.extension}', plate_label,
                                 self.current_ocr)
            else:
                self.mark_saved(track)
                messagebox.showinfo('Já existe SR', f'A pasta já contém mamba_sr.*: {track}')
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            messagebox.showerror('Erro ao adicionar LABEL', str(exc))
            return
        self.next_image()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-opt', type=Path, help='YAML usado no treino: contém datasets.train e train.ocr_opt')
    parser.add_argument('--sr-checkpoint', type=Path, help='Pesos da SR, caso path.pretrain_network_g não esteja definido no YAML de teste')
    parser.add_argument('--max-errors', type=int, default=2)
    parser.add_argument('--dataset', help='Nome do dataset de treinamento no YAML; padrão: train')
    parser.add_argument('--hr-glob', help='Glob relativo à pasta da trilha, ex.: HR/*.png')
    parser.add_argument('--label-key', help='Chave do GT na amostra ou JSON; aceita campo.subcampo')
    parser.add_argument('--track-levels-from-gt', type=int, help='Níveis acima do arquivo GT para localizar a pasta de saída (padrão: automático)')
    parser.add_argument('--extension', choices=('png', 'jpg', 'jpeg'), default='png')
    args, basicsr_args = parser.parse_known_args()
    if args.max_errors < 0 or (args.track_levels_from_gt is not None and args.track_levels_from_gt < 1):
        parser.error('--max-errors deve ser >= 0 e --track-levels-from-gt deve ser >= 1')
    original_argv = sys.argv
    try:
        # O parse_options do BasicSR aceita -opt; este revisor também aceita
        # --opt, usado normalmente em comandos Python.
        normalized_args = []
        for argument in basicsr_args:
            if argument == '--opt':
                normalized_args.append('-opt')
            elif argument.startswith('--opt='):
                normalized_args.append('-opt=' + argument[len('--opt='):])
            else:
                normalized_args.append(argument)
        sys.argv = [original_argv[0], *normalized_args]
        opt, _ = parse_options(str(PROJECT_ROOT), is_train=False)
    finally:
        sys.argv = original_argv

    torch.backends.cudnn.benchmark = True
    make_exp_dirs(opt)
    log_file = Path(opt['path']['log']) / f"review_{opt['name']}_{get_time_str()}.log"
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=str(log_file))
    logger.info(dict2str(opt))
    if args.train_opt:
        with args.train_opt.expanduser().open(encoding='utf-8') as stream:
            training_opt = yaml.safe_load(stream)
        for key, value in training_opt['datasets'].items():
            value['phase'] = key.split('_')[0]
            if 'scale' in training_opt:
                value['scale'] = training_opt['scale']
    else:
        training_opt = opt
    name = args.dataset or next((key for key, value in training_opt['datasets'].items()
                                  if key.lower().startswith('train') or
                                  value.get('phase') in ('train', 'training')), None)
    if name is None or name not in training_opt['datasets']:
        raise ValueError('Informe --train-opt com o YAML que define datasets.train.')
    dataset_opt = training_opt['datasets'][name]
    if not (name.lower().startswith('train') or dataset_opt.get('phase') in ('train', 'training')):
        raise ValueError(f'O dataset {name!r} não está identificado como split de treinamento.')
    ocr_opt = training_opt.get('train', {}).get('ocr_opt')
    if not ocr_opt:
        raise ValueError('Informe --train-opt com train.ocr_opt (load e alphabet).')
    if args.sr_checkpoint:
        opt['path']['pretrain_network_g'] = str(args.sr_checkpoint.expanduser().resolve())
    if not opt['path'].get('pretrain_network_g'):
        raise ValueError('Informe --sr-checkpoint ou path.pretrain_network_g no YAML de teste.')
    dataset = build_dataset(dataset_opt)
    records = training_records(dataset, args.track_levels_from_gt, args.label_key)
    logger.info('LR disponíveis no split de treino %s: %d', name, len(records))
    model = build_model(opt)
    ocr = load_training_ocr(ocr_opt, model.device)
    window = tk.Tk()
    Reviewer(window, records, model, ocr, args, logger, dataset_opt.get('lq_size'),
             int(dataset_opt['scale']))
    window.mainloop()


if __name__ == '__main__':
    main()
