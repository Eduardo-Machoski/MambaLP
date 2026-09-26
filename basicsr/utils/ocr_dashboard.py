"""Live, accumulated OCR plots. GUI rendering stays outside SR timing."""

import csv
import json
import logging
from pathlib import Path
from time import monotonic

import numpy as np


class OCRDashboard:
    def __init__(self, name, output_dir, alphabet, parameter_count, benchmark,
                 temporal=False, update_interval=0.5, top_k=10):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button

        self.plt = plt
        self.name = name
        self.output_dir = Path(output_dir)
        self.alphabet = alphabet
        self.parameter_count = parameter_count
        self.benchmark = benchmark
        self.temporal = temporal
        self.interval = float(update_interval)
        self.top_k = int(top_k)
        if self.interval < 0 or self.top_k < 1:
            raise ValueError('ocr_dashboard requires update_interval >= 0 and top_k >= 1')
        self.last_draw = float('-inf')
        self.latest = None
        self.closed = False
        self.normalized = False
        self.processed = self.total = 0
        try:
            self.fig = plt.figure(figsize=(18, 10))
        except (ImportError, RuntimeError) as error:
            logging.getLogger('basicsr').warning('OCR dashboard: %s; saving plots without a GUI.', error)
            plt.switch_backend('Agg')
            self.fig = plt.figure(figsize=(18, 10))
        backend = str(plt.get_backend()).lower()
        self.interactive = backend not in {'agg', 'pdf', 'svg', 'ps', 'cairo', 'template', 'pgf'} and 'inline' not in backend
        if not self.interactive:
            logging.getLogger('basicsr').warning(
                'No graphical Matplotlib backend: OCR plots will be saved. '
                'For a local desktop, run with MPLBACKEND=TkAgg.')
        if hasattr(self.fig.canvas.manager, 'set_window_title'):
            self.fig.canvas.manager.set_window_title(f'OCR — {name}')
        gs = self.fig.add_gridspec(3, 4, width_ratios=[1, 1, 1.2, 1.2])
        self.char_ax = self.fig.add_subplot(gs[0, :2])
        self.position_ax = self.fig.add_subplot(gs[1, 0])
        self.error_ax = self.fig.add_subplot(gs[1, 1])
        self.confusion_ax = self.fig.add_subplot(gs[2, :2])
        self.matrix_ax = self.fig.add_subplot(gs[:, 2:])
        self.fig.subplots_adjust(left=0.05, right=0.97, bottom=0.11, top=0.84,
                                 hspace=0.78, wspace=0.85)
        self.matrix_image = self.matrix_ax.imshow(np.zeros((1, 1)), cmap='Blues', vmin=0, vmax=1)
        self.colorbar = self.fig.colorbar(self.matrix_image, ax=self.matrix_ax, fraction=0.045, pad=0.025)
        self.hint = self.fig.text(0.53, 0.067, 'Passe o mouse na matriz para ver as contagens.', fontsize=9)
        self.fig.text(0.05, 0.037,
            'Precisão = acertos / predições do caractere. Acerto = acertos / ocorrências no GT.\n'
            'A matriz compara posições. ∅ indica caractere ausente ou extra.', fontsize=9)
        button_ax = self.fig.add_axes([0.76, 0.023, 0.19, 0.037])
        self.button = Button(button_ax, 'Matriz: contagem')
        self.button.on_clicked(self._toggle_normalization)
        self.fig.canvas.mpl_connect('close_event', self._on_close)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_hover)
        self.fig.suptitle(f'OCR — {name}\nAguardando a primeira predição...', fontsize=14, y=0.975)
        if self.interactive:
            plt.show(block=False)
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    def _on_close(self, event):
        self.closed = True

    def _toggle_normalization(self, event):
        self.normalized = not self.normalized
        self.button.label.set_text('Matriz: % por GT' if self.normalized else 'Matriz: contagem')
        if self.latest is not None:
            self._draw_matrix()
            self.fig.canvas.draw_idle()

    @staticmethod
    def _label(char):
        return '∅' if char == '<PAD>' else char

    def _on_hover(self, event):
        if self.latest is None or event.inaxes != self.matrix_ax or event.xdata is None or event.ydata is None:
            return
        col, row = int(round(event.xdata)), int(round(event.ydata))
        matrix = self.latest['confusion_matrix']
        if 0 <= row < len(matrix['labels']) and 0 <= col < len(matrix['labels']):
            count = matrix['counts'][row][col]
            total = sum(matrix['counts'][row])
            self.hint.set_text(f"GT {self._label(matrix['labels'][row])} → OCR {self._label(matrix['labels'][col])}: "
                               f"{count} ocorrências ({100 * count / total if total else 0:.1f}% deste GT)")
            self.fig.canvas.draw_idle()

    def update(self, metrics, processed, total, force=False):
        self.metrics = metrics
        self.processed, self.total = processed, total
        now = monotonic()
        if not force and (self.closed or now - self.last_draw < self.interval):
            return
        self.latest = metrics.to_dict(self.alphabet, self.top_k)
        self.last_draw = now
        self._draw()
        if self.interactive and not self.closed:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    def _draw(self):
        data = self.latest
        for ax in (self.char_ax, self.position_ax, self.error_ax, self.confusion_ax):
            ax.clear()
            ax.spines[['top', 'right']].set_visible(False)
        chars = data['characters']
        x = np.arange(len(chars))
        self.char_ax.bar(x - 0.2, [100 * r['precision'] for r in chars], width=0.4, color='#2563eb', label='Precisão (TP/pred.)')
        self.char_ax.bar(x + 0.2, [100 * r['accuracy'] for r in chars], width=0.4, color='#16a34a', label='Acerto (TP/GT)')
        self.char_ax.set_xticks(x, [r['character'] for r in chars], fontsize=8)
        self.char_ax.set_ylim(0, 105)
        self.char_ax.set_ylabel('%')
        self.char_ax.set_title('Char precision — por caractere', fontsize=11)
        self.char_ax.legend(fontsize=8, loc='lower left', bbox_to_anchor=(0, 1.18), ncol=2)

        positions = data['positions']
        bars = self.position_ax.bar([r['position'] for r in positions], [100 * r['accuracy'] for r in positions], color='#2563eb')
        self.position_ax.set_xticks([r['position'] for r in positions])
        self.position_ax.set_ylim(0, 115)
        self.position_ax.set_ylabel('% de acerto')
        self.position_ax.set_title('Position precision', fontsize=11)
        self.position_ax.set_xlabel('Posição na placa')
        self.position_ax.bar_label(bars, labels=[f"{100*r['accuracy']:.1f}" for r in positions], fontsize=8)

        errors = data['most_common_errors']
        self._horizontal(self.error_ax, [r['character'] for r in errors], [r['errors'] for r in errors], '#dc2626')
        self.error_ax.set_title(f'Most common error — top {self.top_k}', fontsize=11)
        self.error_ax.set_xlabel('Erros por caractere do GT')
        confusions = data['most_common_confusions']
        self._horizontal(self.confusion_ax,
            [f"{self._label(r['expected'])} → {self._label(r['predicted'])}" for r in confusions],
            [r['count'] for r in confusions], '#d97706')
        self.confusion_ax.set_title(f'Most common confusion — top {self.top_k}', fontsize=11)
        self.confusion_ax.set_xlabel('Contagem · GT → OCR')
        self._draw_matrix()
        timing = self.benchmark.summary()
        elapsed = (f"{timing['mean_ms']:.2f} ± {timing['std_ms']:.2f} ms" if timing['mean_ms'] is not None else 'aguardando')
        scope = 'por trilha (majority vote)' if self.temporal else 'por imagem'
        self.fig.suptitle(
            f'OCR — {self.name} · {scope} · Progresso: {self.processed}/{self.total}\n'
            f"Placas: {100 * data['plate_accuracy']:.2f}% ({data['correct_plates']}/{data['total_plates']})   |   "
            f"Caracteres sem considerar ordem: {100 * data['character_accuracy_any_position']:.2f}%\n"
            f'SR: {elapsed} · Parâmetros SR: {self.parameter_count:,}', fontsize=13, y=0.975)

    @staticmethod
    def _horizontal(ax, labels, counts, color):
        if not counts:
            ax.text(0.5, 0.5, 'Nenhum erro', transform=ax.transAxes, ha='center', va='center')
            ax.set_yticks([])
            return
        y = np.arange(len(labels))
        bars = ax.barh(y, counts, color=color)
        ax.set_yticks(y, labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, max(counts) * 1.22)
        ax.bar_label(bars, padding=3, fontsize=8)

    def _draw_matrix(self):
        matrix = self.latest['confusion_matrix']
        counts = np.asarray(matrix['counts'], dtype=float)
        if self.normalized:
            totals = counts.sum(axis=1, keepdims=True)
            values = np.divide(counts * 100, totals, out=np.zeros_like(counts), where=totals != 0)
        else:
            values = counts
        n = len(matrix['labels'])
        self.matrix_image.set_data(values)
        self.matrix_image.set_extent((-0.5, n - 0.5, n - 0.5, -0.5))
        self.matrix_image.set_clim(0, 100 if self.normalized else max(1, values.max()))
        self.matrix_ax.set_xlim(-0.5, n - 0.5)
        self.matrix_ax.set_ylim(n - 0.5, -0.5)
        labels = [self._label(char) for char in matrix['labels']]
        self.matrix_ax.set_xticks(range(n), labels, fontsize=7)
        self.matrix_ax.set_yticks(range(n), labels, fontsize=7)
        self.matrix_ax.set_xlabel('Caractere reconhecido pelo OCR')
        self.matrix_ax.set_ylabel('Caractere esperado (GT)')
        self.matrix_ax.set_title('Matriz de confusão · ' + ('% por GT' if self.normalized else 'contagens'), fontsize=12, pad=12)
        self.colorbar.set_label('%' if self.normalized else 'Ocorrências')

    def finish(self):
        if not hasattr(self, 'metrics'):
            return None
        self.update(self.metrics, self.processed, self.total, force=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        image_path = self.output_dir / 'ocr_dashboard.png'
        json_path = self.output_dir / 'ocr_metrics.json'
        csv_path = self.output_dir / 'character_confusion_matrix.csv'
        self.fig.savefig(image_path, dpi=150)
        json_path.write_text(json.dumps(self.latest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        matrix = self.latest['confusion_matrix']
        with csv_path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['GT / OCR'] + matrix['labels'])
            for label, row in zip(matrix['labels'], matrix['counts']):
                writer.writerow([label] + row)
        return {'plot': str(image_path), 'metrics': str(json_path), 'confusion_matrix': str(csv_path)}

    @staticmethod
    def keep_open(dashboards):
        if any(item.interactive and not item.closed for item in dashboards):
            dashboards[0].plt.show(block=True)
