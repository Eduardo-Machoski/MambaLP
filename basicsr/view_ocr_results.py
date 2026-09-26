"""Browse saved SR/OCR cases; requires only Pillow and Tkinter (no PyTorch)."""

import argparse
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk


def load_records(root):
    root = Path(root).expanduser().resolve()
    manifests = [root] if root.is_file() else sorted(root.rglob('manifest.jsonl'))
    records, warnings = [], []
    for manifest in manifests:
        with manifest.open(encoding='utf-8') as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    if item.get('format') != 'ocr_sr_case_v1':
                        continue
                    item['_root'] = str(manifest.parent)
                    records.append(item)
                except json.JSONDecodeError:
                    warnings.append(f'{manifest.name}:{number}: registro incompleto ou inválido')
    return records, warnings


def filter_records(records, correct='Todos', dataset='Todos', only_errors=False, query=''):
    query = query.strip().upper()
    return [r for r in records
            if (correct == 'Todos' or r['correct_characters'] == int(correct))
            and (dataset == 'Todos' or r['dataset'] == dataset)
            and (not only_errors or not r['plate_correct'])
            and (not query or query in ' '.join((r['target'], r['prediction'], r['original_lq_path'])).upper())]


class OCRReviewApp:
    def __init__(self, root, results=None):
        self.root = root
        root.title('Revisão de SR e OCR')
        root.geometry(f'{min(1500, root.winfo_screenwidth() - 60)}x{min(900, root.winfo_screenheight() - 80)}')
        root.minsize(1000, 650)
        self.results = results
        self.records, self.filtered = [], []
        self.current = None
        self.image_cache = {}
        self.photos = []
        self.resize_job = None
        self.correct = tk.StringVar(value='Todos')
        self.dataset = tk.StringVar(value='Todos')
        self.only_errors = tk.BooleanVar(value=False)
        self.query = tk.StringVar()
        self.status = tk.StringVar(value='Abra a pasta de resultados do teste.')
        self.details = tk.StringVar(value='Selecione uma SR na lista.')
        toolbar = ttk.Frame(root, padding=8)
        toolbar.pack(fill='x')
        ttk.Button(toolbar, text='Abrir resultados', command=self.choose_folder).pack(side='left')
        ttk.Button(toolbar, text='Atualizar lista', command=self.reload).pack(side='left', padx=(6, 14))
        ttk.Label(toolbar, text='Acertos:').pack(side='left')
        self.correct_box = ttk.Combobox(toolbar, textvariable=self.correct, values=['Todos'], width=7, state='readonly')
        self.correct_box.pack(side='left', padx=(4, 12))
        self.correct_box.bind('<<ComboboxSelected>>', self.apply_filter)
        ttk.Label(toolbar, text='Dataset:').pack(side='left')
        self.dataset_box = ttk.Combobox(toolbar, textvariable=self.dataset, values=['Todos'], width=22, state='readonly')
        self.dataset_box.pack(side='left', padx=(4, 12))
        self.dataset_box.bind('<<ComboboxSelected>>', self.apply_filter)
        ttk.Checkbutton(toolbar, text='Apenas com erro', variable=self.only_errors, command=self.apply_filter).pack(side='left')
        searchbar = ttk.Frame(root, padding=(8, 0, 8, 8))
        searchbar.pack(fill='x')
        ttk.Label(searchbar, text='Buscar GT / OCR / caminho:').pack(side='left', padx=(0, 4))
        entry = ttk.Entry(searchbar, textvariable=self.query, width=22)
        entry.pack(side='left', fill='x', expand=True)
        entry.bind('<Return>', self.apply_filter)
        ttk.Button(searchbar, text='Filtrar', command=self.apply_filter).pack(side='left', padx=5)
        body = ttk.Panedwindow(root, orient='horizontal')
        body.pack(fill='both', expand=True, padx=8)
        list_frame = ttk.Frame(body)
        body.add(list_frame, weight=1)
        self.tree = ttk.Treeview(list_frame, columns=('correct', 'gt', 'ocr'), show='headings', selectmode='browse')
        for name, text, width in [('correct', 'Acertos', 60), ('gt', 'GT', 110), ('ocr', 'OCR da SR', 110)]:
            self.tree.heading(name, text=text)
            self.tree.column(name, width=width, minwidth=50, anchor='center')
        tree_scroll = ttk.Scrollbar(list_frame, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.pack(side='left', fill='both', expand=True)
        tree_scroll.pack(side='right', fill='y')
        self.tree.bind('<<TreeviewSelect>>', self.select_record)
        right = ttk.Frame(body, padding=(10, 0))
        body.add(right, weight=4)
        ttk.Label(right, textvariable=self.details, wraplength=950, justify='left').pack(anchor='w', pady=8)
        self.ocr_row = ttk.Frame(right)
        self.ocr_row.pack(anchor='w', pady=(0, 8))
        ttk.Label(right, text='Verde: correto na posição · Vermelho: erro · ∅: ausente · Duplo clique para ampliar').pack(anchor='w')
        images = ttk.Frame(right)
        images.pack(fill='both', expand=True, pady=8)
        self.canvas = tk.Canvas(images, highlightthickness=0, background='#f5f6f8')
        scroll = ttk.Scrollbar(images, orient='vertical', command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scroll.set)
        self.canvas.pack(side='left', fill='both', expand=True)
        scroll.pack(side='right', fill='y')
        self.inner = ttk.Frame(self.canvas, padding=6)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.inner, anchor='nw')
        self.inner.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        self.canvas.bind('<Configure>', self.resize_images)
        footer = ttk.Frame(root, padding=8)
        footer.pack(fill='x')
        ttk.Button(footer, text='◀ Anterior', command=lambda: self.move(-1)).pack(side='left')
        ttk.Button(footer, text='Próxima ▶', command=lambda: self.move(1)).pack(side='left', padx=6)
        ttk.Label(footer, textvariable=self.status).pack(side='left', padx=12)
        root.bind('<Left>', lambda e: self.key_move(e, -1))
        root.bind('<Right>', lambda e: self.key_move(e, 1))
        root.bind('<MouseWheel>', self.wheel)
        root.bind('<Button-4>', self.wheel)
        root.bind('<Button-5>', self.wheel)
        if results:
            root.after(50, self.reload)

    def choose_folder(self):
        folder = filedialog.askdirectory(title='Selecione a pasta do teste ou ocr_images')
        if folder:
            self.results = folder
            self.reload()

    def reload(self):
        if not self.results:
            self.choose_folder()
            return
        try:
            self.records, warnings = load_records(self.results)
        except (OSError, ValueError) as error:
            messagebox.showerror('Erro ao carregar', str(error))
            return
        self.correct_box['values'] = ['Todos'] + [str(n) for n in sorted({r['correct_characters'] for r in self.records})]
        self.dataset_box['values'] = ['Todos'] + sorted({r['dataset'] for r in self.records})
        if self.correct.get() not in self.correct_box['values']:
            self.correct.set('Todos')
        if self.dataset.get() not in self.dataset_box['values']:
            self.dataset.set('Todos')
        self.apply_filter()
        if warnings:
            self.status.set(self.status.get() + f' · {len(warnings)} registros incompletos; atualize novamente.')

    def apply_filter(self, event=None):
        self.filtered = filter_records(self.records, self.correct.get(), self.dataset.get(), self.only_errors.get(), self.query.get())
        self.tree.delete(*self.tree.get_children())
        for i, item in enumerate(self.filtered):
            self.tree.insert('', 'end', iid=str(i), values=(f"{item['correct_characters']}/{item['total_positions']}", item['target'], item['prediction']))
        self.current = None
        for widget in self.inner.winfo_children(): widget.destroy()
        for widget in self.ocr_row.winfo_children(): widget.destroy()
        self.photos.clear()
        self.status.set(f'{len(self.filtered)} de {len(self.records)} SRs · Filtros por OCR individual, antes do majority vote')
        self.details.set('Nenhuma imagem neste filtro.' if self.records else 'Ainda não há SRs salvas. Execute o teste e clique em Atualizar lista.')
        if self.filtered:
            self.tree.selection_set('0')
            self.tree.focus('0')

    def select_record(self, event=None):
        selected = self.tree.selection()
        if not selected: return
        index = int(selected[0])
        self.current = self.filtered[index]
        self.image_cache.clear()
        r = self.current
        self.details.set(
            f"{r['dataset']} · Imagem {index + 1}/{len(self.filtered)} · GT: {r['target']} · "
            f"Acertos: {r['correct_characters']}/{r['total_positions']} · Erros: {r['errors']}\n"
            f"LR original: {r['original_lq_path']}\nOCR abaixo pertence a esta SR individual.")
        for widget in self.ocr_row.winfo_children(): widget.destroy()
        ttk.Label(self.ocr_row, text='OCR da SR: ', font=('TkDefaultFont', 13, 'bold')).pack(side='left')
        for pos in r['positions']:
            tk.Label(self.ocr_row, text=pos['predicted'] or '∅', fg='#15803d' if pos['correct'] else '#dc2626',
                     font=('Courier', 24, 'bold'), padx=2).pack(side='left')
        self.canvas.yview_moveto(0)
        self.render_images()

    def resize_images(self, event):
        self.canvas.itemconfigure(self.canvas_window, width=event.width)
        if self.resize_job is not None: self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(120, self.render_images)

    def render_images(self):
        self.resize_job = None
        if self.current is None: return
        for widget in self.inner.winfo_children(): widget.destroy()
        self.photos.clear()
        r = self.current
        width = max(self.canvas.winfo_width() - 36, 600)
        top = ttk.Frame(self.inner)
        top.pack(fill='x')
        self.image_card(top, 'LR usada pelo modelo', r['lr_image'], width // 2 - 12, 250, 0, 0, True)
        self.image_card(top, f"SR gerada — OCR: {r['prediction']}", r['sr_image'], width // 2 - 12, 250, 0, 1, True)
        ttk.Label(self.inner, text=f"HRs de referência ({len(r['hr_images'])})", font=('TkDefaultFont', 13, 'bold')).pack(anchor='w', pady=(18, 8))
        refs = ttk.Frame(self.inner)
        refs.pack(fill='x')
        columns = 3 if width >= 850 else 2
        for i, path in enumerate(r['hr_images']):
            self.image_card(refs, f'HR {i + 1} — {Path(path).name}', path, width // columns - 12, 230, i // columns, i % columns)
        if not r['hr_images']:
            ttk.Label(refs, text='Nenhuma referência HR registrada.').pack()

    def image_card(self, parent, title, relative_path, max_width, max_height, row, column, nearest=False):
        frame = ttk.LabelFrame(parent, text=title, padding=6)
        frame.grid(row=row, column=column, padx=4, pady=5, sticky='nsew')
        parent.columnconfigure(column, weight=1)
        path = Path(self.current['_root']) / relative_path
        try:
            if str(path) not in self.image_cache:
                with Image.open(path) as source: self.image_cache[str(path)] = source.convert('RGB')
            img = self.image_cache[str(path)]
            scale = min(max_width / img.width, max_height / img.height)
            shown = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                               Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(shown)
            self.photos.append(photo)
            label = ttk.Label(frame, image=photo, anchor='center')
            label.pack(fill='both', expand=True)
            label.bind('<Double-Button-1>', lambda e, p=path, t=title: self.open_zoom(p, t))
            ttk.Label(frame, text=f'{img.width} × {img.height} px', anchor='center').pack(fill='x', pady=(5, 0))
        except (OSError, ValueError) as error:
            ttk.Label(frame, text=f'Não foi possível abrir:\n{path.name}\n{error}', wraplength=max_width).pack()

    def open_zoom(self, path, title):
        with Image.open(path) as source: img = source.convert('RGB')
        img = img.resize((img.width * 4, img.height * 4), Image.Resampling.NEAREST)
        window = tk.Toplevel(self.root)
        window.title(title + ' — ampliação 4×')
        window.geometry('1000x650')
        canvas = tk.Canvas(window)
        xscroll = ttk.Scrollbar(window, orient='horizontal', command=canvas.xview)
        yscroll = ttk.Scrollbar(window, orient='vertical', command=canvas.yview)
        canvas.configure(xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
        xscroll.pack(side='bottom', fill='x'); yscroll.pack(side='right', fill='y')
        canvas.pack(fill='both', expand=True)
        photo = ImageTk.PhotoImage(img)
        canvas.image_reference = photo
        canvas.create_image(0, 0, anchor='nw', image=photo)
        canvas.configure(scrollregion=(0, 0, img.width, img.height))

    def move(self, amount):
        if not self.filtered: return
        selected = self.tree.selection()
        index = int(selected[0]) if selected else 0
        index = max(0, min(len(self.filtered) - 1, index + amount))
        self.tree.selection_set(str(index)); self.tree.focus(str(index)); self.tree.see(str(index))

    def key_move(self, event, amount):
        if event.widget.winfo_class() not in {'Entry', 'TEntry', 'TCombobox', 'Text'}:
            self.move(amount)
            return 'break'

    def wheel(self, event):
        if str(event.widget).startswith(str(self.canvas)):
            direction = -1 if getattr(event, 'num', None) == 4 or getattr(event, 'delta', 0) > 0 else 1
            self.canvas.yview_scroll(direction * 3, 'units')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', help='Pasta do teste, ocr_images, dataset ou manifest.jsonl')
    args = parser.parse_args()
    root = tk.Tk()
    OCRReviewApp(root, args.results)
    root.mainloop()


if __name__ == '__main__':
    main()
