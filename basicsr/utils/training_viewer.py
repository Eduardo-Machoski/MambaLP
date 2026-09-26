import multiprocessing as mp
import queue
import torch
import tkinter as tk

from tkinter import ttk

import cv2
import numpy as np
import os
from PIL import Image, ImageTk


class TrainingViewer:
    def __init__(self) -> None:
        self.ctx = mp.get_context("spawn")

        self.request_queue = self.ctx.Queue()
        self.data_queue = self.ctx.Queue()

        display = os.environ.get("DISPLAY")

        print(
            f"[TrainingViewer] DISPLAY principal: {display}",
            flush=True,
        )

        self.process = self.ctx.Process(
            target=_viewer_process,
            args=(
                self.request_queue,
                self.data_queue,
                display,
            ),
            daemon=False,
        )

        self.process.start()

        print(
            f"[TrainingViewer] Processo iniciado PID={self.process.pid}",
            flush=True,
        )

    def next_requested(self) -> bool:
        requested = False

        while True:
            try:
                message = self.request_queue.get_nowait()

                if message == "next":
                    requested = True

            except queue.Empty:
                break

        return requested

    def send(self, data: dict) -> None:
        # Mantém somente a amostra mais recente.
        while True:
            try:
                self.data_queue.get_nowait()
            except queue.Empty:
                break

        self.data_queue.put(data)

    def close(self) -> None:
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()

def _numpy_to_photoimage(
    image: np.ndarray,
    max_width: int = 350,
    max_height: int = 250,
) -> ImageTk.PhotoImage:

    if image.ndim == 2:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2RGB,
        )

    h, w = image.shape[:2]

    scale = min(
        max_width / w,
        max_height / h,
        1.0,
    )

    new_w = max(
        1,
        int(w * scale),
    )

    new_h = max(
        1,
        int(h * scale),
    )

    image = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST,
    )

    pil_image = Image.fromarray(
        image.astype(np.uint8)
    )

    return ImageTk.PhotoImage(
        pil_image
    )


def _viewer_process(
    request_queue: mp.Queue,
    data_queue: mp.Queue,
    display,
) -> None:

    if display:
        os.environ["DISPLAY"] = display

    print(
        "[TrainingViewer CHILD] DISPLAY:",
        os.environ.get("DISPLAY"),
        flush=True,
    )

    try:
        root = tk.Tk()

    except Exception as error:
        print(
            "[TrainingViewer] ERRO ao criar janela:"
        )
        print(error)
        return

    root.title(
        "MambaIR - Training Viewer"
    )

    root.geometry(
        "1500x520"
    )

    root.minsize(
        1100,
        450,
    )

    # =========================================================
    # Estado
    # =========================================================

    waiting = False

    image_references = {}

    # =========================================================
    # Informações superiores
    # =========================================================

    top_frame = ttk.Frame(root)

    top_frame.pack(
        fill="x",
        padx=15,
        pady=10,
    )

    iteration_var = tk.StringVar(
        value="Iteração: -"
    )

    gt_var = tk.StringVar(
        value="GT Label: -"
    )

    path_var = tk.StringVar(
        value = "HR Path: -"
    )

    status_var = tk.StringVar(
        value="Clique em 'Próxima imagem'."
    )

    ttk.Label(
        top_frame,
        textvariable=iteration_var,
        font=(
            "Arial",
            13,
            "bold",
        ),
    ).pack(
        side="left",
        padx=10,
    )

    ttk.Label(
        top_frame,
        textvariable=gt_var,
        font=(
            "Arial",
            13,
            "bold",
        ),
    ).pack(
        side="left",
        padx=30,
    )


    ttk.Label(
        top_frame,
        textvariable=path_var,
        font=(
            "Arial",
            13,
            "bold",
        ),
    ).pack(
        side="left",
        padx=30,
    )

    ttk.Label(
        top_frame,
        textvariable=status_var,
    ).pack(
        side="right",
        padx=10,
    )

    # =========================================================
    # Imagens
    # =========================================================

    image_frame = ttk.Frame(root)

    image_frame.pack(
        fill="both",
        expand=True,
        padx=10,
        pady=10,
    )

    image_frame.columnconfigure(
        0,
        weight=1,
    )

    image_frame.columnconfigure(
        1,
        weight=1,
    )

    image_frame.columnconfigure(
        2,
        weight=1,
    )

    image_frame.columnconfigure(
        3,
        weight=1,
    )

    cards = {}

    def create_card(
        column: int,
        title: str,
    ):
        frame = ttk.LabelFrame(
            image_frame,
            text=title,
        )

        frame.grid(
            row=0,
            column=column,
            padx=5,
            pady=5,
            sticky="nsew",
        )

        image_label = ttk.Label(
            frame,
            text="Sem imagem",
            anchor="center",
        )

        image_label.pack(
            fill="both",
            expand=True,
            padx=5,
            pady=5,
        )

        # Frame onde cada caractere do OCR será desenhado
        ocr_chars_frame = tk.Frame(
            frame,
        )

        ocr_chars_frame.pack(
            pady=10,
        )

        return {
            "frame": frame,
            "image": image_label,
            "ocr_chars_frame": ocr_chars_frame,
        }

    cards["lq"] = create_card(
        0,
        "LR",
    )

    cards["hr_original"] = create_card(
        1,
        "HR Original",
    )

    cards["hr_model"] = create_card(
        2,
        "HR passada ao modelo",
    )

    cards["sr"] = create_card(
        3,
        "SR Gerada",
    )

    tk.Label(
        cards["lq"]["ocr_chars_frame"],
        text="Entrada",
        font=(
            "Courier",
            12,
            "bold",
        ),
    ).pack()

    # =========================================================
    # Botão
    # =========================================================

    bottom_frame = ttk.Frame(root)

    bottom_frame.pack(
        fill="x",
        padx=15,
        pady=10,
    )

    def request_next() -> None:
        nonlocal waiting

        if waiting:
            return

        request_queue.put(
            "next"
        )

        waiting = True

        next_button.configure(
            state="disabled"
        )

        status_var.set(
            "Aguardando nova amostra..."
        )

    next_button = ttk.Button(
        bottom_frame,
        text="Próxima imagem",
        command=request_next,
    )

    next_button.pack()

    # =========================================================
    # Atualização de OCR
    # =========================================================

    def update_ocr(
        card_name: str,
        prediction: str,
        gt: str,
    ) -> None:

        card = cards[
            card_name
        ]

        prediction = str(
            prediction
        ).upper()

        gt = str(
            gt
        ).upper()

        frame = card[
            "ocr_chars_frame"
        ]

        # =========================================================
        # Limpa OCR anterior
        # =========================================================

        for widget in frame.winfo_children():
            widget.destroy()

        # =========================================================
        # Prefixo
        # =========================================================

        tk.Label(
            frame,
            text="OCR: ",
            font=(
                "Courier",
                12,
                "bold",
            ),
        ).pack(
            side="left"
        )

        # =========================================================
        # Caracteres
        # =========================================================

        for i, char in enumerate(
            prediction
        ):

            correct = (
                i < len(gt)
                and char == gt[i]
            )

            color = (
                "green"
                if correct
                else "red"
            )

            tk.Label(
                frame,
                text=char,
                fg=color,
                font=(
                    "Courier",
                    12,
                    "bold",
                ),
            ).pack(
                side="left"
            )

        # =========================================================
        # Resultado geral
        # =========================================================

        all_correct = (
            prediction == gt
        )

        tk.Label(
            frame,
            text=(
                " ✓"
                if all_correct
                else " ✗"
            ),
            fg=(
                "green"
                if all_correct
                else "red"
            ),
            font=(
                "Courier",
                12,
                "bold",
            ),
        ).pack(
            side="left"
        )

    # =========================================================
    # Atualiza imagem
    # =========================================================

    def update_image(
        card_name: str,
        image,
    ) -> None:

        if image is None:
            return

        # ---------------------------------------------------------
        # Torch Tensor -> NumPy RGB
        # ---------------------------------------------------------

        if isinstance(image, torch.Tensor):

            image = image.detach().cpu()

            # Remove batch:
            # [1, C, H, W] -> [C, H, W]
            if image.ndim == 4:
                image = image[0]

            # [C, H, W] -> [H, W, C]
            if image.ndim == 3:
                image = image.permute(
                    1,
                    2,
                    0,
                )

            image = image.numpy()

            # Normalmente tensors do BasicSR estão em [0, 1]
            if image.dtype != np.uint8:
                image = np.clip(
                    image,
                    0.0,
                    1.0,
                )

                image = (
                    image * 255.0
                ).round().astype(
                    np.uint8
                )

        # ---------------------------------------------------------
        # Garante NumPy
        # ---------------------------------------------------------

        if not isinstance(
            image,
            np.ndarray,
        ):
            print(
                "[TrainingViewer] "
                f"Tipo de imagem não suportado: "
                f"{type(image)}"
            )
            return

        # ---------------------------------------------------------
        # Grayscale
        # ---------------------------------------------------------

        if image.ndim == 2:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2RGB,
            )

        # ---------------------------------------------------------
        # Garante uint8
        # ---------------------------------------------------------

        if image.dtype != np.uint8:

            if (
                image.max() <= 1.0
                and image.min() >= 0.0
            ):
                image = (
                    image * 255.0
                ).round()

            image = np.clip(
                image,
                0,
                255,
            ).astype(
                np.uint8
            )

        # ---------------------------------------------------------
        # Converte para PhotoImage
        # ---------------------------------------------------------

        photo = _numpy_to_photoimage(
            image
        )

        image_references[
            card_name
        ] = photo

        cards[
            card_name
        ][
            "image"
        ].configure(
            image=photo,
            text="",
        )

    # =========================================================
    # Poll da fila
    # =========================================================

    def check_queue() -> None:
        nonlocal waiting

        latest = None

        while True:
            try:
                latest = (
                    data_queue
                    .get_nowait()
                )

            except queue.Empty:
                break

        if latest is not None:

            iteration = latest.get(
                "iteration",
                "-",
            )

            gt = str(
                latest.get(
                    "label",
                    "-",
                )
            )

            path = str(
                  latest.get(
                      "path",
                      "-",
                  )
            )

            iteration_var.set(
                f"Iteração: {iteration}"
            )

            gt_var.set(
                f"GT Label: {gt}"
            )

            
            update_image(
                "lq",
                latest.get(
                    "lq"
                ),
            )

            update_image(
                "hr_original",
                latest.get(
                    "hr_original"
                ),
            )

            update_image(
                "hr_model",
                latest.get(
                    "hr_model"
                ),
            )

            update_image(
                "sr",
                latest.get(
                    "sr"
                ),
            )

            update_ocr(
                "hr_original",
                latest.get(
                    "ocr_hr_original",
                    "",
                ),
                gt,
            )

            update_ocr(
                "hr_model",
                latest.get(
                    "ocr_hr_model",
                    "",
                ),
                gt,
            )

            update_ocr(
                "sr",
                latest.get(
                    "ocr_sr",
                    "",
                ),
                gt,
            )

            status_var.set(
                f"Amostra da iteração "
                f"{iteration}"
            )

            path_var.set(
                f"Path imagem: {path}"
            )

            waiting = False

            next_button.configure(
                state="normal"
            )

        root.after(
            200,
            check_queue,
        )

    root.after(
        200,
        check_queue,
    )

    root.mainloop()
