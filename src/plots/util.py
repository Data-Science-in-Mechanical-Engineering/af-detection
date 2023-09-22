from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import seaborn as sns
from pypdf import PdfMerger

from src.results import Outcome

VIBRANT_COLOR_LIGHT = "#79bee6"
VIBRANT_COLOR_DARK = "#00549f"


class Style:
    LABEL_FONT_SIZE = 7
    LABEL_FONT_SIZE_LARGE = 9
    VIBRANT_COLOR_LIGHT = VIBRANT_COLOR_LIGHT
    VIBRANT_COLOR_DARK = VIBRANT_COLOR_DARK
    VIBRANT_COLOR_PALETTE = sns.blend_palette([VIBRANT_COLOR_LIGHT, VIBRANT_COLOR_DARK], as_cmap=True)
    GREEN_COLOR = "#8dc060"
    RED_COLOR = "#d80032"

    @staticmethod
    def vibrant_colors_discrete(n_colors: int):
        return sns.blend_palette([VIBRANT_COLOR_LIGHT, VIBRANT_COLOR_DARK], n_colors=n_colors)


def merge_pdfs(root: Path, target_pdf_name: str = "merged"):
    assert root.is_dir()

    def merge_pdfs_folder(folder: Path):
        pdf_paths = [Path(folder / f) for f in os.listdir(folder) if os.path.isfile(folder / f) and f.endswith(".pdf")]

        if not pdf_paths:
            return

        with PdfMerger() as merger:
            merger = PdfMerger()

            for pdf_path in pdf_paths:
                merger.append(pdf_path)

            merger.write(folder / f"{target_pdf_name}.pdf")

    folders = [root]

    while folders:
        root = folders.pop()
        folders.extend(Path(root / f) for f in os.listdir(root) if os.path.isdir(root / f))
        merge_pdfs_folder(root)


Metric = Callable[[Outcome], float]
