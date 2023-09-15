from __future__ import annotations

import os
from pathlib import Path

import seaborn as sns
from pypdf import PdfMerger

VIBRANT_COLOR_LIGHT = "#f37651"
VIBRANT_COLOR_DARK = "#701f57"


class Style:
    LABEL_FONT_SIZE = 7
    VIBRANT_COLOR_LIGHT = "#f37651"
    VIBRANT_COLOR_DARK = "#701f57"
    VIBRANT_COLOR_PALETTE = sns.blend_palette([VIBRANT_COLOR_LIGHT, VIBRANT_COLOR_DARK], as_cmap=True)

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
