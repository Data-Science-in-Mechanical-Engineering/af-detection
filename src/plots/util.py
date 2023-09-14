from __future__ import annotations

import seaborn as sns

VIBRANT_COLOR_DARK = "#701f57"
VIBRANT_COLOR_LIGHT = "#f37651"


class Style:
    LABEL_FONT_SIZE = 7
    VIBRANT_COLOR_PALETTE = sns.blend_palette([VIBRANT_COLOR_LIGHT, VIBRANT_COLOR_DARK], as_cmap=True)

    @staticmethod
    def vibrant_colors_discrete(n_colors: int):
        return sns.blend_palette([VIBRANT_COLOR_LIGHT, VIBRANT_COLOR_DARK], n_colors=n_colors)
