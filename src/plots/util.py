import seaborn as sns


class Style:
    LABEL_FONT_SIZE = 7
    VIBRANT_COLOR_PALETTE = sns.color_palette("flare", as_cmap=True)
    VIBRANT_COLOR_PALETTE_DISCRETE = lambda n: sns.color_palette("flare", n_colors=n)
