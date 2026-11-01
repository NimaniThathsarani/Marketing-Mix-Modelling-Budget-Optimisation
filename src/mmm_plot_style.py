"""Shared matplotlib style so every MMM figure looks consistent (and matches the dashboard)."""
import matplotlib as mpl
import matplotlib.pyplot as plt

# Fixed channel -> colour mapping (validated categorical palette, fixed order, never cycled).
CHANNEL_COLORS = {
    "TV": "#2a78d6",
    "Facebook/Instagram": "#eb6834",
    "Google Search": "#1baf7a",
    "YouTube": "#eda100",
    "Radio": "#e87ba4",
    "Outdoor": "#008300",
}
BASE_COLOR = "#c9c8c3"      # base / intercept
CONTROL_COLOR = "#8e8d88"   # external factors
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
ACTUAL_COLOR, PRED_COLOR = "#0b0b0b", "#2a78d6"


def apply():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "lines.linewidth": 2, "legend.frameon": False,
    })


def millions(ax, axis="y"):
    fmt = mpl.ticker.FuncFormatter(lambda v, _: f"{v/1e6:,.0f}M")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)
