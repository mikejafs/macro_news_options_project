"""Shared colours and plot style for every figure in the project.

Importing this module applies the style. The two period colours are chosen to
differ in lightness as well as hue, so they stay distinguishable in greyscale
and under the common forms of colour blindness.

    RED    2022-23, or the second series
    NAVY   2024-26, or the first series
    GREY   a third series and secondary lines
    INK    text and axes
    MUTED  reference lines and annotations
"""
import matplotlib as mpl
from cycler import cycler

RED = "#C8102E"
NAVY = "#1B365D"
GREY = "#8D99AE"
INK = "#1A1A1A"
MUTED = "#6B7280"
PERIOD = {"2021-23": RED, "2022-23": RED, "2024-26": NAVY}


def apply():
    mpl.rcParams.update({
        "axes.prop_cycle": cycler(color=[NAVY, RED, GREY, "#5B7DB1", "#E58A97"]),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "legend.frameon": False,
    })


apply()
