# -*- coding: utf-8 -*-
"""公共工具：路径、字体、绘图风格、数据读取。"""
import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORK = os.path.join(ROOT, "work")
DATA = os.path.join(WORK, "data")
CLEAN = os.path.join(WORK, "clean")
RES = os.path.join(WORK, "results")
FIG = os.path.join(WORK, "figs")
for d in (CLEAN, RES, FIG):
    os.makedirs(d, exist_ok=True)

# 中文字体与论文级排版
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
rcParams["font.family"] = "sans-serif"
rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 150
rcParams["savefig.dpi"] = 600
rcParams["axes.linewidth"] = 0.8
rcParams["xtick.direction"] = "in"
rcParams["ytick.direction"] = "in"
rcParams["xtick.top"] = False
rcParams["ytick.right"] = False
rcParams["legend.frameon"] = False
rcParams["axes.spines.top"] = False
rcParams["axes.spines.right"] = False

# 语义调色板（色盲友好）
C = dict(
    blue="#2E5E8C", lightblue="#6FA3C7", teal="#2A9D8F",
    orange="#E76F51", red="#C0392B", yellow="#E9C46A",
    green="#4C956C", purple="#7B5EA7", grey="#8C8C8C",
    dark="#22333B", sand="#D9C7A7",
)
SEQ = ["#2E5E8C", "#2A9D8F", "#E9C46A", "#E76F51", "#7B5EA7", "#4C956C", "#C0392B", "#6FA3C7"]


def savefig(fig, name, tight=True):
    if tight:
        fig.tight_layout()
    p = os.path.join(FIG, name)
    fig.savefig(p + ".png", bbox_inches="tight")
    fig.savefig(p + ".svg", bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


def read_csv_any(path):
    return pd.read_csv(path)
