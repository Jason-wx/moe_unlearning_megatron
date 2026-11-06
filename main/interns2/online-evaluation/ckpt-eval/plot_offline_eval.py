# -*- coding: utf-8 -*-
"""Intern-S2 main(bio) 线十算法 ckpt 离线测评双柱图（WMDP-Bio vs MMLU），
仿 main/{Gemma4,Qwen35}/plot_offline_eval.py 同款样式。

数据来源（离线全量口径，lm_eval_accelerate.sh 0-shot / bf16，2026-09-26/27）：
  - 本目录 lm_eval_results_interns2-main-<算法>-step2142_wmdp_bio_mmlu/result.json
  - 汇总文档 ../ckpt_eval_summary_20260926.md（主表口径）
  - Base = evaluations/results/baselines/lm_eval_results_interns2_base_wmdp_mmlu
    （0.8366/0.8515，wmdp_bio/mmlu 全量）

臂选择（与汇总文档主表一致）：GD 与 SimNPO 取 saveckpts 重跑版 2142；
PDU 取 lr4e-6 代表工作点（0.408/0.844；lr1e-6 臂 0.311/0.784 为 lr×eps
失配臂，见文档结论速览）；UNDIAL 为 tp4 旧结果——TP>1 梯度 bug 已证
（2026-09-29 作废，≈无遗忘），仅作占位展示，结论以 undial_fixed 重跑为准；
Trace 为独立 TRACE 代码库产物 ckpt-1428，数字出自 TRACE 仓库自家
lm-eval（同为全量 0-shot，n-samples 逐题一致，跨 harness 口径），
结果目录 Safe_MoE/TRACE/evaluation/Intern-S2-base05/
intern_s2_base05_forget1_lr1e-6_ep2_ep8_mb4_gb32_forgetonly/checkpoint-1428/。
排序：margin = MMLU − WMDP-Bio 降序。

头部排版说明：constrained layout 在 CJK 双字体回退下对 suptitle 的宽度测量
失真 → 标题与图例叠印；本脚本用显式 y 坐标三层栈（标题 0.982 → 图例行
0.915 → 坐标区 top 0.875），不依赖引擎测宽。脚注一行在 10 列版会超画布宽
被左右裁（10-04 像素级证实），故压文案 + fontsize 8，两侧余量 ≥40px。
"""

from pathlib import Path as _Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import PathPatch, Patch
from matplotlib.path import Path

# ---------------------------------------------------------------- 数据 ----
# 每项 = (WMDP-Bio acc, MMLU acc)；Base = Intern-S2 离线全量基线
INTERN_S2 = {  # 十算法 + Trace 参照 + Base；键名与图列一致
    "Base":       (0.8366, 0.8515),
    "Trace":      (0.2859, 0.8246),   # 跨 harness（TRACE lm-eval，ckpt-1428）
    "RMU":        (0.3700, 0.8418),   # 唯一深遗忘有效点
    "PDU":        (0.4077, 0.8435),   # lr4e-6 代表臂
    "GradDiff":   (0.6654, 0.8238),   # saveckpts 重跑版
    "CEU":        (0.6976, 0.7578),
    "SatImp":     (0.7321, 0.7755),
    "WGA":        (0.7258, 0.7653),
    "SimNPO":     (0.7423, 0.7780),   # saveckpts 重跑版
    "PureNPO":    (0.7628, 0.7876),
    "UNDIAL":     (0.8342, 0.8421),   # tp4 旧结果（TP 梯度 bug，已作废）
    "GradAscent": (0.2459, 0.2295),   # 崩塌
}

HERE = _Path(__file__).resolve().parent                 # ckpt-eval/
STEM = "offline_eval_wmdp_mmlu_interns2"

# ---------------------------------------------------------------- 样式 ----
C_WMDP, C_MMLU = "#2a78d6", "#eb6834"          # 已过 validate_palette 六项检查
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
SURFACE = "#ffffff"

CJK = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
try:
    font_manager.fontManager.addfont(CJK)
    cjk_name = font_manager.FontProperties(fname=CJK).get_name()
    plt.rcParams["font.family"] = ["DejaVu Sans", cjk_name]  # Latin + 中文回退
except Exception:
    pass                                        # 无中文字体时退化为默认字体
plt.rcParams["axes.unicode_minus"] = False

LBL_BBOX = dict(facecolor=SURFACE, edgecolor="none", pad=0.8)  # 防虚线/网格穿过数字


def rounded_top_bar(fig, ax, xc, y0, y1, w, r_pt=2.2, fc=None, zorder=3):
    """顶部圆角、基线处直角的细柱（r_pt 单位为点，自动换算 dpi）。"""
    inv = ax.transData.inverted()
    r_px = r_pt * fig.dpi / 72.0
    (ux, uy) = np.abs(np.diff(inv.transform([[0, 0], [r_px, r_px]]), axis=0)[0])
    rx = min(ux, w * 0.45)
    ry = min(uy, (y1 - y0) * 0.9) if y1 > y0 else 0
    x0, x1 = xc - w / 2, xc + w / 2
    verts = [(x0, y0), (x0, y1 - ry), (x0, y1), (x0 + rx, y1),
             (x1 - rx, y1), (x1, y1), (x1, y1 - ry), (x1, y0), (x0, y0)]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3,
             Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=fc, edgecolor="none",
                           zorder=zorder))


def panel_order(data):
    """按 margin = MMLU − WMDP 降序。"""
    algos = [a for a in data if a != "Base"]
    return sorted(algos, key=lambda a: data[a][1] - data[a][0], reverse=True)


def draw_axes(ax, data):
    algos = panel_order(data)
    n = len(algos)
    w, off, gutter = 0.32, 0.185, 1.05
    ax.set_xlim(-0.6, n - 0.3 + gutter)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(np.arange(n))
    # 11 列版列宽 ~111px，GradAscent(9pt)平铺 ≈139px 会侵入邻列 → 拆两行
    ax.set_xticklabels(["Grad\nAscent" if a == "GradAscent" else a for a in algos])
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"])

    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_bounds(-0.5, n - 0.35)   # 底轴线不伸进标签留白
    ax.tick_params(axis="x", length=0, labelsize=9, labelcolor=INK2, pad=6)
    ax.tick_params(axis="y", length=0, labelsize=8.5, labelcolor=MUTED)

    # Base 参考线（虚线）+ 右侧留白处的线端标注（离线全量口径）。
    # 两基线值接近（差<0.06）时线上方放不下两行标注 → WMDP 标注移其线下方。
    bw, bm = data["Base"]
    for val, fc in ((bw, C_WMDP), (bm, C_MMLU)):
        ax.hlines(val, -0.5, n - 0.35, color=fc, linestyle=(0, (4, 3)),
                  linewidth=1.2, alpha=0.8, zorder=2)
    anchors = ((bw, C_WMDP, "WMDP", bw - 0.018 if abs(bm - bw) < 0.06 else bw + 0.015,
                "top" if abs(bm - bw) < 0.06 else "bottom"),
               (bm, C_MMLU, "MMLU", bm + 0.015, "bottom"))
    for val, fc, tag, y, va in anchors:
        ax.text(n - 0.25, y, f"Base {tag}\n{val:.3f}", ha="left",
                va=va, fontsize=7, color=INK2, linespacing=1.4,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.5))

    for i, name in enumerate(algos):
        v = data[name]
        close = abs(v[1] - v[0]) < 0.04      # 双柱近等高 → 柱顶两标注并排会相触
        for dx, val, fc in ((-off, v[0], C_WMDP), (off, v[1], C_MMLU)):
            rounded_top_bar(FIG, ax, i + dx, 0, val, w, fc=fc)
            if close and fc is C_MMLU:       # MMLU 数值移柱顶内侧（白底垫片）
                ax.text(i + dx, val - 0.03, f"{val:.3f}", ha="center",
                        va="top", fontsize=7, color=INK2, bbox=LBL_BBOX)
            else:
                ax.text(i + dx, val + 0.02, f"{val:.3f}", ha="center",
                        va="bottom", fontsize=7, color=INK2, bbox=LBL_BBOX)


def make_figure(title, data, outdir, stem):
    global FIG
    # 头部不用 constrained layout（见文件 docstring）：显式 y 三层栈。
    fig, ax = plt.subplots(figsize=(7.8, 4.9), dpi=200, facecolor=SURFACE)
    FIG = fig
    fig.subplots_adjust(left=0.07, right=0.995, top=0.875, bottom=0.15)
    ax.set_facecolor(SURFACE)
    draw_axes(ax, data)
    ax.set_ylabel("Accuracy (0–1)", fontsize=9, color=INK2)

    fig.suptitle(f"{title} — main(bio) ckpt 离线测评（WMDP-Bio vs MMLU）",
                 fontsize=12.5, color=INK, y=0.982, va="top")
    fig.legend(handles=[Patch(facecolor=C_WMDP, label="WMDP-Bio（全量 1273 题）"),
                        Patch(facecolor=C_MMLU, label="MMLU（全量 14042 题）")],
               loc="center", bbox_to_anchor=(0.5, 0.915), ncols=2, frameon=False,
               fontsize=8.5, handlelength=1.2, handleheight=1.0, labelcolor=INK2)
    fig.supxlabel("lm-eval 全量 · 0-shot · bf16 · ckpt s2142（Trace@1428 跨harness）· 按 margin（MMLU−WMDP）降序"
                  " · 虚线=Base · UNDIAL=tp4 作废",
                  fontsize=8, color=MUTED, y=0.045)

    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{stem}.{ext}", facecolor=SURFACE)
    plt.close(fig)
    print(f"saved: {outdir / stem}.png / .pdf")


make_figure("Intern-S2", INTERN_S2, HERE, STEM)
print(f"\nIntern-S2 main(bio)（按 margin 降序）")
for name in panel_order(INTERN_S2):
    v = INTERN_S2[name]
    print(f"  {name:<12} wmdp/mmlu = {v[0]:.4f} / {v[1]:.4f}   margin = {v[1] - v[0]:.4f}")
