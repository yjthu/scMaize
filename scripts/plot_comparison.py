#!/usr/bin/env python3
"""Generate PDF comparison chart for scMaizeExp vs scMaizeGO training results."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Training metrics extracted from logs ──────────────────────────────
epochs = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80])

# scMaizeExp – Expression-only (16.7M params)
exp_loss     = np.array([None, 0.2305, 0.2216, 0.2252, 0.2020, 0.2168, 0.2049, 0.2068,
                         0.2083, 0.2233, 0.2065, 0.1887, 0.1963, 0.1933, 0.1997, 0.1987, 0.1970])
exp_val_mse  = np.array([1.6890, 0.1033, 0.1027, 0.1052, 0.1075, 0.1075, 0.1037, 0.1004,
                         0.1020, 0.1017, 0.0995, 0.0960, 0.0968, 0.0992, 0.0969, 0.0970, 0.0974])
exp_val_pear = np.array([0.0372, 0.7520, 0.7612, 0.7706, 0.7696, 0.7724, 0.7748, 0.7790,
                         0.7774, 0.7818, 0.7799, 0.7810, 0.7834, 0.7815, 0.7831, 0.7833, 0.7830])
exp_test_mse = np.array([None, 0.1034, 0.1027, None, None, None, None, 0.1005,
                         None, None, 0.0996, 0.0961, None, None, None, None, None])
exp_test_pear= np.array([None, 0.7539, 0.7630, None, None, None, None, 0.7807,
                         None, None, 0.7814, 0.7825, None, None, None, None, None])

# scMaizeGO – Expression + GO dual embedding (16.8M params)
go_loss      = np.array([None, 0.2299, 0.2207, 0.2247, 0.2014, 0.2162, 0.2045, 0.2063,
                         0.2079, 0.2228, 0.2061, 0.1884, 0.1959, 0.1930, 0.1993, 0.1983, 0.1967])
go_val_mse   = np.array([1.3233, 0.1240, 0.1059, 0.1136, 0.1039, 0.1031, 0.1048, 0.1028,
                         0.0984, 0.0994, 0.1011, 0.0984, 0.1002, 0.0971, 0.0994, 0.0990, 0.0985])
go_val_pear  = np.array([-0.0767, 0.7496, 0.7594, 0.7675, 0.7727, 0.7727, 0.7750, 0.7769,
                         0.7795, 0.7803, 0.7792, 0.7817, 0.7822, 0.7826, 0.7817, 0.7824, 0.7828])
go_test_mse  = np.array([None, 0.1242, 0.1059, None, 0.1040, 0.1031, None, 0.1028,
                         None, None, None, None, None, 0.0972, None, None, None])
go_test_pear = np.array([None, 0.7513, 0.7611, None, 0.7745, 0.7744, None, 0.7785,
                         None, None, None, None, None, 0.7841, None, None, None])

# ── Final best metrics ───────────────────────────────────────────────
exp_best = {"Val MSE": 0.0960, "Test MSE": 0.0961, "Test Pearson": 0.7825, "epoch": 55}
go_best  = {"Val MSE": 0.0971, "Test MSE": 0.0972, "Test Pearson": 0.7841, "epoch": 65}

# ── Style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})
C_EXP = "#2196F3"    # blue
C_GO  = "#FF5722"    # orange
C_EXP_L = "#64B5F6"
C_GO_L  = "#FF8A65"

# ── Figure ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 14))
fig.suptitle("scMaize V2: scMaizeExp vs scMaizeGO — Training Performance Comparison",
             fontsize=16, fontweight="bold", y=0.98)

# ===== Panel A: Training Loss =====
ax1 = fig.add_subplot(3, 2, 1)
ax1.plot(epochs[1:], exp_loss[1:], "o-", color=C_EXP, linewidth=2, markersize=5, label="scMaizeExp")
ax1.plot(epochs[1:], go_loss[1:],  "s-", color=C_GO,  linewidth=2, markersize=5, label="scMaizeGO")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Training Loss")
ax1.set_title("A  Training Loss")
ax1.legend(frameon=True, fancybox=True, shadow=False)
ax1.grid(True, alpha=0.3); ax1.set_xlim(0, 85)

# ===== Panel B: Validation MSE =====
ax2 = fig.add_subplot(3, 2, 2)
ax2.plot(epochs, exp_val_mse, "o-", color=C_EXP, linewidth=2, markersize=5, label="scMaizeExp")
ax2.plot(epochs, go_val_mse,  "s-", color=C_GO,  linewidth=2, markersize=5, label="scMaizeGO")
# mark best
ax2.axvline(x=exp_best["epoch"], color=C_EXP, linestyle="--", alpha=0.5, linewidth=1)
ax2.axvline(x=go_best["epoch"],  color=C_GO,  linestyle="--", alpha=0.5, linewidth=1)
ax2.annotate(f"Best Exp\n(epoch {exp_best['epoch']})\nMSE={exp_best['Val MSE']:.4f}",
             xy=(exp_best["epoch"], exp_best["Val MSE"]), xytext=(exp_best["epoch"]+6, 0.12),
             arrowprops=dict(arrowstyle="->", color=C_EXP, lw=1.2), fontsize=8, color=C_EXP)
ax2.annotate(f"Best GO\n(epoch {go_best['epoch']})\nMSE={go_best['Val MSE']:.4f}",
             xy=(go_best["epoch"], go_best["Val MSE"]), xytext=(go_best["epoch"]+6, 0.13),
             arrowprops=dict(arrowstyle="->", color=C_GO, lw=1.2), fontsize=8, color=C_GO)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Validation MSE")
ax2.set_title("B  Validation MSE (lower is better)")
ax2.legend(frameon=True, fancybox=True, shadow=False)
ax2.grid(True, alpha=0.3); ax2.set_xlim(0, 85)

# ===== Panel C: Validation Pearson =====
ax3 = fig.add_subplot(3, 2, 3)
ax3.plot(epochs, exp_val_pear, "o-", color=C_EXP, linewidth=2, markersize=5, label="scMaizeExp")
ax3.plot(epochs, go_val_pear,  "s-", color=C_GO,  linewidth=2, markersize=5, label="scMaizeGO")
ax3.axvline(x=exp_best["epoch"], color=C_EXP, linestyle="--", alpha=0.5, linewidth=1)
ax3.axvline(x=go_best["epoch"],  color=C_GO,  linestyle="--", alpha=0.5, linewidth=1)
ax3.annotate(f"Best GO\n(epoch {go_best['epoch']})\nR={go_best['Test Pearson']:.4f}",
             xy=(go_best["epoch"], go_val_pear[go_best["epoch"]//5-1 if go_best["epoch"]>=5 else 0]),
             xytext=(go_best["epoch"]+6, 0.752),
             arrowprops=dict(arrowstyle="->", color=C_GO, lw=1.2), fontsize=8, color=C_GO)
ax3.set_xlabel("Epoch"); ax3.set_ylabel("Validation Pearson r")
ax3.set_title("C  Validation Pearson Correlation (higher is better)")
ax3.legend(frameon=True, fancybox=True, shadow=False)
ax3.grid(True, alpha=0.3); ax3.set_ylim(0, 0.82); ax3.set_xlim(0, 85)

# ===== Panel D: Test Set Metrics (scatter at checkpoint epochs) =====
ax4 = fig.add_subplot(3, 2, 4)
test_epochs_exp = epochs[[1,2,7,9,11]]  # epochs with test metrics
test_epochs_go  = epochs[[1,2,4,5,7,13]]

ax4.scatter(test_epochs_exp, exp_test_pear[test_epochs_exp//5 - (test_epochs_exp==0) if False else [1,2,7,9,11]],
            color=C_EXP, s=80, zorder=5, label="scMaizeExp", edgecolors="white", linewidth=0.8)
# re-do cleanly
# Exp test at epochs: 5,10,35,50,55
ax4.scatter([5, 10, 35, 50, 55],
            [0.7539, 0.7630, 0.7807, 0.7814, 0.7825],
            color=C_EXP, s=80, zorder=5, label="scMaizeExp", edgecolors="white", linewidth=0.8,
            marker="o")
ax4.plot([5, 10, 35, 50, 55], [0.7539, 0.7630, 0.7807, 0.7814, 0.7825],
         color=C_EXP, linewidth=1.5, alpha=0.7)
# GO test at epochs: 5,10,20,25,35,65
go_test_ep = [5, 10, 20, 25, 35, 65]
go_test_p  = [0.7513, 0.7611, 0.7745, 0.7744, 0.7785, 0.7841]
ax4.scatter(go_test_ep, go_test_p,
            color=C_GO, s=80, zorder=5, label="scMaizeGO", edgecolors="white", linewidth=0.8,
            marker="s")
ax4.plot(go_test_ep, go_test_p, color=C_GO, linewidth=1.5, alpha=0.7)
ax4.set_xlabel("Epoch"); ax4.set_ylabel("Test Pearson r")
ax4.set_title("D  Test Set Pearson r (checkpoint epochs)")
ax4.legend(frameon=True, fancybox=True, shadow=False)
ax4.grid(True, alpha=0.3)

# ===== Panel E: Final Best Metrics Bar Chart =====
ax5 = fig.add_subplot(3, 2, 5)
metrics_names = ["Best Val MSE", "Best Test MSE", "Best Test\nPearson r"]
exp_vals = [exp_best["Val MSE"], exp_best["Test MSE"], exp_best["Test Pearson"]]
go_vals  = [go_best["Val MSE"],  go_best["Test MSE"],  go_best["Test Pearson"]]
x = np.arange(len(metrics_names))
width = 0.35
bars1 = ax5.bar(x - width/2, exp_vals, width, color=C_EXP, label="scMaizeExp", edgecolor="white")
bars2 = ax5.bar(x + width/2, go_vals,  width, color=C_GO,  label="scMaizeGO",  edgecolor="white")
# annotate
for bar, val in zip(bars1, exp_vals):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color=C_EXP)
for bar, val in zip(bars2, go_vals):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color=C_GO)
ax5.set_xticks(x); ax5.set_xticklabels(metrics_names)
ax5.set_title("E  Best Performance Summary")
ax5.legend(frameon=True, fancybox=True, shadow=False)
ax5.grid(axis="y", alpha=0.3)
# add delta annotation
delta_mse = exp_best["Val MSE"] - go_best["Val MSE"]
delta_pear = go_best["Test Pearson"] - exp_best["Test Pearson"]
ax5.text(0.5, 0.02, f"Δ Val MSE: {delta_mse:+.4f}  |  Δ Test Pearson: {delta_pear:+.4f}",
         transform=ax5.transAxes, ha="center", fontsize=9, fontstyle="italic",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

# ===== Panel F: Summary Table =====
ax6 = fig.add_subplot(3, 2, 6)
ax6.axis("off")
table_data = [
    ["Metric",         "scMaizeExp",          "scMaizeGO",          "Winner"],
    ["Architecture",   "Expression only",     "Expression + GO",    "—"],
    ["Parameters",     "16.7M",               "16.8M",              "—"],
    ["Init Val MSE",   f"{exp_val_mse[0]:.4f}", f"{go_val_mse[0]:.4f}",  "GO (lower init)"],
    ["Best Val MSE",   f"{exp_best['Val MSE']:.4f} (epoch {exp_best['epoch']})",
                       f"{go_best['Val MSE']:.4f} (epoch {go_best['epoch']})",
                       "Exp" if exp_best['Val MSE'] < go_best['Val MSE'] else "GO"],
    ["Best Test MSE",  f"{exp_best['Test MSE']:.4f} (epoch {exp_best['epoch']})",
                       f"{go_best['Test MSE']:.4f} (epoch {go_best['epoch']})",
                       "Exp" if exp_best['Test MSE'] < go_best['Test MSE'] else "GO"],
    ["Best Test Pearson", f"{exp_best['Test Pearson']:.4f} (epoch {exp_best['epoch']})",
                       f"{go_best['Test Pearson']:.4f} (epoch {go_best['epoch']})",
                       "GO" if go_best['Test Pearson'] > exp_best['Test Pearson'] else "Exp"],
    ["Train Time",     "~6.5 days",            "~7.0 days",          "Exp (faster)"],
]
table = ax6.table(cellText=table_data, cellLoc="center", loc="center",
                  colWidths=[0.22, 0.28, 0.28, 0.22])
table.auto_set_font_size(False)
table.set_fontsize(9)
# style header
for j in range(4):
    table[0, j].set_facecolor("#37474F")
    table[0, j].set_text_props(color="white", fontweight="bold", fontsize=10)
# style rows
for i in range(1, len(table_data)):
    for j in range(4):
        if i % 2 == 0:
            table[i, j].set_facecolor("#ECEFF1")
        else:
            table[i, j].set_facecolor("#FAFAFA")
        table[i, j].set_edgecolor("#B0BEC5")
# highlight winner column
for i in range(1, len(table_data)):
    table[i, 3].set_text_props(fontweight="bold")
ax6.set_title("F  Model Comparison Summary", y=0.72)

plt.tight_layout(rect=[0, 0, 1, 0.95])

# ── Save ─────────────────────────────────────────────────────────────
out_path = "scMaize_v2_model_comparison.pdf"
plt.savefig(out_path, dpi=200, format="pdf")
print(f"PDF saved to: {out_path}")
plt.close()
