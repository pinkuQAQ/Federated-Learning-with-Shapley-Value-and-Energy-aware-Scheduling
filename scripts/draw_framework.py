"""Draw the paper's Fig. 1 as a publication-ready vector schematic."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


COLORS = {
    "ink": "#20262E",
    "muted": "#596673",
    "line": "#AAB3BC",
    "blue": "#1874C9",
    "blue_light": "#E9F3FC",
    "red": "#D93B3B",
    "red_light": "#FCEBEC",
    "green": "#2C8B57",
    "green_light": "#EAF5EE",
    "orange": "#E58B2A",
    "orange_light": "#FFF2E3",
    "cyan": "#64B9C3",
    "gray_light": "#F4F6F8",
    "gray_node": "#AEB5BC",
}


def rounded_box(ax, x, y, w, h, *, edge, face="white", lw=1.4,
                radius=0.12, linestyle="-"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        linestyle=linestyle,
        zorder=1,
    )
    ax.add_patch(patch)
    return patch


def flow_arrow(ax, start, end, *, color, lw=2.0, style="-", mutation=16,
               zorder=4):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation,
        linewidth=lw,
        linestyle=style,
        color=color,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(arrow)
    return arrow


def server_icon(ax, cx, cy):
    rounded_box(
        ax, cx - 0.60, cy - 0.42, 1.20, 0.84,
        edge=COLORS["blue"], face=COLORS["blue_light"], lw=1.6,
        radius=0.08,
    )
    for yoff in (0.23, 0.0, -0.23):
        ax.add_patch(Rectangle(
            (cx - 0.43, cy + yoff - 0.075), 0.86, 0.15,
            linewidth=0.9, edgecolor=COLORS["blue"], facecolor="white",
            zorder=3,
        ))
        ax.add_patch(Circle(
            (cx - 0.31, cy + yoff), 0.025,
            facecolor=COLORS["green"], edgecolor="none", zorder=4,
        ))


def client_icon(ax, cx, cy, selected=True):
    color = COLORS["green"] if selected else COLORS["gray_node"]
    ax.add_patch(FancyBboxPatch(
        (cx - 0.34, cy - 0.30), 0.68, 0.60,
        boxstyle="round,pad=0.015,rounding_size=0.06",
        linewidth=1.5, edgecolor=color, facecolor="white", zorder=3,
    ))
    ax.add_patch(Rectangle(
        (cx - 0.25, cy - 0.18), 0.50, 0.34,
        linewidth=0, facecolor=COLORS["gray_light"], zorder=3,
    ))
    ax.add_patch(Circle(
        (cx, cy - 0.235), 0.025, facecolor=color, edgecolor="none", zorder=4,
    ))
    ax.add_patch(Circle(
        (cx + 0.28, cy + 0.25), 0.13,
        facecolor=color, edgecolor="white", linewidth=1.0, zorder=5,
    ))
    ax.text(
        cx + 0.28, cy + 0.25, "1" if selected else "0",
        ha="center", va="center", fontsize=10, fontweight="bold",
        color="white", zorder=6,
    )


def data_histogram(ax, cx, cy, heights, colors):
    width = 0.11
    gap = 0.045
    total = len(heights) * width + (len(heights) - 1) * gap
    x0 = cx - total / 2
    for i, (height, color) in enumerate(zip(heights, colors)):
        ax.add_patch(Rectangle(
            (x0 + i * (width + gap), cy), width, height,
            linewidth=0, facecolor=color, zorder=3,
        ))


def battery_icon(ax, x, y, level):
    edge = COLORS["green"] if level >= 0.45 else COLORS["orange"]
    ax.add_patch(Rectangle(
        (x, y), 0.48, 0.21, linewidth=1.1,
        edgecolor=edge, facecolor="white", zorder=3,
    ))
    ax.add_patch(Rectangle(
        (x + 0.48, y + 0.055), 0.055, 0.10,
        linewidth=0, facecolor=edge, zorder=3,
    ))
    ax.add_patch(Rectangle(
        (x + 0.035, y + 0.035), 0.40 * level, 0.14,
        linewidth=0, facecolor=edge, zorder=4,
    ))


def signal_icon(ax, x, y, strength):
    for i in range(4):
        color = COLORS["blue"] if i < strength else "#D9E4EE"
        ax.add_patch(Rectangle(
            (x + 0.09 * i, y), 0.055, 0.07 + 0.07 * i,
            linewidth=0, facecolor=color, zorder=3,
        ))


def state_row(ax, y, symbol, label, color):
    ax.add_patch(Circle(
        (11.36, y), 0.11, facecolor=color, edgecolor="white",
        linewidth=0.8, zorder=3,
    ))
    ax.text(11.60, y, symbol, ha="left", va="center", fontsize=13,
            color=COLORS["ink"])
    ax.text(13.25, y, label, ha="left", va="center", fontsize=12.5,
            color=COLORS["muted"])


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})

fig = plt.figure(figsize=(16, 7.4), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 7.4)
ax.axis("off")

# Main layer boundaries, following the visual hierarchy of the reference figure.
rounded_box(
    ax, 0.35, 0.34, 9.65, 6.72,
    edge=COLORS["ink"], face="white", lw=1.7, radius=0.55, linestyle="--",
)
rounded_box(
    ax, 10.75, 0.34, 4.90, 6.72,
    edge=COLORS["ink"], face="white", lw=1.7, radius=0.55, linestyle="--",
)

ax.text(0.72, 6.70, "Wireless Federated Learning Layer",
        fontsize=18, fontweight="bold", color=COLORS["ink"], va="center")
ax.text(11.08, 6.70, "Shapley-Guided Scheduler",
        fontsize=17, fontweight="bold", color=COLORS["ink"], va="center")

# Server and global model.
server_x, server_y = 5.15, 5.80
server_icon(ax, server_x, server_y)
ax.text(server_x, 6.38, "Parameter Server", ha="center", va="center",
        fontsize=14.5, fontweight="bold", color=COLORS["ink"])
ax.text(server_x, 5.22, r"Global model $w_t$", ha="center", va="center",
        fontsize=13, color=COLORS["muted"])

# Clients and private non-IID data.
client_x = [1.45, 3.65, 5.85, 8.75]
client_names = ["Client 1", "Client 2", "Client 3", r"Client $N$"]
selected = [True, True, False, True]
levels = [0.82, 0.58, 0.31, 0.70]
signals = [4, 3, 2, 3]
histograms = [
    ([0.22, 0.54, 0.15, 0.42], [COLORS["red"], COLORS["blue"], COLORS["orange"], COLORS["cyan"]]),
    ([0.48, 0.18, 0.50, 0.13], [COLORS["orange"], COLORS["cyan"], COLORS["blue"], COLORS["red"]]),
    ([0.12, 0.45, 0.52, 0.20], [COLORS["blue"], COLORS["red"], COLORS["cyan"], COLORS["orange"]]),
    ([0.37, 0.16, 0.29, 0.55], [COLORS["cyan"], COLORS["orange"], COLORS["red"], COLORS["blue"]]),
]

rounded_box(
    ax, 0.72, 0.72, 8.92, 1.45,
    edge="#C7D6D9", face="#F2F8F8", lw=1.0, radius=0.55,
)
ax.text(0.92, 1.92, "Private non-IID datasets and device states",
        fontsize=12.5, fontweight="bold", color=COLORS["muted"], va="center")

for idx, (cx, name, is_selected) in enumerate(zip(client_x, client_names, selected)):
    client_icon(ax, cx, 3.34, selected=is_selected)
    ax.text(cx, 3.91, name, ha="center", va="center", fontsize=13.5,
            fontweight="bold", color=COLORS["ink"])

    data_histogram(ax, cx - 0.38, 1.06, *histograms[idx])
    battery_icon(ax, cx - 0.02, 1.08, levels[idx])
    signal_icon(ax, cx + 0.55, 1.08, signals[idx])

    ax.text(cx, 0.87, r"$D_n$", ha="center", va="center", fontsize=12.5,
            color=COLORS["muted"])

    # Internal FL communication links.
    flow_arrow(ax, (server_x - 0.12, 5.04), (cx, 3.72),
               color=COLORS["blue"], lw=1.45, mutation=12, zorder=2)
    flow_arrow(ax, (cx + 0.10, 3.70), (server_x + 0.16, 5.03),
               color=COLORS["red"], lw=1.25, style="--", mutation=11, zorder=2)

ax.text(7.22, 4.95, "Global model broadcast", fontsize=12.5,
        color=COLORS["blue"], fontweight="bold", rotation=-16,
        ha="center", va="center")
ax.text(2.95, 4.95, "Clipped local updates", fontsize=12.5,
        color=COLORS["red"], fontweight="bold", rotation=16,
        ha="center", va="center")
ax.text(7.28, 3.35, r"$\cdots$", fontsize=26, color=COLORS["ink"],
        ha="center", va="center")

# Communication legend.
flow_arrow(ax, (1.03, 0.53), (1.63, 0.53), color=COLORS["blue"],
           lw=1.5, mutation=10)
ax.text(1.72, 0.53, "model broadcast", fontsize=11.5,
        color=COLORS["muted"], va="center")
flow_arrow(ax, (3.47, 0.53), (4.07, 0.53), color=COLORS["red"],
           lw=1.35, style="--", mutation=10)
ax.text(4.16, 0.53, "update upload", fontsize=11.5,
        color=COLORS["muted"], va="center")
ax.add_patch(Circle((5.75, 0.53), 0.09, facecolor=COLORS["green"],
                    edgecolor="none", zorder=3))
ax.text(5.75, 0.53, "1", fontsize=8.5, color="white", fontweight="bold",
        ha="center", va="center")
ax.text(5.91, 0.53, "selected", fontsize=11.5,
        color=COLORS["muted"], va="center")
ax.add_patch(Circle((7.13, 0.53), 0.09, facecolor=COLORS["gray_node"],
                    edgecolor="none", zorder=3))
ax.text(7.13, 0.53, "0", fontsize=8.5, color="white", fontweight="bold",
        ha="center", va="center")
ax.text(7.29, 0.53, "not selected", fontsize=11.5,
        color=COLORS["muted"], va="center")

# Scheduler state block.
rounded_box(ax, 11.08, 4.94, 4.20, 1.32,
            edge=COLORS["line"], face=COLORS["gray_light"], lw=1.1,
            radius=0.10)
ax.text(11.32, 6.02, r"Round-$t$ scheduling states", fontsize=13.5,
        fontweight="bold", color=COLORS["ink"], va="center")
state_row(ax, 5.69, r"$\varphi_n(t)$", "contribution", COLORS["orange"])
state_row(ax, 5.37, r"$B_n(t),\ \chi_n(t)$", "battery and channel", COLORS["cyan"])
state_row(ax, 5.05, r"$Q_n(t)$", "energy pressure", COLORS["red"])

flow_arrow(ax, (13.18, 4.91), (13.18, 4.72), color=COLORS["blue"],
           lw=2.0, mutation=14)

# Feasibility, score, and Softmax sampling.
rounded_box(ax, 11.08, 3.63, 4.20, 1.05,
            edge=COLORS["blue"], face=COLORS["blue_light"], lw=1.5,
            radius=0.11)
ax.text(13.18, 4.43, "Feasibility filter and score", ha="center", va="center",
        fontsize=13.5, fontweight="bold", color=COLORS["blue"])
ax.text(13.18, 4.08,
        r"$\mathrm{Score}_n(t)=V U_n(t)-Q_n(t)E_n(t)$",
        ha="center", va="center", fontsize=13.0, color=COLORS["ink"])
ax.text(13.18, 3.78,
        r"$p_n(t)\propto\exp(\mathrm{Score}_n(t)/\beta)$",
        ha="center", va="center", fontsize=13.0, color=COLORS["ink"])

flow_arrow(ax, (13.18, 3.58), (13.18, 3.28), color=COLORS["blue"],
           lw=2.0, mutation=14)

ax.text(13.18, 3.12, "Sample $K$ clients without replacement",
        ha="center", va="center", fontsize=13.0, fontweight="bold",
        color=COLORS["ink"])
node_x = [11.63, 12.38, 13.13, 13.88, 14.63]
node_selected = [True, False, True, True, False]
for idx, (nx, keep) in enumerate(zip(node_x, node_selected), start=1):
    color = COLORS["green"] if keep else COLORS["gray_node"]
    ax.add_patch(Circle((nx, 2.69), 0.20, facecolor=color,
                        edgecolor="white", linewidth=1.0, zorder=3))
    ax.text(nx, 2.69, str(idx) if idx < 5 else r"$N$", fontsize=10.5,
            color="white", fontweight="bold", ha="center", va="center")
ax.add_patch(FancyBboxPatch(
    (11.38, 2.34), 3.60, 0.18,
    boxstyle="round,pad=0.01,rounding_size=0.06",
    linewidth=0, facecolor="#E2F1D9", zorder=2,
))
ax.text(13.18, 2.43, r"Selection vector $a_n(t)\in\{0,1\}$",
        fontsize=11.5, color=COLORS["muted"], ha="center", va="center")

flow_arrow(ax, (13.18, 2.30), (13.18, 2.00), color=COLORS["blue"],
           lw=2.0, mutation=14)

# Aggregation and state refresh.
rounded_box(ax, 11.08, 0.76, 4.20, 1.18,
            edge=COLORS["orange"], face=COLORS["orange_light"], lw=1.5,
            radius=0.11)
ax.text(13.18, 1.70, "Aggregation and state refresh", ha="center", va="center",
        fontsize=13.5, fontweight="bold", color="#B85F10")
ax.text(13.18, 1.38, "Clipped FedAvg + equivalent channel noise",
        ha="center", va="center", fontsize=12.2, color=COLORS["ink"])
ax.text(13.18, 1.05,
        r"$w_{t+1},\ \varphi_n(t+1),\ B_n(t+1),\ Q_n(t+1)$",
        ha="center", va="center", fontsize=12.8, color=COLORS["ink"])

# Cross-layer interaction arrows, as in the reference's physical/logical split.
flow_arrow(ax, (10.76, 4.87), (9.99, 4.87), color=COLORS["blue"],
           lw=4.0, mutation=22, zorder=7)
ax.text(9.88, 5.10, "selected set", ha="right", va="bottom",
        fontsize=10.5, fontweight="bold", color=COLORS["blue"])

flow_arrow(ax, (10.00, 2.08), (10.76, 2.08), color=COLORS["red"],
           lw=4.0, mutation=22, zorder=7)
ax.text(9.88, 1.83, "clipped updates", ha="right", va="top",
        fontsize=10.5, fontweight="bold", color=COLORS["red"])

# A restrained loop indicates that updated states are reused in the next round.
loop = FancyArrowPatch(
    (15.43, 1.28), (15.43, 5.60),
    arrowstyle="-|>", mutation_scale=13, linewidth=1.2,
    linestyle="--", color=COLORS["muted"], zorder=2,
)
ax.add_patch(loop)
ax.text(15.54, 3.42, "next round", rotation=90, fontsize=10.5,
        color=COLORS["muted"], ha="center", va="center")

output = Path("latex/figures/framework_overview.pdf")
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)
print(f"Saved {output}")
