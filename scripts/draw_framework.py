"""
Two-column FL framework diagram
Left:  Wireless FL Clients (Non-IID Data / Residual Energy / Channel State)
Right: Server-side Scheduler (4 steps + Round Outputs)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import matplotlib.colors as mcolors
import numpy as np

# ─── colour palette ────────────────────────────────────────────────────────
C = {
    'blue':   '#1565C0',
    'teal':   '#00796B',
    'green':  '#2E7D32',
    'orange': '#E65100',
    'purple': '#6A1B9A',
    'dark':   '#263238',
    'gray':   '#546E7A',
    'lgray':  '#ECEFF1',
    'sep':    '#B0BEC5',
}

# ─── icon helpers ──────────────────────────────────────────────────────────

def noniid_icon(ax, cx, cy, r=0.22):
    """Scatter-dot pattern representing heterogeneous data."""
    offsets = [(-0.55, 0.30), (0.05, 0.52), (0.55, 0.18),
               (-0.38,-0.30), (0.22,-0.42), (0.50,-0.08)]
    sizes   = [40, 28, 36, 30, 22, 38]
    colors  = [C['orange'], C['blue']] * 3
    for (dx, dy), s, c in zip(offsets, sizes, colors):
        ax.scatter(cx + dx*r, cy + dy*r, s=s, c=c, zorder=6,
                   edgecolors='none', clip_on=False)

def battery_icon(ax, cx, cy, level=0.65, color=None, w=0.56, h=0.28):
    color = color or C['green']
    ax.add_patch(Rectangle((cx-w/2, cy-h/2), w, h,
                            lw=1.8, edgecolor=color, facecolor='white', zorder=5))
    ax.add_patch(Rectangle((cx+w/2, cy-h/6), w*0.1, h/3,
                            lw=0, facecolor=color, zorder=5))
    ax.add_patch(Rectangle((cx-w/2+0.025, cy-h/2+0.035),
                            max(0, (w-0.05)*level), h-0.07,
                            lw=0, facecolor=color, alpha=0.85, zorder=6))

def signal_icon(ax, cx, cy, strength=3, color=None, size=0.30):
    color = color or C['blue']
    n, bw, sp = 4, size*0.17, size*0.11
    tw = n*bw + (n-1)*sp
    x0 = cx - tw/2
    for i in range(n):
        bh = size*(0.25 + i*0.25)
        ax.add_patch(Rectangle((x0 + i*(bw+sp), cy - size*0.5),
                                bw, bh, lw=0,
                                facecolor=color,
                                alpha=1.0 if i < strength else 0.18,
                                zorder=5))

def status_icon(ax, cx, cy, scheduled=True, r=0.32):
    color = C['green'] if scheduled else C['gray']
    ax.add_patch(Circle((cx, cy), r, facecolor=color,
                         edgecolor='white', lw=1.2, zorder=6))
    sym = u'✓' if scheduled else u'✗'
    ax.text(cx, cy, sym, ha='center', va='center',
            fontsize=22, color='white', fontweight='bold', zorder=7)

def step_box(ax, x, y, w, h, num, title, subtitle, color):
    bg = mcolors.to_rgba(color, 0.10)
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                 boxstyle='round,pad=0.07', lw=1.8,
                                 edgecolor=color, facecolor=bg, zorder=3))
    ax.add_patch(Circle((x+0.28, y+h/2), 0.18,
                         facecolor=color, edgecolor='white', lw=1.2, zorder=4))
    ax.text(x+0.28, y+h/2, str(num),
            ha='center', va='center', fontsize=13,
            color='white', fontweight='bold', zorder=5)
    ax.text(x+0.58, y+h/2 + h*0.14, title,
            ha='left', va='center', fontsize=13,
            color=color, fontweight='bold', zorder=4)
    ax.text(x+0.58, y+h/2 - h*0.20, subtitle,
            ha='left', va='center', fontsize=13, color=C['gray'], zorder=4)

def down_arrow(ax, x, y_top, y_bot, color=C['dark'], lw=1.6):
    ax.annotate('', xy=(x, y_bot), xytext=(x, y_top),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                mutation_scale=14),
                zorder=4)

# ─── figure setup ──────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 9))
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(-0.35, 16.35)
ax.set_ylim(-0.2, 9.2)
ax.axis('off')
fig.patch.set_facecolor('white')

# ══════════════════════════════════════════════════════════════════
# LEFT PANEL  —  Wireless FL Clients
# ══════════════════════════════════════════════════════════════════
LP_X, LP_Y, LP_W, LP_H = 0.15, 0.25, 6.55, 8.50

ax.add_patch(FancyBboxPatch((LP_X, LP_Y), LP_W, LP_H,
                             boxstyle='round,pad=0.18', lw=2.0,
                             edgecolor=C['dark'], facecolor='white',
                             linestyle='--', zorder=1))

ax.text(LP_X + LP_W/2, LP_Y + LP_H - 0.32, 'Wireless FL Clients',
        ha='center', va='center', fontsize=13.5,
        fontweight='bold', color=C['dark'])

# column header strip
ax.add_patch(FancyBboxPatch((LP_X+0.12, 6.95), LP_W-0.24, 0.70,
                             boxstyle='round,pad=0.04', lw=0,
                             facecolor=C['lgray'], zorder=2))

CX = dict(lbl=1.35, dat=2.80, bat=4.05, sig=5.20, sta=6.18)

for x, txt in [(CX['lbl'], 'Client'),
               (CX['dat'], 'Non-IID\nData'),
               (CX['bat'], 'Residual\nEnergy'),
               (CX['sig'], 'Channel\nState'),
               (CX['sta'], 'Status')]:
    ax.text(x, 7.30, txt, ha='center', va='center',
            fontsize=13, fontweight='bold', color=C['dark'],
            linespacing=1.25)

ax.plot([LP_X+0.18, LP_X+LP_W-0.18], [6.92, 6.92],
        '-', color=C['sep'], lw=1.0, zorder=2)

# client rows  (name, battery_level, signal_strength, scheduled)
rows = [
    ('Client 1', 0.80, 3, True),
    ('Client 2', 0.55, 3, True),
    ('Client 3', 0.25, 2, False),
]
ROW_Y = [5.90, 4.45, 2.98]

for cy, (name, blv, sig, sched) in zip(ROW_Y, rows):
    ax.text(CX['lbl'], cy, name, ha='center', va='center',
            fontsize=13, fontweight='bold', color=C['dark'])
    noniid_icon(ax, CX['dat'],  cy)
    battery_icon(ax, CX['bat'], cy, level=blv)
    signal_icon( ax, CX['sig'], cy, strength=sig)
    status_icon( ax, CX['sta'], cy, scheduled=sched)

# row separators
for sy in [3.72, 5.18]:
    ax.plot([LP_X+0.45, LP_X+LP_W-0.45], [sy, sy],
            '--', color='#CFD8DC', lw=0.8, zorder=2)

# legend
LEG_Y = 1.55
ax.text(LP_X+0.35, LEG_Y+0.50, 'Legend:',
        fontsize=13, fontweight='bold', color=C['dark'])
status_icon(ax, LP_X+0.58, LEG_Y, scheduled=True,  r=0.16)
ax.text(LP_X+0.82, LEG_Y, 'Scheduled',
        fontsize=13, va='center', color=C['dark'])
status_icon(ax, LP_X+2.30, LEG_Y, scheduled=False, r=0.16)
ax.text(LP_X+2.54, LEG_Y, 'Unscheduled',
        fontsize=13, va='center', color=C['dark'])

# ══════════════════════════════════════════════════════════════════
# RIGHT PANEL  —  Server-side Scheduler
# ══════════════════════════════════════════════════════════════════
RP_X, RP_Y, RP_W, RP_H = 9.30, 0.25, 6.55, 8.50

ax.add_patch(FancyBboxPatch((RP_X, RP_Y), RP_W, RP_H,
                             boxstyle='round,pad=0.18', lw=2.0,
                             edgecolor=C['dark'], facecolor='white',
                             linestyle='--', zorder=1))

ax.text(RP_X + RP_W/2, RP_Y + RP_H - 0.32, 'Server-side Scheduler',
        ha='center', va='center', fontsize=13.5,
        fontweight='bold', color=C['dark'])

# steps
# SH=0.80, GAP=0.40 → center spacing=1.20
SW, SH = RP_W - 0.55, 0.80
STEPS = [
    (7.20, C['teal'],   'Energy Feasibility Filter',
                      'Remove energy-insufficient clients'),
    (6.00, C['blue'],   'Persistent Shapley Estimate',
                      'Estimate contribution scores'),
    (4.80, C['purple'], 'Score-based Client Scheduling',
                      'Select score-biased clients'),
    (3.60, C['orange'], 'Clipped FedAvg + Channel Noise',
                      'Aggregate with equivalent perturbation'),
]

for idx, (sy, col, title, sub) in enumerate(STEPS):
    step_box(ax, RP_X+0.28, sy-SH/2, SW, SH, idx+1, title, sub, col)
    if idx < len(STEPS)-1:
        curr_bot  = sy - SH/2
        next_top  = STEPS[idx+1][0] + SH/2
        down_arrow(ax, RP_X+RP_W/2, curr_bot-0.04, next_top+0.04)

# round outputs box — top = step4_bottom - 0.40, height = 1.60
OUT_TOP = STEPS[-1][0] - SH/2 - 0.40
OUT_BOT = OUT_TOP - 2.00
ax.add_patch(FancyBboxPatch((RP_X+0.22, OUT_BOT), RP_W-0.44, OUT_TOP-OUT_BOT,
                             boxstyle='round,pad=0.10', lw=1.6,
                             edgecolor=C['blue'],
                             facecolor=mcolors.to_rgba(C['blue'], 0.06),
                             zorder=2))

ax.text(RP_X+RP_W/2, OUT_TOP-0.28, 'Round Outputs',
        ha='center', va='center', fontsize=15,
        fontweight='bold', color=C['blue'], zorder=3)

outputs = [
    (r'$w_{t+1}$',              'Updated Global Model'),
    (r'$\hat{\varphi}_i(t+1)$', 'Updated Shapley Estimates'),
    (r'$\mathcal{S}_{t+1}$',    'Next-round Client Set'),
]
for j, (fml, lbl) in enumerate(outputs):
    oy = OUT_TOP - 0.68 - j*0.47
    ax.add_patch(Circle((RP_X+0.47, oy), 0.09, facecolor=C['blue'], zorder=4))
    ax.text(RP_X+0.65, oy, f'{fml}  —  {lbl}',
            ha='left', va='center', fontsize=13, color=C['dark'], zorder=4)

# arrow from last step to outputs
last_bot = STEPS[-1][0] - SH/2
down_arrow(ax, RP_X+RP_W/2, last_bot-0.04, OUT_TOP+0.04)

# ══════════════════════════════════════════════════════════════════
# MIDDLE ARROWS
# ══════════════════════════════════════════════════════════════════
MX_L = LP_X + LP_W + 0.10
MX_R = RP_X - 0.10
MX_C = (MX_L + MX_R) / 2

# ── Global Model Broadcast  (server → clients, i.e. pointing LEFT) ──
GMB_Y = 6.60
ax.annotate('', xy=(MX_L, GMB_Y), xytext=(MX_R, GMB_Y),
            arrowprops=dict(arrowstyle='->', color=C['blue'], lw=2.8,
                            mutation_scale=20),
            zorder=5)
ax.text(MX_C, GMB_Y+0.32, 'Global Model\nBroadcast',
        ha='center', va='bottom', fontsize=13,
        fontweight='bold', color=C['blue'], linespacing=1.3)
ax.text(MX_C, GMB_Y-0.22, r'$w_t$',
        ha='center', va='top', fontsize=13, color=C['blue'])

# ── Local Update Upload  (clients → server, i.e. pointing RIGHT) ──
LUU_Y = 3.00
ax.annotate('', xy=(MX_R, LUU_Y), xytext=(MX_L, LUU_Y),
            arrowprops=dict(arrowstyle='->', color=C['teal'], lw=2.8,
                            mutation_scale=20),
            zorder=5)
ax.text(MX_C, LUU_Y+0.32, 'Local Update\nUpload',
        ha='center', va='bottom', fontsize=13,
        fontweight='bold', color=C['teal'], linespacing=1.3)
ax.text(MX_C, LUU_Y-0.22, r'$\Delta w_n$',
        ha='center', va='top', fontsize=13, color=C['teal'])

# ─── save ──────────────────────────────────────────────────────────────────
out_pdf = 'latex/figures/framework_overview.pdf'
out_png = 'latex/figures/配图_new.png'
plt.savefig(out_pdf, bbox_inches='tight', dpi=300)
plt.savefig(out_png, bbox_inches='tight', dpi=200)
print(f'Saved  {out_pdf}')
print(f'Saved  {out_png}')
