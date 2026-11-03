#!/usr/bin/env python3
"""
The model's layers, drawn the way the transformer paper draws its own: a box
per layer, a circled plus for each residual add, a shaded group for whatever
repeats - and beside it, the inside of one attention layer.

Structure only - no paging, no voting. Every shape follows minagi/recur.py,
minagi/model.py and minagi/pool.py, and every number is read from the
weights' manifest, so the one number that moves - how many experts the pool
holds - is current whenever this is run. release.sh runs it before a push.

    python3 tools/draw_layers.py                            # assets/layers.svg
    python3 tools/draw_layers.py --out runs/layers.svg

At README width the smallest labels are small; the figure is an SVG, so
opening it full size or zooming keeps them sharp.
"""
import argparse
import json
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INK, GREY = "#111111", "#555555"
FILL = dict(emb="#f8dcdc", norm="#f3f2c6", attn="#fde3c4", ff="#cfe6f3",
            lin="#e2e0f1", soft="#d5ebd7", cat="#ececec", group="#f2f2f2")
# Times New Roman where it exists, and the fonts drawn to its metrics where it
# does not, so the labels keep their widths in any browser
SERIF = ("'Times New Roman', Times, 'Liberation Serif', Tinos, 'TeX Gyre Termes', "
         "'Nimbus Roman', serif")
HEAD = ["#e07b7b", "#e8a35c", "#d4c04a", "#7fbf6a", "#4fb3a9", "#5f93d6", "#9a7fd4", "#d277b4"]


def _italic_measure():
    """Width of a string in the italic the notes are set in - for wrapping."""
    try:
        from PIL import ImageFont
        path = subprocess.run(["fc-match", "-f", "%{file}", "Times New Roman:italic"],
                              capture_output=True, text=True).stdout.strip()
        f = ImageFont.truetype(path, 200)
        return lambda s, size: f.getlength(s) * size / 200
    except Exception:
        return lambda s, size: 0.42 * size * len(s)


class Fig:
    def __init__(self):
        self.out = []

    def add(self, s):
        self.out.append(s)

    def text(self, x, y, s, size=19, italic=False, fill=INK, anchor="middle"):
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        it = ' font-style="italic"' if italic else ""
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{SERIF}" font-size="{size}" '
                 f'fill="{fill}" text-anchor="{anchor}"{it}>{s}</text>')

    def box(self, cx, y, w, h, fill, lines, sub=None, size=19):
        """A layer: a rounded box, its name, and its shape in small italic."""
        self.add(f'<rect x="{cx - w / 2:.1f}" y="{y:.1f}" width="{w}" height="{h}" rx="7" '
                 f'fill="{fill}" stroke="#333333" stroke-width="1.3"/>')
        lh = size * 1.12
        total = len(lines) * lh + (15 if sub else 0)
        y0 = y + (h - total) / 2 + size * 0.82
        for i, ln in enumerate(lines):
            self.text(cx, y0 + i * lh, ln, size)
        if sub:
            self.text(cx, y0 + len(lines) * lh - size * 0.82 + 13, sub, 13.5, True, GREY)

    def line(self, pts, arrow=True, sw=2.0):
        if arrow:
            (x1, y1), (x2, y2) = pts[-2], pts[-1]
            ang = np.arctan2(y2 - y1, x2 - x1)
            hl = 10
            pts = pts[:-1] + [(x2 - hl * np.cos(ang), y2 - hl * np.sin(ang))]
            l_, r_ = ang + np.pi * 0.86, ang - np.pi * 0.86
            tri = [(x2, y2), (x2 + hl * np.cos(l_), y2 + hl * np.sin(l_)),
                   (x2 + hl * np.cos(r_), y2 + hl * np.sin(r_))]
        d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
        self.add(f'<path d="{d}" fill="none" stroke="{INK}" stroke-width="{sw}" stroke-linejoin="round"/>')
        if arrow:
            self.add('<polygon points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in tri)
                     + f'" fill="{INK}"/>')

    def dot(self, x, y):
        self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="{INK}"/>')

    def plus(self, cx, cy, r=11):
        self.add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="white" stroke="{INK}" stroke-width="1.6"/>')
        self.add(f'<path d="M {cx - r} {cy} L {cx + r} {cy} M {cx} {cy - r} L {cx} {cy + r}" '
                 f'stroke="{INK}" stroke-width="1.4"/>')

    def rope(self, cx, cy, r=9):
        """The paper's positional-encoding circle, on a query or key arrow: here the
        positions are rotary, applied inside attention, not added to the input."""
        self.add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="white" stroke="{INK}" stroke-width="1.4"/>')
        self.add(f'<path d="M {cx - 5.5:.1f} {cy + 1.5:.1f} C {cx - 3.5:.1f} {cy - 4:.1f} {cx - 1:.1f} '
                 f'{cy - 4:.1f} {cx:.1f} {cy:.1f} C {cx + 1:.1f} {cy + 4:.1f} {cx + 3.5:.1f} {cy + 4:.1f} '
                 f'{cx + 5.5:.1f} {cy - 1.5:.1f}" fill="none" stroke="{INK}" stroke-width="1.4"/>')

    def group(self, x0, y0, x1, y1):
        self.add(f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" rx="12" '
                 f'fill="{FILL["group"]}" stroke="#333333" stroke-width="1.4"/>')

    def attention(self, cx, y_top, w, y_norm_top, h=58):
        """Masked self-attention fed V, K, Q from the norm below; rotary on K and Q."""
        self.box(cx, y_top, w, h, FILL["attn"], ["Masked multihead", "self-attention"])
        yb = y_top + h
        split_y = y_norm_top - 12
        self.line([(cx, y_norm_top), (cx, split_y)], arrow=False)
        for dx, lab in ((-50, "V"), (0, "K"), (50, "Q")):
            self.line([(cx, split_y), (cx + dx, split_y), (cx + dx, yb)])
            self.text(cx + dx - 9, yb + 16, lab, 15, True, anchor="end")
            if lab in ("K", "Q"):
                self.rope(cx + dx, yb + 31)
        self.dot(cx, split_y)


def draw(man):
    """The layer stack and, beside it, the inside of one attention layer.
    Returns the figure and the box it occupies."""
    cfg = man["cfg"]
    assert cfg["n_recur"] == 1 and cfg["n_coda"] == 0 and cfg["tie_embeddings"], \
        "this drawing is of one recurrent block, no coda, tied embeddings"
    n_exp = int(man["n_experts"])
    d, dff, vocab = cfg["d_model"], cfg["d_ff"], cfg["vocab_size"]
    heads, rows, k = cfg["n_head"], cfg["max_steps"], cfg["pool_top_k"]
    width, prelude, halt = cfg["pool_d_ff"], cfg["n_prelude"], cfg["halt_thresh"]
    context = man.get("context_now") or cfg["block"]
    f = Fig()
    H = 1110
    GAP_QKV = 66                          # norm top to attention bottom: V K Q, rotary circles

    # ---------------------------------------------------------------- the prelude
    L, LW = 320, 200
    y = H - 58
    f.text(L, y, "Input characters", 19)
    f.text(L, y + 20, f"{vocab} symbols: 256 bytes + {vocab - 256} markers", 13.5, True, GREY)
    emb_top = y - 26 - 44
    f.line([(L, y - 22), (L, emb_top + 44)])
    f.box(L, emb_top, LW, 44, FILL["emb"], ["Input Embedding"], f"{vocab} → {d}")
    g1 = emb_top - 22
    s1 = g1 - 16                                   # residual 1 starts here
    n1 = s1 - 14 - 36
    a1 = n1 - GAP_QKV - 58
    p1 = a1 - 16 - 11
    s2 = p1 - 11 - 12                              # residual 2 starts here
    n2 = s2 - 14 - 36
    f1 = n2 - 22 - 50
    p2 = f1 - 16 - 11
    g0 = p2 - 11 - 18
    f.group(L - 152, g0, L + 116, g1)
    f.line([(L, emb_top), (L, s1)], arrow=False)
    f.dot(L, s1)
    f.box(L, n1, LW, 36, FILL["norm"], ["Norm"])
    f.line([(L, s1), (L, n1 + 36)])
    f.attention(L, a1, LW, n1)
    f.plus(L, p1)
    f.line([(L, a1), (L, p1 + 11)])
    f.line([(L, s1), (L - 132, s1), (L - 132, p1), (L - 11, p1)])           # round the attention
    f.line([(L, p1 - 11), (L, s2)], arrow=False)
    f.dot(L, s2)
    f.box(L, n2, LW, 36, FILL["norm"], ["Norm"])
    f.line([(L, s2), (L, n2 + 36)])
    f.box(L, f1, LW, 50, FILL["ff"], ["Feed Forward"], f"SwiGLU {d}→{dff}→{d}")
    f.line([(L, n2), (L, f1 + 50)])
    f.plus(L, p2)
    f.line([(L, f1), (L, p2 + 11)])
    f.line([(L, s2), (L - 132, s2), (L - 132, p2), (L - 11, p2)])           # round the feed-forward
    mid = (g0 + g1) / 2
    f.text(L - 162, mid - 14, f"{prelude}×", 23, True, anchor="end")
    f.text(L - 162, mid + 10, "prelude", 19, True, anchor="end")
    f.text(L - 162, mid + 32, "layer", 19, True, anchor="end")
    x_top = g0 - 26                                # x leaves the prelude here

    # ---------------------------------------------------------------- the recurrent layer
    R, RW = 790, 216
    XL = 480                                       # x runs down between the columns
    gr1 = s1 - 10
    cat_t = gr1 - 20 - 36
    ad_t = cat_t - 24 - 46                         # the adapter: [h ; x] -> d
    rs1 = ad_t - 14
    rn1 = rs1 - 14 - 36
    ra1 = rn1 - GAP_QKV - 58
    rp1 = ra1 - 16 - 11
    rs2 = rp1 - 11 - 12
    rn2 = rs2 - 14 - 36
    moe_h = 96
    rbr = rn2 - 13                                 # where the router's input branches off
    moe_t = rn2 - 26 - moe_h
    rp2 = moe_t - 16 - 11
    gr0 = rp2 - 11 - 18
    f.group(530, gr0, R + 146, gr1)
    f.box(R, cat_t, 124, 36, FILL["cat"], ["Concat"])
    cat_mid = cat_t + 18
    f.line([(L, g0), (L, x_top), (XL, x_top), (XL, cat_mid), (R - 62, cat_mid)])
    f.text(XL - 12, (x_top + cat_mid) / 2 + 8, "x", 23, True, anchor="end")
    f.box(R, ad_t, RW, 46, FILL["lin"], ["Linear"], f"{2 * d}→{d}")
    f.line([(R, cat_t), (R, ad_t + 46)])
    f.line([(R, ad_t), (R, rs1)], arrow=False)
    f.dot(R, rs1)
    f.box(R, rn1, RW, 36, FILL["norm"], ["Norm"])
    f.line([(R, rs1), (R, rn1 + 36)])
    f.attention(R, ra1, RW, rn1)
    f.plus(R, rp1)
    f.line([(R, ra1), (R, rp1 + 11)])
    f.line([(R, rs1), (R + 128, rs1), (R + 128, rp1), (R + 11, rp1)])       # round the attention
    f.line([(R, rp1 - 11), (R, rs2)], arrow=False)
    f.dot(R, rs2)
    f.box(R, rn2, RW, 36, FILL["norm"], ["Norm"])
    f.line([(R, rs2), (R, rn2 + 36)])
    # the mixture of experts, and the router that weighs them
    f.add(f'<rect x="{R - RW / 2:.1f}" y="{moe_t:.1f}" width="{RW}" height="{moe_h}" rx="7" '
          f'fill="{FILL["ff"]}" stroke="#333333" stroke-width="1.3"/>')
    f.text(R, moe_t + 27, "Mixture of Experts", 19)
    sq, sg, nsq = 11, 4, 12                        # a few of the pool, the chosen ones dark
    x0 = R - (nsq * (sq + sg) - sg) / 2
    for i in range(nsq):
        f.add(f'<rect x="{x0 + i * (sq + sg):.1f}" y="{moe_t + 37}" width="{sq}" height="{sq}" rx="2" '
              f'fill="{"#5b8fb3" if i in {0, 2, 3, 5, 7, 8, 9, 11} else "white"}" '
              f'stroke="#5b8fb3" stroke-width="1"/>')
    f.text(R, moe_t + 68, f"top {k} of {n_exp} experts", 13.5, True, GREY)
    f.text(R, moe_t + 85, f"each a SwiGLU {d}→{width}→{d}", 13.5, True, GREY)
    f.line([(R, rn2), (R, moe_t + moe_h)])
    RTx = 612
    f.dot(R, rbr)
    f.line([(R, rbr), (RTx, rbr), (RTx, moe_t + 59)])
    f.box(RTx, moe_t + 23, 104, 36, FILL["lin"], ["Router"])
    f.line([(RTx + 52, moe_t + 41), (R - RW / 2, moe_t + 41)])
    f.plus(R, rp2)
    f.line([(R, moe_t), (R, rp2 + 11)])
    f.line([(R, rs2), (R + 128, rs2), (R + 128, rp2), (R + 11, rp2)])       # round the experts
    # out of the layer comes h: up to the readout, and round again for the next row
    h_dot = gr0 - 14
    f.line([(R, rp2 - 11), (R, h_dot)], arrow=False)
    f.dot(R, h_dot)
    LOOP = R + 178
    f.line([(R, h_dot), (LOOP, h_dot), (LOOP, cat_mid), (R + 62, cat_mid)])
    f.text(LOOP - 9, (gr0 + gr1) / 2 + 30, "h", 23, True, anchor="end")
    gm = (gr0 + gr1) / 2
    f.text(LOOP + 18, gm - 26, f"up to {rows}×", 23, True, anchor="start")
    f.text(LOOP + 18, gm - 2, "recurrent layer", 19, True, anchor="start")
    f.text(LOOP + 18, gm + 20, "the same weights", 15, True, GREY, anchor="start")
    f.text(LOOP + 18, gm + 38, "at every row", 15, True, GREY, anchor="start")
    f.text(LOOP + 18, cat_mid + 5, "h = 0 at row 1", 14, True, GREY, anchor="start")
    # the readout, after every row
    rn_t = h_dot - 14 - 36
    f.box(R, rn_t, RW, 36, FILL["norm"], ["Norm"])
    f.line([(R, h_dot), (R, rn_t + 36)])
    hb = rn_t - 15                                 # the halting head branches off here
    lin_t = rn_t - 30 - 46
    f.line([(R, rn_t), (R, lin_t + 46)])
    f.dot(R, hb)
    f.box(R, lin_t, RW, 46, FILL["lin"], ["Linear"], "tied to the input embedding")
    sm_t = lin_t - 22 - 36
    f.box(R, sm_t, RW, 36, FILL["soft"], ["Softmax"])
    f.line([(R, lin_t), (R, sm_t + 36)])
    f.line([(R, sm_t), (R, sm_t - 22)])
    f.text(R, sm_t - 46, "Output Probabilities", 19)
    f.text(R, sm_t - 28, "for the next character", 13.5, True, GREY)
    HX = 1092
    f.line([(R, hb), (HX, hb), (HX, lin_t + 46)])
    f.box(HX, lin_t, 132, 46, FILL["lin"], ["Linear"], f"{d}→1")
    f.box(HX, sm_t, 132, 36, FILL["soft"], ["Sigmoid"])
    f.line([(HX, lin_t), (HX, sm_t + 36)])
    f.line([(HX, sm_t), (HX, sm_t - 22)])
    f.text(HX, sm_t - 46, "Halt probability", 19)
    f.text(HX, sm_t - 28, f"stop once the running total ≥ {halt}", 13.5, True, GREY)
    f.text(1160, H - 50, f"Width {d} throughout.  Attention: {heads} heads, causal.", 14, True, GREY,
           anchor="end")
    f.text(1160, H - 30, "~ rotary positions on queries and keys.  Norm: RMSNorm.", 14, True, GREY,
           anchor="end")
    top = sm_t - 70

    # ---------------------------------------------------------------- inside one attention
    # minagi/model.py Attention: one Linear makes q, k and v (d -> 3d); each is
    # split into n_head heads of d/n_head; rotary positions turn q and k; every
    # head runs scaled dot-product attention under the causal mask against
    # this layer's cache; the heads' outputs are joined back to d and one more
    # Linear (proj) maps d -> d into the residual add.
    hd = d // heads
    n_cache = prelude + rows                       # model.n_slots(): one cache per application
    measure = _italic_measure()
    FX0, FX1 = 1226, 1886
    ICX = (FX0 + FX1) / 2
    FT, FB = sm_t - 58, H - 12
    f.add(f'<rect x="{FX0}" y="{FT:.1f}" width="{FX1 - FX0}" height="{FB - FT:.1f}" rx="14" '
          f'fill="#fbfbfb" stroke="#9a9a9a" stroke-width="1.4" stroke-dasharray="6 4"/>')
    f.text(ICX, FT + 34, "Inside each masked multihead self-attention", 21)
    f.text(ICX, FT + 55, f"{heads} heads in each of the {prelude + 1} attention layers: {prelude} in the "
           "prelude, 1 in the recurrent layer", 13.5, True, GREY)
    f.text(ICX, FT + 72, f"run up to {prelude} + {rows} = {n_cache} times per character, so up to "
           f"{n_cache * heads} head runs", 13.5, True, GREY)

    # how one head moves information, for one character
    f.text(ICX, FT + 100, "How one head moves information into one character", 16, True)
    chars = list("The cat sa")
    bw, bg = 30, 7
    cx0 = ICX - (len(chars) * (bw + bg) - bg) / 2
    cy = FT + 218
    cur_i = len(chars) - 1
    xc = lambda i: cx0 + i * (bw + bg) + bw / 2                    # noqa: E731
    weights = [0.04, 0.03, 0.10, 0.07, 0.30, 0.05, 0.08, 0.06, 0.20, 0.07]   # an example, not measured
    for i, w_ in enumerate(weights[:-1]):
        x1, x2 = xc(i), xc(cur_i) - 4
        hgt = 16 + (x2 - x1) * 0.27
        f.add(f'<path d="M {x1:.1f} {cy - 2} Q {(x1 + x2) / 2:.1f} {cy - 2 - 2 * hgt:.1f} {x2:.1f} {cy - 2}" '
              f'fill="none" stroke="{HEAD[5]}" stroke-width="{1 + 14 * w_:.1f}" stroke-opacity="0.85" '
              f'stroke-linecap="round"/>')
    xs = xc(cur_i)                                 # the character reading itself
    f.add(f'<path d="M {xs + 4:.1f} {cy - 2} C {xs + 22:.1f} {cy - 40} {xs - 22:.1f} {cy - 40} '
          f'{xs - 4:.1f} {cy - 2}" fill="none" stroke="{HEAD[5]}" '
          f'stroke-width="{1 + 14 * weights[cur_i]:.1f}" stroke-opacity="0.85"/>')
    for i, c in enumerate(chars):
        cur = i == cur_i
        f.add(f'<rect x="{cx0 + i * (bw + bg):.1f}" y="{cy}" width="{bw}" height="32" rx="5" '
              f'fill="{"#fde3c4" if cur else "white"}" stroke="#333333" stroke-width="{1.8 if cur else 1.1}"/>')
        f.text(xc(i), cy + 23, "␣" if c == " " else c, 19)
    f.add(f'<rect x="{xc(cur_i) + bw / 2 + 8:.1f}" y="{cy}" width="{2 * (bw + bg) - bg}" height="32" rx="5" '
          f'fill="none" stroke="#bbbbbb" stroke-width="1.1" stroke-dasharray="4 3"/>')
    f.text(xc(cur_i) + bw / 2 + 8 + (2 * (bw + bg) - bg) / 2, cy + 21, "not yet", 13, True, "#999999")
    ny = cy + 56
    for s_ in ("The last character's query scores its own key and every earlier one's. Softmax",
               "turns the scores into weights (line width; these are an example) and the head",
               "takes that mix of their values. Later characters are masked out. Each of the",
               f"{heads} heads has its own weights, so each looks for something different."):
        f.text(ICX, ny, s_, 13.5, True, GREY)
        ny += 18

    # the path through the layer, bottom to top
    y_out = FT + 360
    y_proj = FT + 384
    y_cat = y_proj + 46 + 26
    y_stack_top = y_cat + 46 + 30
    STK = 6                                        # offset between the stacked heads
    y_front = y_stack_top + STK * (heads - 1)
    front_h = 92
    cell_w, cell_h, nrow = 11, 10, 10              # a cache: heads across, characters down
    cache_h = nrow * cell_h
    y_cache = y_front + front_h + 38
    y_rope = y_cache + cache_h + 34
    y_bars = y_rope + 22
    bar_h = 16
    y_fork = y_bars + bar_h + 24
    y_qkv = y_fork + 20
    y_in = y_qkv + 46 + 26
    assert y_in + 6 < FB, (y_in, FB)
    x_q, x_k, x_v = ICX - 220, ICX - 60, ICX + 100
    seg = 14

    f.text(ICX, y_in, f"from the Norm below: {d} numbers per character", 15, True)
    f.line([(ICX, y_in - 12), (ICX, y_qkv + 46)])
    f.box(ICX, y_qkv, 250, 46, FILL["lin"], ["Linear"], f"{d} → 3 × {d}: query, key, value")
    f.line([(ICX, y_qkv), (ICX, y_fork)], arrow=False)
    f.dot(ICX, y_fork)
    for xx in (x_q, x_k, x_v):
        f.line([(ICX, y_fork), (xx, y_fork), (xx, y_bars + bar_h)])
    for xx, lab in ((x_q, "Q"), (x_k, "K"), (x_v, "V")):            # each split into the heads
        bx = xx - heads * seg / 2
        for j in range(heads):
            f.add(f'<rect x="{bx + j * seg:.1f}" y="{y_bars}" width="{seg}" height="{bar_h}" '
                  f'fill="{HEAD[j % len(HEAD)]}" stroke="#333333" stroke-width="0.8"/>')
        f.text(bx - 8, y_bars + 14, lab, 19, True, anchor="end")
    xr = x_v + heads * seg / 2 + 14
    f.text(xr, y_bars + 4, "each split into", 13.5, True, GREY, anchor="start")
    f.text(xr, y_bars + 21, f"{heads} heads × {hd} numbers", 13.5, True, GREY, anchor="start")
    for xx in (x_q, x_k):                                            # rotary on q and k
        f.line([(xx, y_bars), (xx, y_rope + 9)], arrow=False)
        f.rope(xx, y_rope)
    f.text(x_q - 16, y_rope + 5, "rotary", 13.5, True, GREY, anchor="end")
    # this layer's cache: heads across, one row per character read, the newest at the bottom
    for xx, lab in ((x_k, "keys"), (x_v, "values")):
        gx0 = xx - heads * cell_w / 2
        for r_ in range(nrow):
            if r_ == nrow - 3:
                f.text(xx, y_cache + r_ * cell_h + 8, "⋮", 14, False, GREY)
                continue
            for j in range(heads):
                f.add(f'<rect x="{gx0 + j * cell_w:.1f}" y="{y_cache + r_ * cell_h:.1f}" '
                      f'width="{cell_w - 1}" height="{cell_h - 1}" fill="{HEAD[j % len(HEAD)]}" '
                      f'fill-opacity="{1.0 if r_ == nrow - 1 else 0.42}"/>')
        f.add(f'<rect x="{gx0 - 1.5:.1f}" y="{y_cache + (nrow - 1) * cell_h - 1.5:.1f}" '
              f'width="{heads * cell_w + 2}" height="{cell_h + 2}" fill="none" stroke="#333333" '
              f'stroke-width="1.4"/>')
        f.line([(xx, y_rope - 9 if lab == "keys" else y_bars), (xx, y_cache + cache_h + 2)])
        f.text(gx0 - 6, y_cache + cache_h / 2 + 5, lab, 13.5, True, GREY, anchor="end")
    nx = x_v + heads * cell_w / 2 + 14
    yy = y_cache + 8
    for para in (f"This layer's cache: the key and value of every character so far, up to "
                 f"{context:,}. The newest row is this character's own.",
                 f"{n_cache} caches in all: one per prelude layer, one per row."):
        cur = ""
        lines = []
        for w_ in para.split(" "):
            t_ = (cur + " " + w_).strip()
            if measure(t_, 13) <= FX1 - 14 - nx or not cur:
                cur = t_
            else:
                lines.append(cur)
                cur = w_
        for ln in lines + [cur]:
            f.text(nx, yy, ln, 13, True, GREY, anchor="start")
            yy += 15
        yy += 7
    assert yy < y_bars - 4, (yy, y_bars)
    # the heads, stacked as in the paper
    fx0, fw = ICX - 240, 470
    for j in reversed(range(heads)):
        f.add(f'<rect x="{fx0 + STK * j:.1f}" y="{y_front - STK * j:.1f}" width="{fw}" height="{front_h}" '
              f'rx="7" fill="{FILL["attn"]}" stroke="#333333" stroke-width="1.1"/>')
        f.add(f'<rect x="{fx0 + STK * j:.1f}" y="{y_front - STK * j:.1f}" width="8" height="{front_h}" '
              f'rx="3" fill="{HEAD[j % len(HEAD)]}"/>')
    f.text(ICX - 5, y_front + 30, "Scaled dot-product attention", 19)
    f.text(ICX - 5, y_front + 54, f"softmax( q · kᵀ / √{hd}, masked ) · v", 16, True)
    f.text(ICX - 5, y_front + 76, f"{heads} heads side by side, each on its own {hd} of the {d} numbers",
           13.5, True, GREY)
    f.line([(x_q, y_rope - 9), (x_q, y_front + front_h)])
    f.text(x_q - 8, y_front + front_h + 24, "q", 17, True, anchor="end")
    for xx, lab in ((x_k, "K"), (x_v, "V")):
        f.line([(xx, y_cache), (xx, y_front + front_h)])
        f.text(xx - 8, y_front + front_h + 24, lab, 17, True, anchor="end")
    # joined and projected back into the residual stream
    f.line([(ICX, y_stack_top), (ICX, y_cat + 46)])
    f.box(ICX, y_cat, 250, 46, FILL["cat"], ["Concat"], f"{heads} × {hd} → {d}")
    f.line([(ICX, y_cat), (ICX, y_proj + 46)])
    f.box(ICX, y_proj, 250, 46, FILL["lin"], ["Linear"], f"{d} → {d}")
    f.line([(ICX, y_proj), (ICX, y_out + 8)])
    f.text(ICX, y_out, "added to the character's state: the ⊕ above the attention", 15, True)

    return f, (0, top, 1900, H - top)


def write(path, elements, box, title):
    x, y, w, h = box
    svg = (f'<?xml version="1.0" encoding="UTF-8"?>\n'
           f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0f}" height="{h:.0f}" '
           f'viewBox="{x:.0f} {y:.0f} {w:.0f} {h:.0f}">\n<title>{title}</title>\n'
           f'<rect x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{h:.0f}" fill="white"/>\n'
           + "\n".join(elements) + "\n</svg>\n")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        fh.write(svg)
    os.replace(path + ".tmp", path)
    print(f"  {os.path.relpath(path, ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights"))
    ap.add_argument("--out", default=os.path.join(ROOT, "assets", "layers.svg"))
    a = ap.parse_args()
    with open(os.path.join(a.weights, "manifest.json")) as fh:
        man = json.load(fh)
    f, box = draw(man)
    write(a.out, f.out, box, "mini-AGI: the layers")


if __name__ == "__main__":
    main()
