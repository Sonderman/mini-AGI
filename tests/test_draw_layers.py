#!/usr/bin/env python3
"""
The architecture figure is drawn from the weights, and the README shows it.

  tools/draw_layers.py reads a weights directory's manifest and writes one
  well-formed SVG: the layer stack, and beside it the inside of one
  attention layer

  the numbers in it are the manifest's - the pool's size above all, which
  moves as it grows - not numbers typed into the drawing

  a model the drawing cannot be of (a coda, more than one recurrent block)
  is refused rather than drawn wrong

  the README points at the file where it is written, and release.sh redraws
  it before a push
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.dom.minidom

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


def texts(path):
    dom = xml.dom.minidom.parse(path)
    return " ".join(n.firstChild.data for n in dom.getElementsByTagName("text") if n.firstChild)


CFG = dict(vocab_size=265, d_model=512, n_head=8, d_ff=1408, block=4096, n_prelude=2,
           n_recur=1, n_coda=0, max_steps=24, pool_top_k=8, pool_d_ff=2048, halt_thresh=0.9,
           tie_embeddings=True)

tmp = tempfile.mkdtemp(prefix="minagi-layers-")
try:
    wdir, out = os.path.join(tmp, "weights"), os.path.join(tmp, "assets", "layers.svg")
    os.makedirs(wdir)
    with open(os.path.join(wdir, "manifest.json"), "w") as f:
        json.dump({"cfg": CFG, "n_experts": 123, "context_now": 4096}, f)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools/draw_layers.py"),
                        "--weights", wdir, "--out", out], capture_output=True, text=True)
    check("draw_layers runs on a weights directory", r.returncode == 0, r.stderr[-300:])

    try:
        t_fig = texts(out)
        check("the figure is well-formed SVG", True)
    except Exception as e:                                    # noqa: BLE001
        t_fig = ""
        check("the figure is well-formed SVG", False, str(e))

    check("the pool's size is the manifest's", "top 8 of 123 experts" in t_fig)
    check("every layer of the stack is named",
          all(s in t_fig for s in ("Input Embedding", "Masked multihead", "Feed Forward", "Concat",
                                      "Router", "Mixture of Experts", "Softmax", "Sigmoid",
                                      "up to 24×", "2×")))
    check("beside it, the attention layer's heads and cache",
          all(s in t_fig for s in ("8 heads", "64 numbers", "Scaled dot-product attention",
                                   "26 caches in all", "up to 4,096")))

    with open(os.path.join(wdir, "manifest.json"), "w") as f:
        json.dump({"cfg": dict(CFG, n_coda=1), "n_experts": 123}, f)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools/draw_layers.py"),
                        "--weights", wdir, "--out", os.path.join(tmp, "bad", "layers.svg")],
                       capture_output=True, text=True)
    check("a model with a coda is refused, not drawn wrong",
          r.returncode != 0 and not os.path.exists(os.path.join(tmp, "bad", "layers.svg")))

    readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    check("the README shows the figure from assets/, once",
          readme.count("](assets/layers.svg)") == 1 and "attention.svg" not in readme)
    release = open(os.path.join(ROOT, "release.sh"), encoding="utf-8").read()
    check("release.sh redraws it before a push",
          "python3 tools/draw_layers.py" in release)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n  all architecture-figure tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
