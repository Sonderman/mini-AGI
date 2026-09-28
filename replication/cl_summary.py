#!/usr/bin/env python3
"""
Every number the continual-learning write-up quotes, pulled from the runs.

Nothing in the document is typed by hand: a figure that is transcribed is a
figure that drifts from the run it came from. This prints the table the
markdown quotes, so regenerating the document after a re-run is one command.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def R(name):
    return os.path.join(HERE, "results", name)


ARMS = [
    (R("static.json"), "experts frozen, trunk = expert LR"),
    (R("none.json"), "swapping, trunk = expert LR"),
    (R("trunk01.json"), "swapping, trunk at 0.1x  (the run)"),
]
CONTROL = R("control.json")
RAND = float(np.log(265))


def summarise(path):
    j = json.load(open(path))
    doms, base, rows = j["domains"], j["baseline"], j["rows"]
    lanes = j.get("lanes", [j["domain"]])
    read = [d for d in doms if d in lanes]
    other = [d for d in doms if d not in lanes]
    last = rows[-1]
    return {
        "tag": j.get("tag"), "path": path, "steps": last["steps"],
        "n_read": len(read),
        "chars": last["chars"], "lr": j.get("lr"), "warmup": j.get("warmup", 0),
        "trunk": j.get("trunk_lr_mult"), "optim": j.get("optim"),
        "no_swap": j.get("no_swap", False), "rotating": bool(j.get("rotating")),
        "touched": last.get("experts_touched"), "total": j.get("experts_total"),
        "read_delta": float(np.mean([last[d] - base[d] for d in read])) if read else None,
        "other_delta": float(np.mean([last[d] - base[d] for d in other])) if other else None,
        "all_delta": float(np.mean([last[d] - base[d] for d in doms])),
        "retained": float(np.mean([(RAND - last[d]) / (RAND - base[d])
                                   for d in (other or doms)])),
        "per_domain": {d: (base[d], last[d], last[d] - base[d]) for d in doms},
        "curve": [(r["chars"], float(np.mean([r[d] - base[d]
                                              for d in (other or doms)])))
                  for r in rows],
    }


def main():
    print(f"  chance (uniform over 265 byte values) = {RAND:.4f} nats\n")
    ctrl = None
    if os.path.exists(CONTROL):
        ctrl = summarise(CONTROL)
        print("CONTROL - the arm that certifies the settings")
        print(f"  lr {ctrl['lr']:g}  warmup {ctrl['warmup']}  "
              f"trunk {ctrl['trunk']}x  {ctrl['optim']}")
        print(f"  {ctrl['steps']} steps, {ctrl['chars']/1000:.0f}K characters")
        # EVERY eval domain has a training lane behind it - chat_hermes looks
        # held out but train/chat is 77% hermes files - so the control reads
        # all eight and nothing in it can be forgetting. Any drift it shows is
        # the instrument.
        print(f"  drift, the {ctrl['n_read']} rotated lanes  : "
              f"{ctrl['read_delta']:+.4f} nats")
        oth = ctrl["other_delta"]
        print(f"  drift, chat_hermes          : "
              + (f"{oth:+.4f} nats (a split of the chat lane, not held out)"
                 if oth is not None else "n/a"))
        # Only DEGRADATION is contamination. The test used to be on the
        # absolute value, which called a control that improved the lanes it
        # reads - the ideal outcome, and what interleaving actually does -
        # contaminated.
        verdict = ("CLEAN - read domains hold or improve, so the massed arms "
                   "measure forgetting rather than damage"
                   if ctrl["read_delta"] < 0.02 else
                   "CONTAMINATED - the probe degrades domains it is READING; "
                   "treat every massed arm as an upper bound, not a measurement")
        print(f"  verdict: {verdict}\n")
    print(f"  {'arm':<34} {'read':>8} {'unread':>8} {'retained':>9} {'experts':>9}")
    for path, name in ARMS:
        if not os.path.exists(path):
            print(f"  {name:<34} {'(not run)':>8}")
            continue
        s = summarise(path)
        print(f"  {name:<34} {s['read_delta']:>+8.4f} {s['other_delta']:>+8.4f} "
              f"{s['retained']*100:>8.2f}% {s['touched']:>4}/{s['total']:<4}")
    if ctrl:
        print(f"\n  floor from the control: {ctrl['all_delta']:+.4f} nats "
              f"(subtract only if the control is NOT flat, and say so)")
    print("\n  per-domain, each arm:")
    for path, name in ARMS:
        if not os.path.exists(path):
            continue
        s = summarise(path)
        print(f"\n   {name}")
        for d, (b, l, dd) in sorted(s["per_domain"].items(),
                                    key=lambda kv: -kv[1][2]):
            print(f"     {d:<14} {b:.4f} -> {l:.4f}  {dd:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
