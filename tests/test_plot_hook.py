"""
The graphs are redrawn whenever the sample log gains an entry.

A graph that has to be remembered is a graph that is out of date when it is
looked at, so this is not left to be run by hand. The properties that matter
are that it never blocks a training step and never raises into one: a run that
has been going for days must not be stopped by matplotlib.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train as T  # noqa: E402

ok = True


def check(name, cond, detail=""):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    ok = ok and cond


# A log of its own. This pointed at runs/samples.txt, which meant the test
# passed only while one particular run's log happened to be sitting there -
# and failed the moment it was cleared for a fresh run. A test that depends on
# the state of the machine it runs on is not testing the hook.
import tempfile                                              # noqa: E402

_tmp = tempfile.mkdtemp()
LOG = os.path.join(_tmp, "samples.txt")
with open(LOG, "w") as _f:
    for i in range(30):
        c = 10.0 + i
        _f.write(
            f"step {1000 * (i + 1):,}   {c:.1f}M of 7,901M characters (0.13%)"
            f"   {10 * (i + 1)} min   64 experts\n"
            f"context 4,096 characters of 4,096   reading 900 char/s"
            f"   still gaining +0.0100 deep into it\n"
            f"grad norm 0.90 against a clip of 1   under the clip\n"
            f"train loss {1.5 - i * 0.01:.4f}   lr 3.00e-05"
            f"   evidence t +1.00 over 60.0 (effect +0.0100)   rate x0.500\n"
            f"held-out loss {1.6 - i * 0.01:.4f} +/-0.0500 nats"
            f"   {2.3 - i * 0.01:.4f} bits/char   perplexity 4.95   gap +0.1000\n"
            f"  arithmetic 0.700   chat 1.100   chat_hermes 1.300   chess 0.600"
            f"   code 1.000   reasoning 1.000   stories 0.700   wikipedia 1.500\n"
            f"repeats 5% of 8-grams, greedy with no guard\n\n")

saved = dict(T._PLOTS)
try:
    # --- off means off ---------------------------------------------------
    T._PLOTS.update(on=False, proc=None, since=None, log=None,
                    weights=None)
    T.refresh_plots()
    check("disabled draws nothing", T._PLOTS["proc"] is None)

    # --- it returns without waiting --------------------------------------
    # the graphs go to a folder of the test's own: the plotter would
    # otherwise draw this fake run over the real run's runs/ graphs, and
    # the README copies runs/training_progress.png
    T._PLOTS.update(on=True, proc=None, since=None, log=LOG,
                    weights=None, out=_tmp)
    t0 = time.time()
    T.refresh_plots()
    el = time.time() - t0
    p = T._PLOTS["proc"]
    check("returns immediately", el < 1.0, f"{el*1000:.0f} ms")
    check("something was spawned", p is not None)
    if p is not None:
        p.wait()
        check("and it succeeds, drawing where it was told and only there",
              p.returncode == 0
              and os.path.exists(os.path.join(_tmp, "training_progress.png")),
              f"exit {p.returncode}")

    # --- a drawing still running is not queued behind another ------------
    class Busy:
        returncode = None

        def poll(self):
            return None                      # still going

    T._PLOTS.update(proc=Busy())
    T.refresh_plots()
    check("a second call while busy is skipped",
          isinstance(T._PLOTS["proc"], Busy))

    # --- a missing tool disables it rather than raising every time -------
    # pointed at a folder with no plotter in it: renaming the real one, as
    # this did, could switch a running run's own graphs off for good
    T._PLOTS.update(on=True, proc=None, tools=os.path.join(_tmp, "no-tools"))
    T.refresh_plots()
    check("a missing plotter turns the hook off, quietly",
          T._PLOTS["on"] is False and T._PLOTS["proc"] is None)
    T._PLOTS.update(on=True, tools=None)

    # --- write_samples calls it ------------------------------------------
    import inspect
    src = inspect.getsource(T.write_samples)
    body = src[:src.rindex("return path")]
    check("write_samples calls it before returning",
          "refresh_plots()" in body)
finally:
    T._PLOTS.clear()
    T._PLOTS.update(saved)

print("\n  " + ("all plot-hook tests pass" if ok else "FAILURES ABOVE"))
sys.exit(0 if ok else 1)
