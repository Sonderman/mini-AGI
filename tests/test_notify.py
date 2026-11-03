#!/usr/bin/env python3
"""
The notifier must be invisible unless configured, and removable without trace.

Three properties, and the first two matter more than the feature:

  INERT     with no credentials it makes no network call and raises nothing:
            no .env, a .env without its two settings, or only one of them all
            leave it off. A run on a machine that has never been configured
            behaves exactly as it would without the file.
  SILENT    it cannot stop a run. Every failure - no network, bad token, rate
            limit - is swallowed. A notifier that can crash training is worse
            than no notifier.
  REMOVABLE deleting minagi/notify.py leaves a working train.py. The call site
            imports inside a try/except for exactly this reason.

Nothing here reads the machine's own .env: the module is pointed at files this
test writes, so it passes the same with or without a configured notifier.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


for k in ("MINAGI_TELEGRAM_TOKEN", "MINAGI_TELEGRAM_CHAT"):
    os.environ.pop(k, None)
from minagi import notify                                    # noqa: E402

tmp = tempfile.mkdtemp()


def env_file(text=None):
    """Point the module at a .env of this test's own, or at none at all."""
    path = os.path.join(tmp, ".env")
    if os.path.exists(path):
        os.remove(path)
    if text is not None:
        with open(path, "w") as f:
            f.write(text)
    notify.ENV_FILE = path
    notify._ENV_CACHE = None


# ------------------------------------------- 1. inert without credentials
env_file(None)
check("not configured with no .env and the variables unset",
      not notify.configured())
check("send() is a no-op and returns False", notify.send("x") is False)
check("best() is a no-op and returns False",
      notify.best(1.1, 286e6, 395, prev=1.2) is False)
check("nothing was counted as sent or failed",
      notify.status()["sent"] == 0 and notify.status()["failed"] == 0,
      "an unconfigured notifier does not even try")
env_file("# nothing for the notifier here\nOTHER_SETTING=1\n")
check("a .env without its two settings leaves it off", not notify.configured())

# ---------------------------- 2. half-configured is still not configured
env_file("MINAGI_TELEGRAM_TOKEN=not-a-real-token\n")
check("a token alone in .env is not enough", not notify.configured())
env_file(None)
os.environ["MINAGI_TELEGRAM_TOKEN"] = "not-a-real-token"
check("a token alone in the environment is not enough", not notify.configured())
os.environ["MINAGI_TELEGRAM_CHAT"] = "0"
check("both together are configured", notify.configured())

# ------------------------------- 3. a bad token fails quietly, not loudly
sent = notify.send("this will not reach telegram", timeout=3.0)
check("a bad token returns False rather than raising", sent is False)
check("...and the failure is recorded", notify.status()["failed"] >= 1,
      str(notify.status()["last_error"])[:52])
for k in ("MINAGI_TELEGRAM_TOKEN", "MINAGI_TELEGRAM_CHAT"):
    os.environ.pop(k, None)

# ---------------------------------- 4. the call site survives its absence
src = open(os.path.join(ROOT, "train.py")).read()
check("train.py imports it lazily, inside the loop",
      "from minagi import notify" in src
      and src.count("from minagi import notify") == 1)
i = src.index("from minagi import notify")
window = src[i - 400:i + 400]
check("...guarded by try/except so deleting the file is enough",
      "try:" in window and "except Exception:" in window)
check("nothing else in the project imports it",
      sum("import notify" in open(os.path.join(ROOT, p)).read()
          for p in ("train.py", "serve.py")) == 1,
      "one call site, one file to delete")

# ------------------------------- 5. .env is read, and gitignored
check(".env is gitignored",
      ".env" in open(os.path.join(ROOT, ".gitignore")).read(),
      "a token in a committed file is a token published")
env_file("MINAGI_TELEGRAM_TOKEN='t'\nMINAGI_TELEGRAM_CHAT=\"c\"\n")
check("...and .env alone is enough to be configured, quotes stripped",
      notify.configured() and notify.get("MINAGI_TELEGRAM_TOKEN") == "t")
os.environ["MINAGI_TELEGRAM_TOKEN"] = "from-env"
check("the environment wins over .env",
      notify.get("MINAGI_TELEGRAM_TOKEN") == "from-env")
os.environ.pop("MINAGI_TELEGRAM_TOKEN", None)
env_file(None)

# ------------------------------------ 6. no credential is in the repository
for path in ("train.py", "minagi/notify.py", "config.yaml"):
    body = open(os.path.join(ROOT, path)).read()
    leaked = [w for w in body.split()
              if len(w) > 25 and ":" in w and w.count(":") == 1
              and w.split(":")[0].isdigit()]
    check(f"no bot token literal in {path}", not leaked, str(leaked)[:40])

print("\n  all notify tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
