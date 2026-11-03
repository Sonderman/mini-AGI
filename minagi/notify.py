"""
Tell someone when the model reaches a new best. Optional, and removable.

REMOVING THIS. Delete this file. That is the whole procedure: every call site
imports it inside a try/except and carries on if the import fails, so the run
does not notice it is gone. The hook in train.py is five lines marked with the
same word as this docstring, and can go with it.

CREDENTIALS ARE NEVER IN THE REPOSITORY. It reads two settings and does nothing
at all unless BOTH are present, from either the environment or a .env file in
the project root:

    MINAGI_TELEGRAM_TOKEN=123456:ABC...      # from @BotFather
    MINAGI_TELEGRAM_CHAT=987654321           # python3 -m minagi.notify

.env is gitignored. Python does not read it on its own - nothing does unless a
program goes looking - which is why setting a token there and expecting it to
work quietly does nothing at all. This module reads it.

With either unset this module is inert - no network call, no error, no log
line. That is deliberate: notification is opt-in, and a run on a machine that
has never been configured should not notice that this exists.

WHAT LEAVES THE MACHINE. One line of numbers per new best: the loss, the
characters read, the expert count. No text the model generated, no corpus, no
paths. Telegram is a third party and a message sent there is out of your hands,
so the content is kept to what you would put in a commit message.

Free: the Bot API has no cost and no account beyond Telegram itself.

Nothing here can stop a run. Every failure - no network, bad token, rate limit,
Telegram down - is swallowed and counted. A notifier that can crash training is
worse than no notifier.
"""

import json
import os
import urllib.error
import urllib.parse
import urllib.request

_STATE = {"sent": 0, "failed": 0, "last_error": None}
_ENV_CACHE = None
# the .env this reads, in the project root
ENV_FILE = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), ".env")


def _dotenv():
    """
    KEY=VALUE lines from .env in the project root, read once.

    The environment still wins, so an exported value overrides the file. Quotes
    are stripped because people write them and a token with quotes around it
    fails with a 404 that says nothing about quotes.
    """
    global _ENV_CACHE
    if _ENV_CACHE is not None:
        return _ENV_CACHE
    _ENV_CACHE = {}
    try:
        with open(ENV_FILE) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                _ENV_CACHE[k.strip()] = v.strip().strip("'\"")
    except OSError:
        pass
    return _ENV_CACHE


def get(name):
    """Environment first, then .env."""
    return os.environ.get(name) or _dotenv().get(name) or ""


def configured():
    """Both settings present, or this module does nothing."""
    return bool(get("MINAGI_TELEGRAM_TOKEN") and get("MINAGI_TELEGRAM_CHAT"))


def send(text, timeout=5.0):
    """
    Post one message. Returns True if it went, False for every other outcome.

    Never raises. The timeout is short on purpose: this is called from the
    training loop, and a hung socket would stall reading.
    """
    if not configured():
        return False
    url = f"https://api.telegram.org/bot{get('MINAGI_TELEGRAM_TOKEN')}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": get("MINAGI_TELEGRAM_CHAT"),
        "text": text,
        "disable_notification": "false",
    }).encode()
    try:
        with urllib.request.urlopen(url, data=data, timeout=timeout) as r:
            ok = json.loads(r.read().decode()).get("ok", False)
        _STATE["sent" if ok else "failed"] += 1
        if not ok:
            _STATE["last_error"] = "telegram returned ok=false"
        return bool(ok)
    except (urllib.error.URLError, OSError, ValueError, TimeoutError) as e:
        _STATE["failed"] += 1
        _STATE["last_error"] = f"{type(e).__name__}: {e}"
        return False


def best(val, chars, experts=None, prev=None, extra=""):
    """A new best held-out. Numbers only."""
    import math
    bits = val / math.log(2)
    lines = [f"new best held-out {val:.4f} nats ({bits:.3f} bits/char)"]
    if prev is not None and prev != float("inf"):
        lines.append(f"was {prev:.4f}  ({val - prev:+.4f})")
    lines.append(f"{chars / 1e6:.1f}M characters read"
                 + (f", {experts} experts" if experts is not None else ""))
    if extra:
        lines.append(extra)
    return send("\n".join(lines))


def status():
    return dict(_STATE, configured=configured())


def _api(method, **params):
    """One Bot API call. Returns (ok, payload-or-error-string)."""
    tok = get("MINAGI_TELEGRAM_TOKEN")
    if not tok:
        return False, "MINAGI_TELEGRAM_TOKEN is not set"
    url = f"https://api.telegram.org/bot{tok}/{method}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            d = json.loads(r.read().decode())
        return bool(d.get("ok")), d.get("result", d.get("description", ""))
    except urllib.error.HTTPError as e:
        try:
            return False, json.loads(e.read().decode()).get("description", str(e))
        except Exception:
            return False, str(e)
    except (urllib.error.URLError, OSError, ValueError, TimeoutError) as e:
        return False, f"{type(e).__name__}: {e}"


def setup():
    """
    Find the chat id and check the token, with the failure named.

        python3 -m minagi.notify

    `getUpdates` returns an EMPTY list until you have messaged the bot, which
    is the step people skip - so the failure is reported as "message your bot
    first", not as an empty parse.
    """
    ok, who = _api("getMe")
    if not ok:
        print(f"  token rejected: {who}")
        print("  check MINAGI_TELEGRAM_TOKEN in .env - no quotes, no spaces,")
        print("  and the whole thing including the digits before the colon")
        return 1
    print(f"  token works: @{who.get('username')} ({who.get('first_name')})")

    chat = get("MINAGI_TELEGRAM_CHAT")
    if not chat:
        ok, ups = _api("getUpdates", limit=10)
        if not ok:
            print(f"  getUpdates failed: {ups}")
            return 1
        ids = []
        for u in ups or []:
            for key in ("message", "edited_message", "channel_post"):
                c = (u.get(key) or {}).get("chat")
                if c and c.get("id") not in [i for i, _ in ids]:
                    ids.append((c["id"], c.get("username") or c.get("title")
                                or c.get("first_name") or ""))
        if not ids:
            print(f"  no messages yet. Open Telegram, find @{who.get('username')},")
            print("  send it any message, then run this again.")
            print("  (getUpdates is empty until the bot has heard from you -")
            print("   this is the step that looks like the token being wrong)")
            return 1
        print("  chat id(s) that have messaged this bot:")
        for i, name in ids:
            print(f"    {i}   {name}")
        print(f"\n  add to .env:\n    MINAGI_TELEGRAM_CHAT={ids[0][0]}")
        return 0

    print(f"  chat id set: {chat}")
    if send("mini-AGI notifier is configured."):
        print("  sent a test message - check Telegram")
        return 0
    print(f"  send failed: {status()['last_error']}")
    return 1


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(setup())
