#!/usr/bin/env python3
"""
Tool use that is run as it is written: the `tools` lane.

Every call in this lane is executed against a small world made for its
episode - a calculator, a folder tree, a memory, pages from the wikipedia
lane, a calendar, unit tables - and the result written after the call is
what the tool returned. The reply is then composed from that result. So
`replay` can run an episode's calls again on a fresh copy of its world and
get the same results back, and the reply quotes them.

The same tasks make a benchmark. `episode` returns the world the calls ran
against and the answers expected, with the transcript: a harness can hand
the model the tools and the request, run whatever it calls against the
world, and compare what it says.

Beyond the calls themselves, five habits:
  - a tool for what is hard - a seven-digit product, a weekday in 1987, a
    unit factor - and none for what is not: 3 + 4 is answered directly
  - look before answering: a config is listed, read, then quoted
  - recover: a wrong path comes back as an error, and the next call looks
  - say so: a search with no hits, a recall with nothing saved, a request
    none of the tools can serve - each ends in plain words, not a guess
  - ask: a request missing what a tool needs gets a question back

The files, logs, tables and notes are generated; the encyclopedia is pages
of the wikipedia lane. A fact in a reply is therefore either on a page the
model also reads or in a file the episode made up and showed it.

    python3 -m corpora agent --only toolbox
"""

import ast
import datetime as dt
import html
import math
import os
import random
import re
import zlib
from collections import Counter
from decimal import Decimal, localcontext
from fractions import Fraction


# --------------------------------------------------------------- the tools
def _spec(name, description, **params):
    """A tool as <tools> lists it. A parameter is (type, description), or
    (type, description, default) - a default makes it optional."""
    props, required = {}, []
    for k, p in params.items():
        props[k] = {"type": p[0], "description": p[1]}
        if len(p) > 2:
            props[k]["default"] = p[2]
        else:
            required.append(k)
    out = {"name": name, "description": description,
           "parameters": {"type": "object", "properties": props}}
    if required:
        out["parameters"]["required"] = required
    return out


SPECS = {s["name"]: s for s in [
    _spec("calculator",
          "Evaluate an arithmetic expression exactly: + - * / // % **, "
          "parentheses, and abs, round, min, max, gcd, lcm, sqrt.",
          expression=("string", "The expression, e.g. (1250 - 175) * 0.2")),
    _spec("list_files",
          "List a folder: each file with its size in bytes, each folder "
          "ending in /.",
          path=("string", "An absolute path, e.g. /etc")),
    _spec("read_file", "Read a text file.",
          path=("string", "An absolute path")),
    _spec("search_files",
          "Find the lines that contain some text, in a file or every file "
          "under a folder. Case-insensitive.",
          text=("string", "What to look for"),
          path=("string", "Where to look", "/")),
    _spec("write_file", "Write a text file, replacing whatever was there.",
          path=("string", "An absolute path"),
          content=("string", "The whole new content")),
    _spec("remember", "Save a note that later conversations can recall.",
          key=("string", "A short name for it"),
          value=("string", "What to remember")),
    _spec("recall", "Look up saved notes by name.",
          key=("string", "The name, or part of it")),
    _spec("search",
          "Search the encyclopedia. Returns the best-matching page titles, "
          "each with its first line.",
          query=("string", "Words to search for")),
    _spec("open_page", "Read the start of an encyclopedia page.",
          title=("string", "The exact title, as search returns it")),
    _spec("date_info",
          "The weekday, day of the year and ISO week of a date.",
          date=("string", "YYYY-MM-DD")),
    _spec("days_between",
          "Days from one date to another; negative when the second is "
          "earlier.",
          start=("string", "YYYY-MM-DD"), end=("string", "YYYY-MM-DD")),
    _spec("add_days",
          "The date a number of days after another, or before it when the "
          "number is negative.",
          date=("string", "YYYY-MM-DD"), days=("integer", "How many days")),
    _spec("convert_units",
          "Convert an amount between units of length, mass, volume, speed "
          "or temperature.",
          value=("number", "The amount"),
          from_unit=("string", "e.g. mi, km, lb, kg, F, C"),
          to_unit=("string", "e.g. km")),
]}

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
            "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


class ToolError(Exception):
    """A tool's refusal; the model reads it as the call's result."""


# ------------------------------------------------------------ calculator
_LIMIT = 10 ** 60


def _checked(v):
    if abs(v.numerator) > _LIMIT or v.denominator > _LIMIT:
        raise ToolError("the result is too large")
    return v


def _half_up(v, places=0):
    """v rounded to `places` decimals, halves away from zero - the rounding
    people expect of money, not Python's halves-to-even."""
    q = Fraction(10) ** places
    r = math.floor(abs(v) * q + Fraction(1, 2))
    return (r if v >= 0 else -r) / q


def _ev(n):
    if isinstance(n, ast.Constant) and type(n.value) in (int, float):
        return Fraction(str(n.value)) if isinstance(n.value, float) \
            else Fraction(n.value)
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
        v = _ev(n.operand)
        return v if isinstance(n.op, ast.UAdd) else -v
    if isinstance(n, ast.BinOp):
        a, b = _ev(n.left), _ev(n.right)
        op = type(n.op)
        if op is ast.Add:
            return _checked(a + b)
        if op is ast.Sub:
            return _checked(a - b)
        if op is ast.Mult:
            return _checked(a * b)
        if op in (ast.Div, ast.FloorDiv, ast.Mod):
            if b == 0:
                raise ToolError("division by zero")
            if op is ast.Div:
                return _checked(a / b)
            if op is ast.FloorDiv:
                return Fraction(math.floor(a / b))
            return _checked(a - b * math.floor(a / b))
        if op is ast.Pow:
            if b.denominator != 1 or abs(b) > 64:
                raise ToolError("a power must be a whole number up to 64")
            if a == 0 and b < 0:
                raise ToolError("division by zero")
            if abs(a.numerator) > 1 and abs(b) * math.log10(
                    max(abs(a.numerator), a.denominator)) > 60:
                raise ToolError("the result is too large")
            return _checked(a ** int(b))
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) \
            and not n.keywords:
        f, args = n.func.id, [_ev(x) for x in n.args]
        if f == "abs" and len(args) == 1:
            return abs(args[0])
        if f == "round" and len(args) in (1, 2):
            if len(args) == 2 and (args[1].denominator != 1
                                   or not 0 <= args[1] <= 12):
                raise ToolError("round takes 0 to 12 decimal places")
            return _half_up(args[0], int(args[1]) if len(args) == 2 else 0)
        if f in ("min", "max") and args:
            return min(args) if f == "min" else max(args)
        if f in ("gcd", "lcm") and len(args) >= 2:
            if any(x.denominator != 1 for x in args):
                raise ToolError(f"{f} takes whole numbers")
            r = int(args[0])
            for x in map(int, args[1:]):
                r = math.gcd(r, x) if f == "gcd" else \
                    (abs(r * x) // math.gcd(r, x) if r and x else 0)
            return _checked(Fraction(r))
        if f == "sqrt" and len(args) == 1:
            if args[0] < 0:
                raise ToolError("no square root of a negative number")
            with localcontext() as c:
                c.prec = 30
                r = (Decimal(args[0].numerator)
                     / Decimal(args[0].denominator)).sqrt()
            return Fraction(r)
        raise ToolError(f"{f} is not one of the calculator's functions")
    raise ToolError("only numbers, + - * / // % ** and the listed functions "
                    "are allowed")


def _decimal(v, places):
    """v written with at most `places` decimals, rounded half up, trailing
    zeros dropped."""
    r = _half_up(v, places)
    q = 10 ** places
    whole, frac = divmod(abs(r.numerator * (q // r.denominator)), q)
    s = f"{whole}.{frac:0{places}d}".rstrip("0").rstrip(".") if places \
        else str(whole)
    return ("-" + s) if r < 0 else s


def number(v):
    """A result as the calculator writes it: whole numbers in full, a
    decimal that ends written to its end (up to 12 places), any other
    rounded to 10."""
    if v.denominator == 1:
        return str(v.numerator)
    d = v.denominator
    for p in (2, 5):
        while d % p == 0:
            d //= p
    return _decimal(v, 12 if d == 1 else 10)


def calc(expression):
    try:
        tree = ast.parse(str(expression).strip(), mode="eval")
    except SyntaxError:
        raise ToolError(f"cannot read {expression!r} as an expression")
    return number(_ev(tree.body))


def significant(v, n=6):
    """v to n significant figures, without an exponent."""
    if v == 0:
        return "0"
    e = math.floor(math.log10(abs(float(v))))
    if e >= n:
        q = 10 ** (e - n + 1)
        return str(int(_half_up(v / q)) * q)
    return _decimal(v, n - 1 - e)


# ----------------------------------------------------------------- units
_LENGTH = {"m": "1", "km": "1000", "cm": "0.01", "mm": "0.001",
           "mi": "1609.344", "yd": "0.9144", "ft": "0.3048", "in": "0.0254"}
_MASS = {"kg": "1", "g": "0.001", "lb": "0.45359237", "oz": "0.028349523125"}
_VOLUME = {"l": "1", "ml": "0.001", "gal": "3.785411784",
           "qt": "0.946352946", "cup": "0.2365882365"}
_SPEED = {"m/s": "1", "km/h": "5/18", "mph": "0.44704", "kn": "463/900"}
UNITS = {}
for _dim, _table in (("length", _LENGTH), ("mass", _MASS),
                     ("volume", _VOLUME), ("speed", _SPEED)):
    for _u, _f in _table.items():
        UNITS[_u] = (_dim, Fraction(_f))
for _u in ("C", "F", "K"):
    UNITS[_u] = ("temperature", None)
_ALIAS = {"meter": "m", "meters": "m", "metre": "m", "metres": "m",
          "kilometer": "km", "kilometers": "km", "kilometre": "km",
          "kilometres": "km", "centimeters": "cm", "millimeters": "mm",
          "mile": "mi", "miles": "mi", "yard": "yd", "yards": "yd",
          "foot": "ft", "feet": "ft", "inch": "in", "inches": "in",
          "kilogram": "kg", "kilograms": "kg", "gram": "g", "grams": "g",
          "pound": "lb", "pounds": "lb", "lbs": "lb", "ounce": "oz",
          "ounces": "oz", "liter": "l", "liters": "l", "litre": "l",
          "litres": "l", "milliliters": "ml", "gallon": "gal",
          "gallons": "gal", "quart": "qt", "quarts": "qt", "cups": "cup",
          "kph": "km/h", "knot": "kn", "knots": "kn", "celsius": "C",
          "fahrenheit": "F", "kelvin": "K", "°c": "C", "°f": "F"}


def _unit(u):
    s = str(u).strip()
    if s in UNITS:
        return s
    low = s.lower()
    if low in UNITS:
        return low
    if low in _ALIAS:
        return _ALIAS[low]
    if s.upper() in ("C", "F", "K"):
        return s.upper()
    raise ToolError(f"unknown unit {s!r}")


def _to_kelvin(v, u):
    if u == "K":
        return v
    if u == "C":
        return v + Fraction("273.15")
    return (v - 32) * Fraction(5, 9) + Fraction("273.15")


def _from_kelvin(v, u):
    if u == "K":
        return v
    if u == "C":
        return v - Fraction("273.15")
    return (v - Fraction("273.15")) * Fraction(9, 5) + 32


def _amount(x):
    if isinstance(x, bool):
        raise ToolError("value must be a number")
    if isinstance(x, (int, float)):
        return Fraction(str(x)) if isinstance(x, float) else Fraction(x)
    try:
        return Fraction(str(x).replace(",", "").strip())
    except (ValueError, ZeroDivisionError):
        raise ToolError(f"{x!r} is not a number")


# ---------------------------------------------------------- encyclopedia
def _words(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def wiki_text(raw):
    """A page of the wikipedia lane as plain paragraphs: the wikitext's
    templates, tables, references, file links and markup taken out. None for
    a redirect."""
    m = re.search(r"<text[^>]*>(.*?)(?:</text>|$)", raw, re.S)
    # unescaped twice: once for the XML the page is stored in, once for the
    # entities the wikitext itself writes (&ndash;, &nbsp;)
    t = html.unescape(html.unescape(m.group(1) if m else raw))
    if t.lstrip().lower().startswith("#redirect"):
        return None
    for _ in range(8):                       # templates nest; innermost first
        t2 = re.sub(r"\{\{[^{}]*\}\}", "", t)
        if t2 == t:
            break
        t = t2
    t = re.sub(r"\{\|.*?\|\}", "", t, flags=re.S)
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)
    t = re.sub(r"<ref[^>]*/>", "", t)
    t = re.sub(r"<ref[^>]*>.*?</ref>", "", t, flags=re.S)
    t = re.sub(r"<[^>]+>", "", t)
    t = re.sub(r"\[\[(?:[^\[\]|]*:)[^\[\]]*(?:\[\[[^\]]*\]\][^\[\]]*)*\]\]",
               "", t)                        # files, categories, other wikis
    t = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"\[\[([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"\[https?://\S+ ([^\]]*)\]", r"\1", t)
    t = re.sub(r"\[https?://[^\]]*\]", "", t)
    t = re.sub(r"'{2,}", "", t)
    paras, cur = [], []
    for line in t.splitlines():
        s = line.strip()
        if not s or s.startswith(("=", "*", "#", ":", ";", "|", "!", "{",
                                  "}")):
            if cur:
                paras.append(" ".join(cur))
                cur = []
            continue
        cur.append(" ".join(s.split()))
    if cur:
        paras.append(" ".join(cur))
    return "\n\n".join(p for p in paras if len(p) > 1)


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")


def sentences(text, n):
    """The first n sentences, not split after an initial or a short
    abbreviation (J. S. Bach, St. Louis, c. 1900)."""
    out, cur = [], ""
    for piece in _SENT.split(text):
        cur = f"{cur} {piece}".strip() if cur else piece
        last = cur.rsplit(" ", 1)[-1]
        if len(last) <= 3 and last.endswith("."):
            continue
        out.append(cur)
        cur = ""
        if len(out) == n:
            break
    if cur and len(out) < n:
        out.append(cur)
    return " ".join(out)


def _clip(s, n):
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + " ..."


class Library:
    """Encyclopedia pages, title -> plain text, with a word index."""

    def __init__(self, pages):
        self.pages = dict(pages)
        self.titles = sorted(self.pages)
        self._lower = {t.lower(): t for t in self.titles}
        self._index = {}
        for t in self.titles:
            for w in set(_words(t)):
                self._index.setdefault(w, []).append(t)

    def search(self, query, k=3):
        """The k titles that share the most words with the query, an exact
        title first and closer titles before longer ones."""
        q = query.strip().lower()
        votes = Counter()
        for w in set(_words(query)):
            for t in self._index.get(w, [])[:5000]:
                votes[t] += 1
        best = sorted(votes, key=lambda t: (
            t.lower() != q, -votes[t] / (len(set(_words(t))) + 0.5),
            len(t), t))[:k]
        return [(t, self.lead(t)) for t in best]

    def page(self, title):
        t = self._lower.get(str(title).strip().lower())
        return None if t is None else self.pages[t]

    def lead(self, title):
        return self.pages[title].split("\n\n", 1)[0]

    @classmethod
    def from_lane(cls, root, n, held_out, rng):
        """Up to n pages of the wikipedia lane's files. A page is on the
        held-out side when a hash of its file name says so, one in a hundred,
        so the episodes held out ask about pages the training ones never
        mention - and the side is known before a file is opened."""
        names = []
        for dp, _, fs in os.walk(root):
            names.extend(os.path.join(dp, f) for f in fs if f.endswith(".txt")
                         and (zlib.crc32(f.encode()) % 100 == 0) == held_out)
        names.sort()
        rng.shuffle(names)
        pages = {}
        for path in names:
            if len(pages) >= n:
                break
            try:
                with open(path, errors="replace") as f:
                    raw = f.read(400_000)
            except OSError:
                continue
            m = re.search(r"<title>(.*?)</title>", raw)
            title = html.unescape(m.group(1)).strip() if m else \
                os.path.splitext(os.path.basename(path))[0]
            text = wiki_text(raw)
            if not text or ":" in title:
                continue
            first = sentences(text.split("\n\n", 1)[0], 1)
            if not (40 <= len(first) <= 400) or re.search(r"[{}\[\]|<>]",
                                                          first):
                continue
            pages[title] = text[:6000]
        return cls(pages) if pages else None


# ----------------------------------------------------------------- world
def _path(p):
    p = "/" + str(p).strip().strip("/")
    return re.sub(r"/+", "/", p)


def _date(s):
    try:
        return dt.date.fromisoformat(str(s).strip())
    except ValueError:
        raise ToolError(f"{s!r} is not a date written YYYY-MM-DD")


def _memkey(k):
    k = " ".join(str(k).lower().split()).strip(" ?.!")
    for lead in ("my ", "the ", "our "):
        if k.startswith(lead):
            k = k[len(lead):]
    return k


class World:
    """What one episode's tools act on: files, saved notes, the library."""

    def __init__(self, files=None, memory=None, library=None):
        self.files = dict(files or {})
        self.memory = dict(memory or {})
        self.library = library

    def copy(self):
        return World(self.files, self.memory, self.library)

    def run(self, name, args):
        """A call's result, as the model reads it. Mistakes come back as an
        error to read, as they would from a real tool, never as an
        exception."""
        spec = SPECS.get(name)
        if spec is None:
            return f"Error: there is no tool called {name}."
        if not isinstance(args, dict):
            return "Error: the arguments must be an object."
        params = spec["parameters"]
        missing = [k for k in params.get("required", []) if k not in args]
        if missing:
            return f"Error: missing {', '.join(missing)}."
        unknown = [k for k in args if k not in params["properties"]]
        if unknown:
            return f"Error: {name} has no argument {unknown[0]}."
        try:
            return getattr(self, "_" + name)(**args)
        except ToolError as e:
            return f"Error: {e}."

    def _calculator(self, expression):
        return calc(expression)

    def _inside(self, p):
        root = p.rstrip("/") + "/"
        return [f for f in sorted(self.files) if f.startswith(root)]

    def _list_files(self, path):
        p = _path(path)
        if p in self.files:
            raise ToolError(f"{p} is a file; read_file reads it")
        root = p.rstrip("/") + "/"
        kids = {}
        for f in self._inside(p):
            head, _, rest = f[len(root):].partition("/")
            kids[head + "/" if rest else head] = None if rest else \
                len(self.files[f].encode())
        if not kids:
            raise ToolError(f"no such folder: {p}")
        return "\n".join(k if v is None else f"{k}  {v} bytes"
                         for k, v in sorted(kids.items()))

    def _read_file(self, path):
        p = _path(path)
        if p not in self.files:
            if self._inside(p):
                raise ToolError(f"{p} is a folder; list_files shows what is "
                                "in it")
            raise ToolError(f"no such file: {p}")
        return self.files[p] or "(empty file)"

    def _search_files(self, text, path="/"):
        p = _path(path)
        want = str(text).lower()
        if not want.strip():
            raise ToolError("nothing to search for")
        scope = [p] if p in self.files else self._inside(p)
        if not scope:
            raise ToolError(f"no such file or folder: {p}")
        hits = [f"{f}:{i}: {line}" for f in scope
                for i, line in enumerate(self.files[f].splitlines(), 1)
                if want in line.lower()]
        if not hits:
            return "No matches."
        more = len(hits) - 30
        return "\n".join(hits[:30]) + (f"\n... and {more} more" if more > 0
                                       else "")

    def _write_file(self, path, content):
        p = _path(path)
        if self._inside(p):
            raise ToolError(f"{p} is a folder")
        self.files[p] = str(content)
        return f"Wrote {len(str(content).encode())} bytes to {p}."

    def _remember(self, key, value):
        k = " ".join(str(key).split())
        if not k:
            raise ToolError("the key is empty")
        self.memory[k] = str(value)
        return f"Saved: {k}."

    def _recall(self, key):
        want = _memkey(key)
        hits = [(k, v) for k, v in self.memory.items()
                if want and (want in _memkey(k) or _memkey(k) in want)]
        if not hits:
            return f"Nothing saved under '{key}'."
        return "\n".join(f"{k}: {v}" for k, v in hits)

    def _search(self, query):
        if self.library is None:
            raise ToolError("the encyclopedia is not available")
        hits = self.library.search(str(query))
        if not hits:
            return "No results."
        return "\n".join(f"{i}. {t}: {_clip(s, 160)}"
                         for i, (t, s) in enumerate(hits, 1))

    def _open_page(self, title):
        page = None if self.library is None else self.library.page(title)
        if page is None:
            raise ToolError(f"no page titled {title!r}; search finds titles")
        return _clip(page, 1500)

    def _date_info(self, date):
        d = _date(date)
        return (f"{d.isoformat()} is a {WEEKDAYS[d.weekday()]}, day "
                f"{d.timetuple().tm_yday} of {d.year}, ISO week "
                f"{d.isocalendar()[1]}.")

    def _days_between(self, start, end):
        return f"{(_date(end) - _date(start)).days} days"

    def _add_days(self, date, days):
        if not isinstance(days, int) or isinstance(days, bool):
            raise ToolError("days must be a whole number")
        try:
            d = _date(date) + dt.timedelta(days=days)
        except OverflowError:
            raise ToolError("that date is out of range")
        return f"{d.isoformat()} ({WEEKDAYS[d.weekday()]})"

    def _convert_units(self, value, from_unit, to_unit):
        v, a, b = _amount(value), _unit(from_unit), _unit(to_unit)
        (da, fa), (db, fb) = UNITS[a], UNITS[b]
        if da != db:
            raise ToolError(f"cannot convert {da} to {db}")
        r = _from_kelvin(_to_kelvin(v, a), b) if da == "temperature" \
            else v * fa / fb
        return f"{number(v)} {a} = {significant(r)} {b}"


# ------------------------------------------------------------- recording
class Turn:
    """A bot turn being written. Every call is run on the world as it is
    recorded, so a result is never anything but what the tool said."""

    def __init__(self, world, used):
        self.world, self.used, self.steps = world, used, []

    def think(self, text):
        self.steps.append(("think", text))

    def call(self, name, **args):
        out = self.world.run(name, args)
        self.steps += [("call", name, args), ("result", out)]
        self.used.add(name)
        return out

    def say(self, text):
        self.steps.append(("text", text))


class Script:
    """One episode's turns, the tools it needs and offers, and what each
    request should be answered with."""

    def __init__(self, world):
        self.world, self.turns = world, []
        self.used, self.offer, self.answers = set(), set(), []

    def user(self, text):
        self.turns.append(("user", text))

    def bot(self):
        t = Turn(self.world, self.used)
        self.turns.append(("bot", t.steps))
        return t


# ------------------------------------------------------------- wordlists
APPS = ["nginx", "redis", "postgres", "billing", "inventory", "scheduler",
        "gateway", "mailer", "search-api", "payments", "telemetry", "backup",
        "auth", "ledger", "uploader", "notifier", "reports", "metrics"]
USERS = ["anna", "omar", "li", "sofia", "kwame", "maria", "jonas", "priya",
         "tomas", "yuki", "elena", "david", "amara", "lucas", "nadia", "ivan"]
PEOPLE = ["Alice", "Bruno", "Chen", "Dalia", "Emeka", "Farah", "Goran",
          "Hana", "Isabel", "Jamal", "Keiko", "Lars", "Mei", "Nikos", "Olga",
          "Pedro", "Quinn", "Rosa", "Samir", "Tess", "Umar", "Vera", "Wen",
          "Ximena", "Yusuf", "Zoe", "Aiko", "Bilal", "Carmen", "Dmitri"]
CITIES = ["Lisbon", "Nairobi", "Osaka", "Lima", "Oslo", "Hanoi", "Toronto",
          "Krakow", "Accra", "Seville", "Brisbane", "Tbilisi", "Quito",
          "Porto", "Lyon", "Pune", "Cork", "Graz", "Tunis", "Bergen"]
THINGS = ["budget", "roadmap", "launch", "audit", "offsite", "hiring plan",
          "migration", "quarterly review", "vendor contract", "design review",
          "training day", "security drill", "inventory count", "newsletter"]
CHORES = ["buy milk", "renew passport", "call the plumber", "book the dentist",
          "pay rent", "fix the bike", "water the plants", "back up laptop",
          "return library books", "order printer ink", "cancel old gym plan",
          "pick up dry cleaning", "send the invoice", "update the CV",
          "repot the fern", "buy a birthday card", "check tyre pressure"]
PROJECTS = ["atlas", "beacon", "comet", "delta", "ember", "falcon", "harbor",
            "juniper", "kestrel", "lumen", "mosaic", "nimbus", "orbit",
            "pioneer", "quartz", "relay"]
DOES = ["resizes uploaded images", "sends the weekly digest email",
        "tracks warehouse stock", "exports invoices to CSV",
        "checks links on the docs site", "rotates the API keys",
        "turns sensor readings into hourly averages",
        "keeps the team calendar in sync"]
ITEMS = [("notebooks", "notebook"), ("pens", "pen"), ("coffee bags", "coffee bag"),
         ("light bulbs", "light bulb"), ("USB cables", "USB cable"),
         ("tickets", "ticket"), ("mugs", "mug"), ("batteries", "battery")]
FACTS = [  # how it is asked for, the key it is saved under, how it is told
    ("What's my locker code?", "locker code", "my locker code is {v}",
     "Your locker code is {v}.", "digits4"),
    ("What's the wifi password?", "wifi password", "the wifi password is {v}",
     "The wifi password is {v}.", "password"),
    ("When is my dentist appointment?", "dentist appointment",
     "my dentist appointment is on {v}", "Your dentist appointment is on {v}.",
     "when"),
    ("Where did I park?", "parking spot", "I parked on {v}",
     "You parked on {v}.", "parking"),
    ("What's my bike lock combination?", "bike lock combination",
     "my bike lock combination is {v}", "Your bike lock combination is {v}.",
     "digits3"),
    ("What's the office door code?", "office door code",
     "the office door code is {v}", "The office door code is {v}.", "door"),
    ("What's my flight number?", "flight number", "my flight number is {v}",
     "Your flight number is {v}.", "flight"),
    ("Which room is the workshop in?", "workshop room",
     "the workshop is in room {v}", "The workshop is in room {v}.", "room"),
    ("What's my library card number?", "library card number",
     "my library card number is {v}", "Your library card number is {v}.",
     "digits8"),
    ("Who is my doctor?", "doctor", "my doctor is {v}", "Your doctor is {v}.",
     "doctor"),
]
CANT = [  # a request no tool here serves, and what it would take
    ("What's the weather in {city} right now?", "check the weather"),
    ("Send an email to {person} saying I'll be late.", "send email"),
    ("Book a table for two at eight tonight.", "make bookings"),
    ("What's the latest news about the {thing}?", "read the news"),
    ("Play some music, please.", "play music"),
    ("Order more printer paper.", "place orders"),
    ("Text {person} that the meeting moved.", "send messages"),
    ("What time does the {city} train station open?", "look up timetables"),
    ("Translate this menu photo for me.", "read photos"),
]
CAN = {"calculator": "do exact arithmetic",
       "list_files": "work with the files here", "read_file": None,
       "search_files": None, "write_file": None,
       "remember": "save and recall notes", "recall": None,
       "search": "look things up in the encyclopedia", "open_page": None,
       "date_info": "work with dates", "days_between": None, "add_days": None,
       "convert_units": "convert units"}


def _value(rng, kind):
    if kind.startswith("digits"):
        n = int(kind[6:])
        return "".join(rng.choice("0123456789") for _ in range(n))
    if kind == "password":
        return rng.choice(["maple", "harbor", "copper", "violet", "falcon",
                           "pepper"]) + str(rng.randint(10, 9999))
    if kind == "when":
        return (f"{MONTHS[rng.randrange(12)]} {rng.randint(1, 28)} at "
                f"{rng.randint(8, 17)}:{rng.choice(['00', '15', '30', '45'])}")
    if kind == "parking":
        return f"level {rng.randint(1, 6)}, row {rng.choice('ABCDEFGH')}"
    if kind == "door":
        return "".join(rng.choice("0123456789") for _ in range(4)) + "#"
    if kind == "flight":
        return (rng.choice(["BA", "LH", "KL", "TK", "QR", "AF", "IB"])
                + " " + str(rng.randint(100, 9999)))
    if kind == "room":
        return f"{rng.randint(1, 5)}{rng.randint(0, 2)}{rng.randint(1, 9)}"
    if kind == "doctor":
        return f"Dr. {rng.choice(PEOPLE)} {rng.choice(['Okafor', 'Lindqvist', 'Moreau', 'Tanaka', 'Silva', 'Kowalski'])}"
    raise ValueError(kind)


def _when(rng):
    return (f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d} "
            f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:"
            f"{rng.randint(0, 59):02d}")


def make_files(rng):
    """A machine's worth of files: a service's config and log, a person's
    todo list and notes, a table, a project."""
    app, user, proj = rng.choice(APPS), rng.choice(USERS), rng.choice(PROJECTS)
    files = {}
    conf = {"host": rng.choice(["0.0.0.0", "127.0.0.1", f"10.0.{rng.randint(0, 9)}.{rng.randint(2, 250)}"]),
            "port": str(rng.randint(1025, 65000)),
            "log_level": rng.choice(["debug", "info", "warning", "error"]),
            "workers": str(rng.randint(1, 32)),
            "timeout_s": str(rng.choice([5, 10, 15, 30, 60, 90, 120])),
            "data_dir": f"/var/lib/{app}"}
    keys = list(conf)
    rng.shuffle(keys)
    files[f"/etc/{app}/{app}.conf"] = (f"# {app} settings\n" + "\n".join(
        f"{k} = {conf[k]}" for k in keys) + "\n")
    if rng.random() < 0.5:
        files[f"/etc/{app}/README"] = (f"Settings for {app} live in "
                                       f"{app}.conf, one per line.\n")
    hosts = ["db1", "db2", "cache1", "queue1", "api2", "storage3"]
    lines = []
    for _ in range(rng.randint(6, 24)):
        level = rng.choices(["INFO", "WARN", "ERROR"], [6, 2, 1])[0]
        msg = {"INFO": [f"Started worker {rng.randint(1, 16)}",
                        f"Served request in {rng.randint(3, 900)} ms",
                        f"Cache warmed with {rng.randint(100, 90000)} keys",
                        "Config reloaded", "Health check passed",
                        f"Job {rng.choice(['cleanup', 'export', 'sync'])} finished"],
               "WARN": [f"Slow query took {rng.randint(1000, 9000)} ms",
                        f"Retrying connection to {rng.choice(hosts)}",
                        f"Disk usage at {rng.randint(80, 97)}%"],
               "ERROR": [f"Connection refused by {rng.choice(hosts)}",
                         f"Timed out after {rng.randint(5, 60)} s waiting for {rng.choice(hosts)}",
                         f"Could not write {conf['data_dir']}/state.db",
                         f"Worker {rng.randint(1, 16)} crashed"]}[level]
        lines.append((_when(rng), level, rng.choice(msg)))
    lines.sort()
    files[f"/var/log/{app}.log"] = "".join(f"{t} {lv} {m}\n"
                                           for t, lv, m in lines)
    chores = rng.sample(CHORES, rng.randint(2, 6))
    files[f"/home/{user}/todo.txt"] = "".join(f"- {c}\n" for c in chores)
    notes = []
    for _ in range(rng.randint(2, 5)):
        p, th = rng.choice(PEOPLE), rng.choice(THINGS)
        notes.append(rng.choice([
            f"Call {p} about the {th} on {rng.choice(WEEKDAYS)}.",
            f"The {th} meeting moved to {rng.randint(9, 17)}:{rng.choice(['00', '30'])}.",
            f"{p}'s new extension is {rng.randint(100, 999)}.",
            f"Send the {th} notes to {p} before Friday.",
            f"Parking at the {rng.choice(CITIES)} office is {rng.randint(3, 25)} a day."]))
    files[f"/home/{user}/notes.txt"] = "\n".join(dict.fromkeys(notes)) + "\n"
    table = rng.choice(["team", "club", "class", "panel"])
    names = rng.sample(PEOPLE, rng.randint(4, 8))
    scores = rng.sample(range(40, 100), len(names))
    files[f"/data/{table}.csv"] = "name,city,age,score\n" + "".join(
        f"{n},{rng.choice(CITIES[:8])},{rng.randint(19, 68)},{s}\n"
        for n, s in zip(names, scores))
    files[f"/projects/{proj}/README.md"] = (f"# {proj}\n\n{proj.capitalize()} "
                                            f"{rng.choice(DOES)}.\n")
    files[f"/projects/{proj}/VERSION"] = (f"{rng.randint(0, 4)}."
                                          f"{rng.randint(0, 20)}."
                                          f"{rng.randint(0, 30)}\n")
    return files


def make_memory(rng):
    out = {}
    for _ask, key, _tell, _say, kind in rng.sample(FACTS, rng.randint(0, 4)):
        out[key] = _value(rng, kind)
    return out


def _find(files, pattern):
    return next((f for f in sorted(files) if re.fullmatch(pattern, f)), None)


def _human(n):
    """A whole number as people write it in a sentence."""
    return f"{n:,}" if abs(n) >= 10_000 else str(n)


def _money(s):
    c = _half_up(Fraction(s), 2)
    whole, cents = divmod(abs(int(c * 100)), 100)
    return ("-" if c < 0 else "") + f"{whole:,}.{cents:02d}"


def _date_words(rng, d):
    return rng.choice([f"{MONTHS[d.month - 1]} {d.day}, {d.year}",
                       f"{d.day} {MONTHS[d.month - 1]} {d.year}",
                       d.isoformat()])


def _rand_date(rng, lo=1900, hi=2100):
    start = dt.date(lo, 1, 1).toordinal()
    return dt.date.fromordinal(rng.randint(start,
                                           dt.date(hi, 12, 31).toordinal()))


# ----------------------------------------------------------------- tasks
# Each task writes one request and its answer into the script, running every
# call as it goes, and returns True - or gives up, returning nothing, and the
# episode takes back whatever it wrote. Every reply is composed from what the
# calls returned, not from what the task knows the answer to be.

def t_arithmetic(s, rng):
    s.offer.add("calculator")
    kind = rng.choice(["mul", "mul", "add", "sub", "div", "pow", "mod", "gcd",
                       "mix", "easy"])
    if kind == "easy":                         # no tool for what is easy
        a, b = rng.randint(0, 9), rng.randint(0, 9)
        op = rng.choice(["+", "×", "-"])
        v = a + b if op == "+" else a * b if op == "×" else a - b
        s.user(rng.choice([f"What's {a} {op} {b}?", f"Quick one: {a} {op} {b}?"]))
        s.bot().say(f"{a} {op} {b} = {v}.")
        s.answers.append(str(v))
        return True

    def big(lo, hi):
        return rng.randint(10 ** (lo - 1), 10 ** hi - 1)

    times = rng.choice(["×", "*", "times"])
    if kind == "mul":
        a, b = big(3, 7), big(2, 5)
        expr, said = f"{a} * {b}", f"{_human(a)} {times} {_human(b)}"
        reply = f"{_human(a)} × {_human(b)} = {{r}}."
    elif kind == "add":
        xs = [big(3, 8) for _ in range(rng.randint(3, 6))]
        expr, said = " + ".join(map(str, xs)), " + ".join(map(_human, xs))
        reply = "The sum is {r}."
    elif kind == "sub":
        a, b = big(5, 9), big(4, 9)
        expr, said = f"{a} - {b}", f"{_human(a)} minus {_human(b)}"
        reply = f"{_human(a)} - {_human(b)} = {{r}}."
    elif kind == "div":
        a, b = big(4, 8), big(2, 4)
        expr, said = f"{a} / {b}", f"{_human(a)} divided by {_human(b)}"
        reply = f"{_human(a)} ÷ {_human(b)} = {{r}}."
    elif kind == "pow":
        a, b = rng.randint(2, 19), rng.randint(5, 15)
        expr, said = f"{a} ** {b}", f"{a} to the power of {b}"
        reply = f"{a}^{b} = {{r}}."
    elif kind == "mod":
        a, b = big(5, 9), big(2, 4)
        expr = f"{a} % {b}"
        said = f"the remainder when {_human(a)} is divided by {_human(b)}"
        reply = "The remainder is {r}."
    elif kind == "gcd":
        g = rng.randint(2, 999)
        a, b = g * rng.randint(2, 9999), g * rng.randint(2, 9999)
        expr = f"gcd({a}, {b})"
        said = f"the greatest common divisor of {_human(a)} and {_human(b)}"
        reply = "Their greatest common divisor is {r}."
    else:
        a, b, c, d = big(2, 4), big(2, 4), rng.randint(2, 99), big(2, 5)
        expr, said = f"({a} + {b}) * {c} - {d}", f"({a} + {b}) × {c} - {d}"
        reply = f"({a} + {b}) × {c} - {d} = {{r}}."
    s.user(rng.choice([f"What is {said}?", f"Calculate {said}.",
                       f"Can you work out {said}?", f"I need {said}."]))
    t = s.bot()
    if rng.random() < 0.3:
        t.think("Too many digits to get right by hand; the calculator does it "
                "exactly.")
    res = t.call("calculator", expression=expr)
    t.say(reply.format(r=res))
    s.answers.append(res)
    return True


def t_word_problem(s, rng):
    s.offer.add("calculator")
    kind = rng.choice(["shop", "split", "percent", "change", "mean",
                       "interest", "speed", "unit_price"])
    if kind == "shop":
        (p1, s1), (p2, s2) = rng.sample(ITEMS, 2)
        n1, n2 = rng.randint(2, 12), rng.randint(1, 9)
        c1, c2 = rng.randint(50, 4999) / 100, rng.randint(50, 4999) / 100
        ask = (f"I bought {n1} {p1} at ${c1:.2f} each and {n2} "
               f"{p2 if n2 > 1 else s2} at ${c2:.2f} each. What did I spend "
               "in total?")
        expr, reply = f"{n1} * {c1:.2f} + {n2} * {c2:.2f}", "You spent ${m}."
    elif kind == "split":
        total, k = rng.randint(2000, 60000) / 100, rng.randint(2, 9)
        tip = rng.choice([10, 12, 15, 18, 20])
        ask = (f"Dinner came to ${total:.2f} for {k} of us and we're adding "
               f"a {tip}% tip. How much does each person pay?")
        expr = f"round({total:.2f} * (1 + {tip} / 100) / {k}, 2)"
        reply = "Each person pays ${m}."
    elif kind == "percent":
        p = rng.choice([2.5, 7, 12.5, 15, 17.5, 23, 33, 45, 62.5])
        n = rng.randint(40, 990000)
        ask = f"What is {p}% of {_human(n)}?"
        expr, reply = f"{p} / 100 * {n}", f"{p}% of {_human(n)} is {{r}}."
    elif kind == "change":
        a = rng.randint(20, 5000)
        b = max(1, a + rng.randint(-a // 2, a))
        ask = f"A price went from {a} to {b}. By what percentage did it change?"
        expr, reply = f"round(({b} - {a}) / {a} * 100, 2)", None
    elif kind == "mean":
        xs = [rng.randint(1, 999) for _ in range(rng.randint(3, 8))]
        ask = (f"What's the average of {', '.join(map(str, xs[:-1]))} and "
               f"{xs[-1]}?")
        expr, reply = f"({' + '.join(map(str, xs))}) / {len(xs)}", \
            "The average is {r}."
    elif kind == "interest":
        P, y = rng.randint(5, 500) * 100, rng.randint(2, 30)
        r = rng.choice([1.5, 2, 3, 3.5, 4, 5, 6.5])
        ask = (f"If I put ${_human(P)} in an account paying {r}% a year, "
               f"compounded yearly, how much is there after {y} years?")
        expr = f"round({P} * (1 + {r} / 100) ** {y}, 2)"
        reply = f"After {y} years there would be ${{m}}."
    elif kind == "speed":
        d, h = rng.randint(30, 900), rng.randint(0, 9)
        m = rng.choice([10, 15, 20, 30, 40, 45, 50])
        ask = (f"A train covers {d} km in {h} hours and {m} minutes. What's "
               "its average speed?")
        expr = f"round({d} / ({h} + {m} / 60), 2)"
        reply = "Its average speed is {r} km/h."
    else:
        g = rng.choice([125, 200, 250, 340, 400, 500, 750])
        c = rng.randint(99, 1999) / 100
        ask = f"A {g} g pack costs ${c:.2f}. What is that per kilogram?"
        expr, reply = f"round({c:.2f} / {g} * 1000, 2)", "That's ${m} per kilogram."
    s.user(ask)
    t = s.bot()
    if rng.random() < 0.3:
        t.think(f"As one expression: {expr}")
    res = t.call("calculator", expression=expr)
    if res.startswith("Error"):
        return None
    if reply is None:
        v = Fraction(res)
        t.say(f"It rose by {res}%." if v > 0 else
              f"It fell by {number(-v)}%." if v < 0 else "It did not change.")
    else:
        t.say(reply.format(r=res, m=_money(res)))
    s.answers.append(res)
    return True


def t_config(s, rng):
    conf = _find(s.world.files, r"/etc/[^/]+/[^/]+\.conf")
    app = conf.split("/")[2]
    key = rng.choice(["port", "log_level", "workers", "timeout_s", "host"])
    ask = {"port": f"What port does {app} listen on?",
           "log_level": f"What log level is {app} set to?",
           "workers": f"How many workers is {app} configured with?",
           "timeout_s": f"What's {app}'s timeout?",
           "host": f"Which address does {app} bind to?"}[key]
    how = rng.choice(["find", "find", "given", "wrong"])
    s.offer |= {"list_files", "read_file"}
    if how == "given":
        ask += f" The config is {conf}."
    elif how == "wrong":
        ask += f" I think the config is /etc/{app}.conf."
    s.user(ask)
    t = s.bot()
    if how == "find":
        if rng.random() < 0.3:
            t.think(f"Settings for {app} should be in a file under /etc/{app}.")
        t.call("list_files", path=f"/etc/{app}")
    elif how == "wrong":
        t.call("read_file", path=f"/etc/{app}.conf")
        t.call("list_files", path=f"/etc/{app}")
    body = t.call("read_file", path=conf)
    m = re.search(rf"^{key} = (.+)$", body, re.M)
    if not m:
        return None
    v = m.group(1)
    t.say({"port": f"{app} listens on port {v}.",
           "log_level": f"{app}'s log level is {v}.",
           "workers": f"{app} is configured with {v} workers.",
           "timeout_s": f"{app}'s timeout is {v} seconds.",
           "host": f"{app} binds to {v}."}[key]
          + (f" The config is {conf}, not /etc/{app}.conf." if how == "wrong"
             else ""))
    s.answers.append(v)
    return True


def t_log(s, rng):
    log = _find(s.world.files, r"/var/log/[^/]+\.log")
    app = log.split("/")[-1][:-4]
    level = rng.choice(["ERROR", "ERROR", "WARN"])
    word = {"ERROR": "errors", "WARN": "warnings"}[level]
    s.offer |= {"search_files", "read_file"}
    s.user(rng.choice([f"How many {word} has {app} logged?",
                       f"Did {app} log any {word}? How many?",
                       f"Count the {word} in the {app} log."]))
    t = s.bot()
    res = t.call("search_files", text=level, path=log)
    hits = [] if res == "No matches." else res.splitlines()
    if not hits:
        t.say(f"None: there are no {level} lines in {log}.")
        s.answers.append("0")
        return True
    stamp, rest = hits[-1].split(": ", 1)[1].split(f" {level} ", 1)
    t.say(f"{len(hits)} {word if len(hits) > 1 else word[:-1]} in {log}. The "
          f"latest, at {stamp}: {rest}.")
    s.answers.append(str(len(hits)))
    return True


def t_table(s, rng):
    path = _find(s.world.files, r"/data/[^/]+\.csv")
    kind = rng.choice(["best", "mean", "city"])
    s.offer.add("read_file")
    if kind == "mean":
        s.offer.add("calculator")
        s.user(f"What's the average age in {path}?")
    elif kind == "best":
        s.user(f"Who has the highest score in {path}?")
    else:
        city = rng.choice(CITIES[:8])
        s.user(f"How many people in {path} live in {city}?")
    t = s.bot()
    rows = [r.split(",") for r in
            t.call("read_file", path=path).strip().splitlines()[1:]]
    if kind == "best":
        top = max(rows, key=lambda r: int(r[3]))
        t.say(f"{top[0]}, with {top[3]}.")
        s.answers.append(top[0])
    elif kind == "mean":
        res = t.call("calculator",
                     expression=f"({' + '.join(r[2] for r in rows)}) / {len(rows)}")
        t.say(f"The average age is {res}.")
        s.answers.append(res)
    else:
        who = [r[0] for r in rows if r[1] == city]
        t.say(f"{len(who)}: {', '.join(who)}." if who else
              f"None of them live in {city}.")
        s.answers.append(str(len(who)))
    return True


def t_todo(s, rng):
    path = _find(s.world.files, r"/home/[^/]+/todo\.txt")
    chore = rng.choice([c for c in CHORES
                        if f"- {c}\n" not in s.world.files[path]])
    s.offer |= {"list_files", "read_file", "write_file"}
    given = rng.random() < 0.5
    s.user(f"Add '{chore}' to my todo list" + (f" ({path})." if given else "."))
    t = s.bot()
    if not given:
        home = t.call("list_files", path="/home").splitlines()[0].rstrip("/")
        t.call("list_files", path=f"/home/{home}")
    body = t.call("read_file", path=path)
    t.call("write_file", path=path, content=body.rstrip("\n") + f"\n- {chore}\n")
    t.say(f"Added \"{chore}\" to {path}.")
    s.answers.append(chore)
    return True


def t_find(s, rng):
    path = _find(s.world.files, r"/home/[^/]+/notes\.txt")
    line = rng.choice(s.world.files[path].strip().splitlines())
    names = [p for p in PEOPLE if p in line] + [t for t in THINGS if t in line]
    if not names:
        return None
    what = rng.choice(names)
    s.offer |= {"search_files", "read_file"}
    s.user(rng.choice([f"Which file mentions {what}?",
                       f"Where did I write something about {what}?",
                       f"Find where {what} comes up in my files."]))
    t = s.bot()
    res = t.call("search_files", text=what, path="/")
    if res == "No matches.":
        return None
    hits = res.splitlines()
    where = sorted({h.split(":", 1)[0] for h in hits})
    if len(hits) == 1:
        f, n, text = hits[0].split(":", 2)
        t.say(f"{f}, line {n}: {text.strip()}")
    else:
        t.say(f"{what} comes up {len(hits)} times, in "
              + ", ".join(where) + ".")
    s.answers.append(" ".join(where))
    return True


def t_version(s, rng):
    vfile = _find(s.world.files, r"/projects/[^/]+/VERSION")
    proj = vfile.split("/")[2]
    s.offer |= {"list_files", "read_file"}
    s.user(rng.choice([f"What version is {proj} at?",
                       f"Which version of {proj} is checked out here?"]))
    t = s.bot()
    if rng.random() < 0.5:
        t.call("list_files", path=f"/projects/{proj}")
    v = t.call("read_file", path=vfile).strip()
    t.say(f"{proj} is at version {v}.")
    s.answers.append(v)
    return True


def t_memory(s, rng):
    s.offer |= {"remember", "recall"}
    facts = {f[1]: f for f in FACTS}
    known = [k for k in s.world.memory if k in facts]
    kind = rng.choice(["recall", "recall", "missing", "store"] if known
                      else ["missing", "store", "store"])
    if kind == "recall":
        key = rng.choice(known)
        ask, _, _, say, _ = facts[key]
        s.user(ask)
        t = s.bot()
        lines = dict(x.split(": ", 1) for x in
                     t.call("recall", key=key).splitlines() if ": " in x)
        if key not in lines:
            return None
        t.say(say.format(v=lines[key]))
        s.answers.append(lines[key])
    elif kind == "missing":
        ask, key, _, _, _ = rng.choice([f for f in FACTS
                                        if f[1] not in s.world.memory])
        s.user(ask)
        t = s.bot()
        if not t.call("recall", key=key).startswith("Nothing saved"):
            return None
        t.say(f"I don't have your {key} saved. If you tell me, I'll "
              "remember it.")
        s.answers.append("")
    else:
        _, key, tell, say, kind_v = rng.choice(FACTS)
        v = _value(rng, kind_v)
        s.user(f"Remember that {tell.format(v=v)}.")
        t = s.bot()
        t.call("remember", key=key, value=v)
        t.say(f"Saved. {say.format(v=v)}")
        s.answers.append(v)
    return True


def t_lookup(s, rng):
    lib = s.world.library
    if lib is None:
        return None
    s.offer |= {"search", "open_page"}
    kind = rng.choice(["what", "what", "what", "two", "none"])
    if kind == "none":
        made = "".join(rng.choice(["ka", "lo", "ve", "rin", "tha", "mu", "zor",
                                   "pel", "qui", "dra"]) for _ in range(3))
        made = made.capitalize()
        s.user(rng.choice([f"Look up {made} for me.", f"What is {made}?"]))
        t = s.bot()
        if t.call("search", query=made) != "No results.":
            return None
        t.say(f"The encyclopedia has nothing on {made}, so I can't say what "
              "it is.")
        s.answers.append("")
        return True
    title = rng.choice(lib.titles)
    text = lib.lead(title) if kind == "what" else lib.page(title)
    want = sentences(" ".join(text.split()), 1 if kind == "what" else 2)
    if kind == "what":
        s.user(rng.choice([f"Tell me about {title}.", f"Look up {title}.",
                           f"What does the encyclopedia say about {title}?"]))
    else:
        s.user(f"Give me the first two sentences of the page on {title}.")
    t = s.bot()
    top = t.call("search", query=title).split("\n")[0]
    if not top.startswith(f"1. {title}: "):
        return None
    if kind == "two" or want not in top:     # the snippet cut it short
        if want not in t.call("open_page", title=title):
            return None
    t.say(f"From the page \"{title}\": {want}" if kind == "what" else want)
    s.answers.append(want)
    return True


def t_dates(s, rng):
    kind = rng.choice(["weekday", "between", "add"])
    if kind == "weekday":
        d = _rand_date(rng)
        s.offer.add("date_info")
        said = _date_words(rng, d)
        s.user(rng.choice([f"What day of the week was {said}?",
                           f"Which weekday is {said}?"]))
        t = s.bot()
        day = t.call("date_info", date=d.isoformat()).split(" is a ", 1)[1]
        day = day.split(",", 1)[0]
        t.say(f"{said} is a {day}.")
        s.answers.append(day)
    elif kind == "between":
        a = _rand_date(rng, 1990, 2060)
        b = a + dt.timedelta(days=rng.randint(-4000, 4000))
        s.offer.add("days_between")
        s.user(f"How many days are there from {_date_words(rng, a)} to "
               f"{_date_words(rng, b)}?")
        t = s.bot()
        n = int(t.call("days_between", start=a.isoformat(),
                       end=b.isoformat()).split()[0])
        t.say(f"{_human(n)} days." if n >= 0 else
              f"The second date is {_human(-n)} days before the first.")
        s.answers.append(str(n))
    else:
        d = _rand_date(rng, 1990, 2060)
        n = rng.choice([7, 10, 14, 30, 45, 60, 90, 100, 180, 365, 1000])
        n = n if rng.random() < 0.85 else -n
        s.offer.add("add_days")
        s.user(f"What's the date {abs(n)} days {'after' if n > 0 else 'before'} "
               f"{_date_words(rng, d)}?")
        t = s.bot()
        iso, day = t.call("add_days", date=d.isoformat(), days=n).split(" ", 1)
        e = dt.date.fromisoformat(iso)
        t.say(f"{day.strip('()')}, {MONTHS[e.month - 1]} {e.day}, {e.year}.")
        s.answers.append(iso)
    return True


_SAY_UNIT = {"m": "meters", "km": "kilometers", "cm": "centimeters",
             "mm": "millimeters", "mi": "miles", "yd": "yards", "ft": "feet",
             "in": "inches", "kg": "kilograms", "g": "grams", "lb": "pounds",
             "oz": "ounces", "l": "liters", "ml": "milliliters",
             "gal": "US gallons", "qt": "US quarts", "cup": "US cups",
             "m/s": "meters per second", "km/h": "kilometers per hour",
             "mph": "miles per hour", "kn": "knots", "C": "degrees Celsius",
             "F": "degrees Fahrenheit", "K": "kelvin"}


# the conversions people ask for, each with amounts that make sense for it
PAIRS = [("mi", "km", 1, 500), ("km", "mi", 1, 800), ("ft", "m", 1, 30000),
         ("m", "ft", 1, 9000), ("in", "cm", 1, 120), ("cm", "in", 1, 300),
         ("yd", "m", 1, 1000), ("mm", "in", 1, 1000), ("lb", "kg", 1, 400),
         ("kg", "lb", 1, 200), ("oz", "g", 1, 64), ("g", "oz", 1, 2000),
         ("gal", "l", 1, 100), ("l", "gal", 1, 400), ("cup", "ml", 1, 8),
         ("qt", "l", 1, 20), ("mph", "km/h", 5, 200), ("km/h", "mph", 5, 300),
         ("kn", "km/h", 1, 60), ("m/s", "km/h", 1, 100),
         ("F", "C", -40, 250), ("C", "F", -40, 120), ("K", "C", 0, 500),
         ("C", "K", -273, 1000)]


def t_units(s, rng):
    a, b, lo, hi = rng.choice(PAIRS)
    v = rng.randint(lo, hi) if rng.random() < 0.6 or a in ("C", "F", "K") \
        else round(rng.uniform(max(lo, 0.1), hi), 1)
    s.offer.add("convert_units")
    s.user(rng.choice([f"How many {_SAY_UNIT[b]} is {v} {_SAY_UNIT[a]}?",
                       f"Convert {v} {_SAY_UNIT[a]} to {_SAY_UNIT[b]}.",
                       f"{v} {a} in {b}?"]))
    t = s.bot()
    r = t.call("convert_units", value=v, from_unit=a, to_unit=b)
    r = r.split(" = ", 1)[1].rsplit(" ", 1)[0]
    t.say(f"{v} {_SAY_UNIT[a]} is {r} {_SAY_UNIT[b]}.")
    s.answers.append(r)
    return True


def t_cant(s, rng):
    ask, needs = rng.choice(CANT)
    s.user(ask.format(city=rng.choice(CITIES), person=rng.choice(PEOPLE),
                      thing=rng.choice(THINGS)))
    s.bot().say(f"I can't {needs} here: none of my tools does that.")
    s.answers.append("")
    return True


def t_clarify(s, rng):
    kind = rng.choice(["units", "weekday", "percent"])
    if kind == "units":
        s.offer.add("convert_units")
        s.user("Can you convert that to kilometers for me?")
        s.bot().say("Sure - what distance, and in which unit?")
        v = rng.randint(2, 300)
        s.user(f"{v} miles.")
        t = s.bot()
        r = t.call("convert_units", value=v, from_unit="mi", to_unit="km")
        r = r.split(" = ", 1)[1].rsplit(" ", 1)[0]
        t.say(f"{v} miles is {r} kilometers.")
        s.answers += ["", r]
    elif kind == "weekday":
        s.offer.add("date_info")
        s.user("What day of the week is my birthday this year?")
        s.bot().say("What's the date of your birthday, and which year do you "
                    "mean?")
        d = _rand_date(rng, 2020, 2035)
        s.user(f"{MONTHS[d.month - 1]} {d.day}, {d.year}.")
        t = s.bot()
        day = t.call("date_info", date=d.isoformat()).split(" is a ", 1)[1]
        day = day.split(",", 1)[0]
        t.say(f"It's a {day}.")
        s.answers += ["", day]
    else:
        s.offer.add("calculator")
        p = rng.choice([5, 8, 12, 15, 20, 25])
        s.user(f"How much is {p}% of it?")
        s.bot().say(f"{p}% of what amount?")
        n = rng.randint(100, 99999)
        s.user(f"{_human(n)}.")
        t = s.bot()
        res = t.call("calculator", expression=f"{p} / 100 * {n}")
        t.say(f"{p}% of {_human(n)} is {res}.")
        s.answers += ["", res]
    return True


TASKS = [(t_arithmetic, 14), (t_word_problem, 12), (t_config, 8), (t_log, 6),
         (t_table, 6), (t_todo, 4), (t_find, 4), (t_version, 3),
         (t_memory, 9), (t_lookup, 12), (t_dates, 9), (t_units, 6),
         (t_cant, 4), (t_clarify, 3)]


def episode(rng, library=None):
    """One episode: (transcript, the world as it was before the first call,
    the answer each request expects - "" where the right reply asks a
    question back or says it cannot be done).

    Most episodes hold one request; some hold two or three in a row against
    the same world, so a note saved in the first is there to recall in the
    third. The tools offered are the ones the requests need and up to three
    they do not, in random order, so choosing is part of the task."""
    start = World(make_files(rng), make_memory(rng), library)
    s = Script(start.copy())
    fns, weights = zip(*TASKS)
    for _ in range(rng.choices([1, 2, 3], [70, 22, 8])[0]):
        mark = (len(s.turns), len(s.answers), set(s.offer), set(s.used),
                s.world.copy())
        if not rng.choices(fns, weights)[0](s, rng):
            # take back the half-written request and whatever its calls did
            del s.turns[mark[0]:], s.answers[mark[1]:]
            s.offer, s.used, s.world = mark[2], mark[3], mark[4]
    if not s.turns:
        return None
    if rng.random() < 0.1:
        s.user(rng.choice(["Thanks!", "Great, thank you.", "Perfect, thanks."]))
        s.bot().say(rng.choice(["You're welcome.", "Glad to help.",
                                "Any time."]))
        s.answers.append("")
    need = s.offer | s.used
    rest = [n for n in SPECS if n not in need]
    if not need:                              # a "can't" still has tools
        need = set(rng.sample(rest, 2))
        rest = [n for n in rest if n not in need]
    names = sorted(need) + rng.sample(rest, rng.randint(0, min(3, len(rest))))
    rng.shuffle(names)
    return {"tools": [SPECS[n] for n in names], "turns": s.turns}, start, s.answers


def replay(ep, world):
    """Run an episode's calls again, in order, on a copy of the world it
    started from. Returns the calls whose result differs from the one
    recorded, as (call, recorded, rerun)."""
    w, bad = world.copy(), []
    for role, body in ep["turns"]:
        if role != "bot":
            continue
        for i, step in enumerate(body):
            if step[0] != "call":
                continue
            got = w.run(step[1], step[2])
            want = body[i + 1][1] if i + 1 < len(body) \
                and body[i + 1][0] == "result" else None
            if got != want:
                bad.append((step, want, got))
    return bad


def episodes(seed, wiki=None, pages=30000, held_out=False):
    """Episodes without end, from one seed. The encyclopedia is `pages` pages
    of the wikipedia lane at `wiki`, the held-out side's or the training
    side's; without it, look-ups are left out."""
    rng = random.Random(seed)
    lib = None
    if wiki and os.path.isdir(wiki):
        lib = Library.from_lane(wiki, pages, held_out, random.Random(seed))
    while True:
        got = episode(rng, lib)
        if got is not None:
            yield got[0]
