#!/usr/bin/env python3
"""
The corpus for the stage after the first pass: calling tools, acting in a
loop on what comes back, and ARC.

    python3 -m corpora agent --check     # the plan: sizes, licences, what is missing
    python3 -m corpora agent             # says what it needs, asks, then builds
    python3 -m corpora agent --only xlam toolbox --limit 200 --yes   # a trial

Nothing here runs by itself. --check only looks: it downloads nothing and
writes nothing. A build first prints what it will keep and write and the
memory it will take, and starts only once that is agreed to (--yes agrees
for it). Every dataset is taken whole: no curated subset stands in for the
rest of it.

ONE PROTOCOL FOR EVERY SOURCE. Each dataset writes a tool call its own way -
<tool_call> around one JSON object or around a list of them, <functioncall>
with the arguments as a quoted string, Python-style calls with spaces in the
function names, a message holding a Python dict's repr, OpenAI's
function_call, <TOOLCALL> lists, bare JSON answers. A model this size would
learn each convention separately for one act, so every source is rewritten
into one:

    <tools>
    {"name": "get_weather", "description": "...", "parameters": {...}}
    </tools>
    <user>
    What's the weather in Paris?
    </user>
    <bot>
    <call>{"name": "get_weather", "arguments": {"city": "Paris"}}</call>
    <result>
    {"temp_c": 18, "sky": "clear"}
    </result>
    It is 18 °C and clear in Paris.
    </bot>

then the end-of-text marker. The tools come first, outside any turn: they are
what the harness says, not what the user says. Each is one line of JSON
Schema, whatever dialect the source wrote it in (xLAM's flat parameters,
ToolACE's "dict", When2Call's "required" beside the parameters, MCP's
pydantic titles). A call is one JSON object between <call> and </call>; the
harness stops reading there, runs it, and writes the result back inside the
same bot turn, so the reply can quote it. Parallel calls are written one
call, one result, in order - the harness only ever has one call to run.

The tags are spelled as text. The markers the tokenizer treats as single
symbols are fixed with the vocabulary, and a new one would change its size;
<call> as six bytes costs six characters and nothing else.

SEVEN LANES, because they are different things to learn, and each held-out
subject should say which one is being learned:

    tools           generated and run here (corpora/toolbox.py): every result
                    is what the tool returned and every reply is composed
                    from it. Unlimited, so --tools-chars sets its size.
    function-calls  choosing a function and filling its arguments: xLAM,
                    ToolACE, Hermes, When2Call, SmolTalk, Glaive, and the tldr
                    pages for shell commands, in every language they have.
    trajectories    many steps, each reading what the last returned: Toucan's
                    runs against real MCP servers, all 1.6 million, and
                    TextWorld and ALFWorld played through by their own
                    solvers (corpora/worlds.py).
    arc             ARC as people made it: ARC-AGI-2's and ARC-AGI-1's training
                    tasks, ConceptARC and the community's sets, the hand-made
                    ones also turned, mirrored and recolored; and RE-ARC's
                    generated examples of ARC-AGI-1's rules (corpora/arc.py).
    arc-synthetic   tasks made at scale: BARC's, each also as the program that
                    makes it, and Nemotron-RL's puzzles.
    arc-traces      Nemotron-SFT's open models solving ARC puzzles, reasoning
                    and running Python as they go.
    arc-games       ARC-AGI-3's public games, played by a search
                    (corpora/arc3_play.py).

A new top-level folder of data/train is a new lane at the next session's
start, and data/val/<lane> its held-out subject: running this is what starts
the stage.

KEPT OUT OF TRAINING. When2Call's test sets and ALFWorld's valid_unseen
games are downloaded and kept for benchmarks; SmolTalk's test split and
ALFWorld's valid_seen go to held-out. ARC's public evaluation tasks are
copied to data/eval/arc-agi-1 and data/eval/arc-agi-2, every fifth public
ARC-AGI-3 game is listed in data/eval/arc-agi-3 and never played, and every
ARC task bound for training is checked against the evaluation tasks by name
and by content (corpora/arc.py). That costs 376 of ARC-AGI-2's 1,000
training tasks, which are ARC-AGI-1's evaluation tasks;
--arc1-eval-for-training takes them, and ARC-AGI-1's evaluation stops being
a clean benchmark. Everything else is split by a hash of the
first request - a long one, so a question and its duplicate land on the same
side - one record in a hundred held out.

DUPLICATES. SmolTalk's apigen-80k is largely xLAM's queries again, Hermes's
glaive-5k is Glaive's conversations again, and Toucan's SFT set is drawn
from its other three. Records are compared as rendered, with the tool list
left out, and the first source in SOURCES to have one keeps it; an ARC task
is compared by its grids, so all its copies follow it.

LICENCES are in SOURCES and in the manifest the build writes. xLAM,
When2Call, tldr-pages and Nemotron's sets are CC BY 4.0, which asks for
attribution wherever the model trained on them is published; Nemotron's
are for research and development. Four state no licence at all and are
built only when asked for by name: --with agentinstruct nvarc
nvarc-artifacts arc-community-unlicensed.

DOWNLOADS ARE KEPT, under data/raw/agent, unlike fetch.py's dataset cache:
a converter can then be fixed and rerun without fetching anything again.
`data` is outside what the self-knowledge lane copies, so none of it is
mistaken for the repository's own text.

RERUNS. A source already built is left alone - its duplicate keys are kept,
so the sources after it still compare against it - and --force rebuilds it.
A source is written under data/raw/agent/partial and moved into place when
complete, so a session starting mid-build never reads half a source. What
failed is named at the end with the command that retries it.
"""

import argparse
import ast
import glob
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import zipfile
from urllib.request import urlopen

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from minagi.tokenizer import SPECIALS                        # noqa: E402

EOT = "<|endoftext|>"
assert EOT in SPECIALS


# ------------------------------------------------------------- the format
# An episode is {"tools": [...], "schema": str or None, "turns": [...]}: a
# turn is ("user", text) or ("bot", steps), and a step is ("think", text),
# ("text", text), ("call", name, arguments) or ("result", text).

_MARK = re.compile("|".join(re.escape(m) for m in SPECIALS))
# terminal escapes (AgentInstruct's shell output sets the window title and
# erases lines) and every other control character but newline and tab
_CONTROL = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b\[[0-9;?]*[ -/]*[@-~]"
                      r"|\x1b[@-_]|[\x00-\x08\x0b-\x1f\x7f]")


def defuse(text):
    """Text from a source as it may appear inside a record: a marker the
    tokenizer would read as structure is written out as a spelling instead,
    a stray end-of-text marker (Glaive ends every reply with one) is
    dropped, and so are terminal escapes. A <user> inside a tool result would
    otherwise open a turn."""
    text = _CONTROL.sub("", str(text))
    return _MARK.sub(lambda m: "" if m.group(0) == EOT else
                     m.group(0).replace("<", "&lt;").replace(">", "&gt;"),
                     text).strip()


def interleave(steps):
    """Parallel calls as the harness meets them: k calls and then k results
    become call, result, call, result. Unequal runs are left as written."""
    out, i = [], 0
    while i < len(steps):
        if steps[i][0] != "call":
            out.append(steps[i])
            i += 1
            continue
        j = i
        while j < len(steps) and steps[j][0] == "call":
            j += 1
        k = j
        while k < len(steps) and steps[k][0] == "result":
            k += 1
        if j - i > 1 and j - i == k - j:
            for c, r in zip(steps[i:j], steps[j:k]):
                out += [c, r]
        else:
            out += steps[i:k]
        i = k
    return out


def _step(s):
    if s[0] == "think":
        return f"<think>\n{defuse(s[1])}\n</think>"
    if s[0] == "call":
        return "<call>" + defuse(json.dumps(
            {"name": s[1], "arguments": s[2]}, ensure_ascii=False)) + "</call>"
    if s[0] == "result":
        return f"<result>\n{defuse(s[1])}\n</result>"
    return defuse(s[1])


def tidy(ep):
    """The episode as it can be written, or None: user and bot alternate from
    a user turn, no step is empty, and the last bot turn does not stop on a
    tool result nobody read - a conversation cut off there would teach the
    model to fall silent after a call returns."""
    if not ep or not ep.get("turns"):
        return None
    turns = []
    for role, body in ep["turns"]:
        if role == "user":
            text = defuse(body)
            if not text:
                continue
            if turns and turns[-1][0] == "user":
                turns[-1] = ("user", turns[-1][1] + "\n\n" + text)
            else:
                turns.append(("user", text))
        else:
            steps = [s for s in body if s[0] == "call" or defuse(s[1])]
            if not steps or not turns:
                continue
            if turns[-1][0] == "bot":
                turns[-1] = ("bot", turns[-1][1] + steps)
            else:
                turns.append(("bot", steps))
    while turns and turns[-1][0] == "user":
        turns.pop()
    if turns:
        steps = interleave(turns[-1][1])
        while steps and steps[-1][0] == "result":
            steps.pop()
        turns[-1] = ("bot", steps)
        if not steps:
            turns.pop()
    if not turns or turns[-1][0] != "bot":
        return None
    return {"tools": ep.get("tools") or [], "schema": ep.get("schema"),
            "turns": turns}


def render_turns(turns):
    out = []
    for role, body in turns:
        if role == "user":
            out.append(f"<user>\n{body}\n</user>\n")
        else:
            out.append("<bot>\n" + "\n".join(_step(s) for s in interleave(body))
                       + "\n</bot>\n")
    return "".join(out)


def render(ep):
    """One record as the model reads it, or None if it has nothing to
    teach."""
    ep = tidy(ep)
    if ep is None:
        return None
    head = ""
    if ep["tools"]:
        head += "<tools>\n" + "\n".join(
            defuse(json.dumps(t, ensure_ascii=False)) for t in ep["tools"]) \
            + "\n</tools>\n"
    if ep.get("schema"):
        head += f"<schema>\n{defuse(ep['schema'])}\n</schema>\n"
    return head + render_turns(ep["turns"]) + EOT + "\n"


# ------------------------------------------------------- tool definitions
_TYPES = {"str": "string", "string": "string", "text": "string",
          "int": "integer", "integer": "integer", "float": "number",
          "number": "number", "double": "number", "bool": "boolean",
          "boolean": "boolean", "list": "array", "array": "array",
          "tuple": "array", "set": "array", "dict": "object",
          "object": "object", "null": "null", "none": "null"}


def _json_type(t):
    """(the JSON Schema spelling of a source's type, whether the source
    marked it optional). A spelling with no plain equivalent is kept."""
    if not isinstance(t, str):
        return t, False
    s, optional = t.strip(), False
    if s.lower().endswith(", optional"):
        s, optional = s[:-len(", optional")].strip(), True
    if s.lower().startswith("optional[") and s.endswith("]"):
        s, optional = s[len("optional["):-1].strip(), True
    return _TYPES.get(s.lower().split("[", 1)[0].strip(), s), optional


def tool_schema(t):
    """A tool definition in the one form <tools> lists: name, description,
    and parameters as a JSON Schema object. None if it has no name."""
    if isinstance(t, str):
        t = _loads(t)
    if not isinstance(t, dict):
        return None
    if isinstance(t.get("function"), dict):            # {"type": "function", ...}
        t = t["function"]
    name = t.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    params = t.get("parameters", t.get("input_schema")) or {}
    if isinstance(params, str):
        params = _loads(params) or {}
    if not isinstance(params, dict):
        params = {}
    flat = ("properties" not in params
            and params.get("type") not in ("object", "dict")
            and all(isinstance(v, dict) for v in params.values()))
    given = params if flat else (params.get("properties") or {})
    req = None if flat else params.get("required")
    if req is None:
        req = t.get("required")
    required = [r for r in req if isinstance(r, str)] if isinstance(req, list) \
        else []
    props = {}
    for k, p in (given.items() if isinstance(given, dict) else ()):
        p = p if isinstance(p, dict) else {"type": p}
        q = {a: b for a, b in p.items() if a not in ("title", "required")}
        if "type" in q:
            q["type"], optional = _json_type(q["type"])
        else:
            optional = False
        if isinstance(q.get("description"), str):
            q["description"] = " ".join(q["description"].split())
        if q.get("default") == "":
            del q["default"]
        if flat and p.get("required", not optional) is True \
                and k not in required:
            required.append(k)
        props[str(k)] = q
    out = {"name": name.strip(),
           "description": " ".join(str(t.get("description") or "").split()),
           "parameters": {"type": "object", "properties": props}}
    required = [r for r in required if r in props]
    if required:
        out["parameters"]["required"] = required
    return out


def _tools(items):
    out = [tool_schema(t) for t in (items if isinstance(items, list) else [])]
    return [t for t in out if t]


# ----------------------------------------------------------------- parsing
def _loads(s, default=None):
    """JSON, allowing what models writing it often leave in: a raw newline or
    tab inside a string. Written back out, it is escaped as JSON wants."""
    if not isinstance(s, str):
        return s if s is not None else default
    try:
        return json.loads(s, strict=False)
    except ValueError:
        return default


def _pyobj(s):
    """JSON, or failing that a Python literal (Toucan writes its calls as the
    repr of a dict)."""
    v = _loads(s)
    if v is not None or not isinstance(s, str):
        return v
    try:
        return ast.literal_eval(s.strip())
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None


def _args(a):
    """Call arguments as a dict, or None if they cannot be read - a record
    whose call cannot be read teaches nothing about calling."""
    if a is None or a == "":
        return {}
    if isinstance(a, str):
        a = _pyobj(a)
    return a if isinstance(a, dict) else None


def _result(v):
    if isinstance(v, dict) and set(v) >= {"name", "content"}:
        v = v["content"]
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def _calls_in(text, tag):
    """A reply's steps: the text around the tagged calls, and the calls. The
    tag may hold one JSON object or a list of them. None if a call cannot be
    read."""
    steps, at = [], 0
    for m in tag.finditer(text):
        before = text[at:m.start()].strip()
        if before:
            steps.append(("text", before))
        v = _pyobj(m.group(1).strip())
        for c in (v if isinstance(v, list) else [v]):
            if not isinstance(c, dict) or not isinstance(c.get("name"), str):
                return None
            a = _args(c.get("arguments", c.get("parameters")))
            if a is None:
                return None
            steps.append(("call", c["name"], a))
        at = m.end()
    rest = text[at:].strip()
    if rest:
        steps.append(("text", rest))
    return steps


def _json_values(text):
    """Every top-level JSON object or array in a text, in order: Glaive's
    system prompt lists its functions one after another."""
    dec, out, i = json.JSONDecoder(), [], 0
    while True:
        m = re.compile(r"[\[{]").search(text, i)
        if not m:
            return out
        try:
            v, i = dec.raw_decode(text, m.start())
            out.append(v)
        except ValueError:
            i = m.start() + 1


def _think(text):
    """A reply that opens with its reasoning between think markers, as steps."""
    m = re.match(r"\s*<think>(.*?)</think>\s*(.*)$", text, re.S)
    if not m:
        return [("text", text)]
    return [("think", m.group(1).strip())] + \
        ([("text", m.group(2).strip())] if m.group(2).strip() else [])


def _bot(turns, steps):
    if turns and turns[-1][0] == "bot":
        turns[-1][1].extend(steps)
    else:
        turns.append(("bot", list(steps)))


def _results(turns, texts):
    """Tool results join the bot turn that made the calls; None (drop the
    record) if no call came before them."""
    if not turns or turns[-1][0] != "bot":
        return None
    turns[-1][1].extend(("result", _result(t)) for t in texts)
    return True


# -------------------------------------------------------------- converters
def from_xlam(row):
    calls = _loads(row.get("answers"))
    if not isinstance(calls, list) or not calls:
        return None
    steps = []
    for c in calls:
        a = _args(c.get("arguments")) if isinstance(c, dict) else None
        if a is None or not isinstance(c.get("name"), str):
            return None
        steps.append(("call", c["name"], a))
    return {"tools": _tools(_loads(row.get("tools"), [])),
            "turns": [("user", row.get("query") or ""), ("bot", steps)]}


_GLAIVE = re.compile(r"(?:^|\n)\s*(USER|ASSISTANT|FUNCTION RESPONSE):[ \t]*")
_FUNCALL = re.compile(r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*'
                      r"(.*)\}\s*$", re.S)


def from_glaive(row):
    """USER: / ASSISTANT: / FUNCTION RESPONSE: in one string, the call as
    <functioncall> {"name": ..., "arguments": '<JSON in single quotes>'}."""
    tools = [v for v in _json_values(row.get("system") or "")
             if isinstance(v, dict) and "name" in v]
    parts = _GLAIVE.split(row.get("chat") or "")
    turns = []
    for who, text in zip(parts[1::2], parts[2::2]):
        text = text.replace(EOT, "").strip()
        if who == "USER":
            turns.append(("user", text))
        elif who == "FUNCTION RESPONSE":
            if _results(turns, [text]) is None:
                return None
        elif "<functioncall>" in text:
            before, call = text.split("<functioncall>", 1)
            m = _FUNCALL.match(call.strip())
            if not m:
                return None
            raw = m.group(2).strip()
            if raw[:1] == "'" and raw[-1:] == "'":
                raw = raw[1:-1]
            a = _args(raw)
            if a is None:
                return None
            _bot(turns, ([("text", before.strip())] if before.strip() else [])
                 + [("call", m.group(1), a)])
        else:
            _bot(turns, [("text", text)])
    return {"tools": _tools(tools), "turns": turns}


_TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
_TOOL_RESPONSE = re.compile(r"<tool_response>(.*?)</tool_response>", re.S)
_TOOLS_TAG = re.compile(r"<tools>(.*?)</tools>", re.S)


def from_hermes(row):
    """system / human / gpt / tool turns; calls in <tool_call>, results in
    <tool_response>; the json-mode files carry a schema instead of tools."""
    conv = row.get("conversations") or []
    tools = _loads(row.get("tools"))
    if not isinstance(tools, list):
        sys_text = next((t.get("value") or "" for t in conv
                         if t.get("from") == "system"), "")
        m = _TOOLS_TAG.search(sys_text)
        tools = _loads(m.group(1).strip(), []) if m else []
    turns = []
    for t in conv:
        who, val = t.get("from"), (t.get("value") or "").strip()
        if who == "system" or not val:
            continue
        if who == "human":
            turns.append(("user", val))
        elif who == "gpt":
            steps = _calls_in(val, _TOOL_CALL)
            if steps is None:
                return None
            _bot(turns, steps)
        elif who == "tool":
            got = [_loads(x.strip(), x.strip())
                   for x in _TOOL_RESPONSE.findall(val)] or [val]
            if _results(turns, got) is None:
                return None
    schema = row.get("schema")
    return {"tools": _tools(tools), "turns": turns,
            "schema": schema.strip() if isinstance(schema, str) else None}


def _pycalls(s):
    """ToolACE's calls, [Name With Spaces(a="x", b=[1, 2]), Other()], as
    [(name, arguments)]; None if the text is not calls."""
    s = s.strip()
    if not (s.startswith("[") and s.endswith("]") and "(" in s):
        return None
    body, out, i = s[1:-1], [], 0
    while i < len(body):
        while i < len(body) and body[i] in " ,\n":
            i += 1
        if i >= len(body):
            break
        j = body.find("(", i)
        if j < 0:
            return None
        name, depth, quote, k = body[i:j].strip(), 0, None, j
        while k < len(body):
            ch = body[k]
            if quote:
                if ch == "\\":
                    k += 1
                elif ch == quote:
                    quote = None
            elif ch in "\"'":
                quote = ch
            elif ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        if depth or not name:
            return None
        args = _kwargs(body[j + 1:k])
        if args is None:
            return None
        out.append((name, args))
        i = k + 1
    return out or None


_JSON_WORDS = {"true": True, "false": False, "null": None}


def _literal(n):
    if isinstance(n, ast.Name) and n.id in _JSON_WORDS:
        return _JSON_WORDS[n.id]
    if isinstance(n, (ast.List, ast.Tuple)):
        return [_literal(e) for e in n.elts]
    if isinstance(n, ast.Dict):
        return {_literal(k): _literal(v) for k, v in zip(n.keys, n.values)}
    return ast.literal_eval(n)


def _kwargs(s):
    if not s.strip():
        return {}
    try:
        node = ast.parse(f"f({s})", mode="eval").body
        if node.args or any(kw.arg is None for kw in node.keywords):
            return None
        return {kw.arg: _literal(kw.value) for kw in node.keywords}
    except (SyntaxError, ValueError, TypeError, RecursionError):
        return None


def from_toolace(row):
    """The functions as a JSON list inside the system prompt; the calls
    Python-style; the results a JSON list of {"name", "results"}."""
    system = row.get("system") or ""
    at = system.find("[", max(0, system.find("invoke")))
    tools = []
    if at >= 0:
        try:
            tools = json.JSONDecoder().raw_decode(system, at)[0]
        except ValueError:
            pass
    turns = []
    for t in row.get("conversations") or []:
        who, val = t.get("from"), (t.get("value") or "").strip()
        if not val:
            continue
        if who == "user":
            turns.append(("user", val))
        elif who == "assistant":
            calls = _pycalls(val)
            _bot(turns, [("call", n, a) for n, a in calls] if calls
                 else [("text", val)])
        elif who == "tool":
            v = _loads(val, val)
            items = v if isinstance(v, list) else [v]
            if _results(turns, [x.get("results", x) if isinstance(x, dict)
                                else x for x in items]) is None:
                return None
    return {"tools": _tools(tools), "turns": turns}


_TOOLCALL = re.compile(r"<TOOLCALL>(.*?)</TOOLCALL>", re.S)


def from_when2call(row):
    """Tools as a list of JSON strings, OpenAI-style messages, calls in
    <TOOLCALL>. The preference set adds the chosen reply; the rejected one
    is never seen."""
    msgs = list(row.get("messages") or [])
    if isinstance(row.get("chosen_response"), dict):
        msgs.append(row["chosen_response"])
    turns = []
    for m in msgs:
        role, content = m.get("role"), (m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            turns.append(("user", content))
        elif role == "assistant":
            steps = _calls_in(content, _TOOLCALL)
            if steps is None:
                return None
            _bot(turns, steps)
        elif role == "tool":
            if _results(turns, [content]) is None:
                return None
    return {"tools": _tools([_loads(t) for t in row.get("tools") or []]),
            "turns": turns}


def from_smoltalk(row):
    """Messages; the tools inside <tools> in the system message, the calls a
    JSON list inside <tool_call>."""
    tools, turns = [], []
    for m in row.get("messages") or []:
        role, content = m.get("role"), (m.get("content") or "").strip()
        if role == "system":
            t = _TOOLS_TAG.search(content)
            tools = _loads(t.group(1).strip(), []) if t else []
        elif not content:
            continue
        elif role == "user":
            turns.append(("user", content))
        elif role == "assistant":
            steps = _calls_in(content, _TOOL_CALL)
            if steps is None:
                return None
            _bot(turns, steps)
        elif role == "tool":
            if _results(turns, [content]) is None:
                return None
    return {"tools": _tools(tools), "turns": turns}


def _call(c):
    """A call written as a dict, its JSON or its Python repr - with the
    arguments as a dict or as a JSON string - as a step; None if unreadable."""
    c = _pyobj(c) if isinstance(c, str) else c
    if isinstance(c, dict) and isinstance(c.get("function"), dict):
        c = c["function"]
    a = _args(c.get("arguments")) if isinstance(c, dict) else None
    if a is None or not isinstance(c.get("name"), str):
        return None
    return ("call", c["name"], a)


def from_chat(msgs, tools, keep_system=False):
    """OpenAI-style messages as an episode. A call is an assistant message's
    function_call or tool_calls, or a message of its own (role tool_call, as
    Toucan's SFT set writes it, a dict's repr); a result is a tool, function
    or tool_response message; reasoning_content is the reply's think step.
    keep_system heads the first request with the system message, where a
    source explains the task there (Nemotron's ARC puzzles)."""
    msgs = _loads(msgs)
    if not isinstance(msgs, list):
        return None
    turns, system = [], ""
    for m in msgs:
        if not isinstance(m, dict):
            return None
        role, content = m.get("role"), m.get("content")
        if isinstance(content, list):                # content parts
            content = "".join(c.get("text", "") for c in content
                              if isinstance(c, dict))
        content = content.strip() if isinstance(content, str) else ""
        if role == "system":
            system = content if keep_system else ""
        elif role == "user":
            if system:
                content, system = f"{system}\n\n{content}", ""
            turns.append(("user", content))
        elif role == "assistant":
            steps = []
            why = m.get("reasoning_content")
            if isinstance(why, str) and why.strip():
                steps.append(("think", why.strip()))
            if content:
                steps += _think(content)
            listed = m.get("tool_calls") or []
            if isinstance(listed, str):                 # a list's repr
                listed = _pyobj(listed) or []
            for c in ([m["function_call"]] if m.get("function_call") else []) \
                    + list(listed):
                step = _call(c)
                if step is None:
                    return None
                steps.append(step)
            if steps:
                _bot(turns, steps)
        elif role == "tool_call":
            step = _call(content)
            if step is None:
                return None
            _bot(turns, [step])
        elif role in ("tool_response", "tool", "function"):
            if _results(turns, [content]) is None:
                return None
    return {"tools": _tools(_loads(tools, []) or []), "turns": turns}


def from_toucan(row):
    """Tools and messages as JSON strings. The SFT set gives each call a
    message of its own and each result one; the other three put the call on
    an assistant message as function_call, the result in a `function`
    message, and gpt-oss's reasoning in reasoning_content."""
    return from_chat(row.get("messages"),
                     row.get("tools") or row.get("available_tools"))


def from_agentinstruct(row):
    """The environment speaks as the user and the agent answers in its own
    Thought/Action words: the six environments each have their own action
    syntax, and this keeps them as written rather than guessing a mapping."""
    turns = []
    for t in row.get("conversations") or []:
        who, val = t.get("from"), (t.get("value") or "").strip()
        if who == "human":
            turns.append(("user", val))
        elif who == "gpt":
            _bot(turns, [("text", val)])
    return {"tools": [], "turns": turns}


_PLATFORM = {"common": "", "linux": " on Linux", "osx": " on macOS",
             "windows": " on Windows", "android": " on Android",
             "freebsd": " on FreeBSD", "netbsd": " on NetBSD",
             "openbsd": " on OpenBSD", "sunos": " on SunOS"}


def _tldr_words(s):
    s = re.sub(r"\[([^\[\]\s]{1,12})\]", r"\1", s)     # [c]reate -> create
    return s.strip().rstrip(":").strip()


_LANGUAGE = {"ar": "Arabic", "bn": "Bengali", "bs": "Bosnian", "ca": "Catalan",
             "cs": "Czech", "da": "Danish", "de": "German", "el": "Greek",
             "es": "Spanish", "fa": "Persian", "fi": "Finnish", "fr": "French",
             "hi": "Hindi", "hu": "Hungarian", "id": "Indonesian",
             "it": "Italian", "ja": "Japanese", "ko": "Korean", "lo": "Lao",
             "ml": "Malayalam", "ne": "Nepali", "nl": "Dutch",
             "no": "Norwegian", "pl": "Polish", "pt_BR": "Brazilian Portuguese",
             "pt_PT": "European Portuguese", "ro": "Romanian", "ru": "Russian",
             "sh": "Serbo-Croatian", "sr": "Serbian", "sv": "Swedish",
             "ta": "Tamil", "th": "Thai", "tr": "Turkish", "uk": "Ukrainian",
             "uz": "Uzbek", "zh": "Chinese", "zh_TW": "Traditional Chinese"}


def from_tldr(page, platform, language="en"):
    """A tldr page as a conversation: in English, what the command is for and
    then how to do each of its examples; a translation is asked for whole, in
    its language, so the English question never answers itself in another."""
    title = next((l[2:].strip() for l in page.splitlines()
                  if l.startswith("# ")), None)
    about = [_tldr_words(l[1:].strip()) for l in page.splitlines()
             if l.startswith(">")]
    ex, ask = [], None
    for line in page.splitlines():
        if line.startswith("- "):
            ask = _tldr_words(line[2:])
        elif line.startswith("`") and line.endswith("`") and ask:
            ex.append((ask, re.sub(r"\{\{(.*?)\}\}", r"\1", line[1:-1])))
            ask = None
    if not title or not ex:
        return None
    where = _PLATFORM.get(platform, f" on {platform}")
    if language != "en":
        body = "\n".join(about) + "\n\n" + "\n\n".join(
            f"- {q}:\n`{cmd}`" for q, cmd in ex)
        return {"tools": [], "turns": [
            ("user", f"Show me the tldr page for `{title}`{where}, in "
                     f"{_LANGUAGE.get(language, language)}."),
            ("bot", [("text", f"# {title}\n\n{body.strip()}")])]}
    turns = [("user", f"What is `{title}` for{where}?"),
             ("bot", [("text", "\n".join(about) or f"`{title}` is a command.")])]
    for ask, cmd in ex:
        first, _, rest = ask.partition(" ")
        if not (len(first) > 1 and first.isupper()):
            first = first[:1].lower() + first[1:]
        turns += [("user", f"With `{title}`, how do I {first} {rest}".rstrip()
                   + "?"),
                  ("bot", [("text", f"`{cmd}`")])]
    return {"tools": [], "turns": turns}


# ----------------------------------------------------------------- sources
# In order: a record kept by an earlier source is a duplicate in a later one.
#   mb       what the source keeps under data/raw: its downloads (from each
#            repository's listing; tldr's is a guess, since GitHub builds
#            that archive when asked), unpacked where they are unpacked
#   text_mb  the text it writes, estimated from rows sampled from it; None
#            where the size is set by an argument
#   ram_mb   the most memory it takes to build
SOURCES = {
    "toolbox": dict(lane="tools", make="toolbox", mb=0, text_mb=None,
                    ram_mb=600, licence="generated here", dedup=False),
    "xlam": dict(lane="function-calls", mb=96, text_mb=90, ram_mb=1500,
                 gated=True, repo="Salesforce/xlam-function-calling-60k",
                 files=["xlam_function_calling_60k.json"],
                 convert=from_xlam, licence="CC BY 4.0"),
    "toolace": dict(lane="function-calls", mb=37, text_mb=40, ram_mb=500,
                    repo="Team-ACE/ToolACE", files=["data.json"],
                    convert=from_toolace, licence="Apache-2.0"),
    "hermes": dict(lane="function-calls", mb=67, text_mb=35, ram_mb=500,
                   repo="NousResearch/hermes-function-calling-v1",
                   files=["func-calling-singleturn.json", "func-calling.json",
                          "glaive-function-calling-5k.json",
                          "json-mode-agentic.json",
                          "json-mode-singleturn.json"],
                   convert=from_hermes, licence="Apache-2.0"),
    "when2call": dict(lane="function-calls", mb=63, text_mb=35, ram_mb=300,
                      repo="nvidia/When2Call",
                      files=["train/when2call_train_sft.jsonl",
                             "train/when2call_train_pref.jsonl"],
                      keep=["test/*.jsonl"], convert=from_when2call,
                      licence="CC BY 4.0"),
    "smoltalk": dict(lane="function-calls", mb=52, text_mb=125, ram_mb=400,
                     repo="HuggingFaceTB/smoltalk",
                     files=["data/apigen-80k/train-*.parquet"],
                     held_out=["data/apigen-80k/test-*.parquet"],
                     convert=from_smoltalk,
                     licence="Apache-2.0 (from xLAM: CC BY 4.0)"),
    "glaive": dict(lane="function-calls", mb=271, text_mb=240, ram_mb=3000,
                   repo="glaiveai/glaive-function-calling-v2",
                   files=["glaive-function-calling-v2.json"],
                   convert=from_glaive, licence="Apache-2.0"),
    "tldr": dict(lane="function-calls", make="tldr", mb=30, text_mb=60,
                 ram_mb=300,
                 url="https://github.com/tldr-pages/tldr/archive/refs/heads/"
                     "main.zip",
                 licence="CC BY 4.0"),
    # all four of Toucan's sets: SFT first, the subset its authors curated,
    # so its copies in the other three are the ones dropped as duplicates
    "toucan": dict(lane="trajectories", mb=21_800, text_mb=33_000, ram_mb=2500,
                   repo="Agent-Ark/Toucan-1.5M",
                   files=["SFT/train-*.parquet", "Kimi-K2/train-*.parquet",
                          "OSS/train-*.parquet", "Qwen3/train-*.parquet"],
                   convert=from_toucan, licence="Apache-2.0"),
    "textworld": dict(lane="trajectories", make="textworld", mb=0,
                      text_mb=None, ram_mb=None, needs="textworld",
                      dedup=False, licence="MIT; the games are made here"),
    "alfworld": dict(lane="trajectories", make="alfworld", mb=1300,
                     text_mb=15, ram_mb=1500, needs="alfworld", dedup=False,
                     licence="MIT"),
    "agentinstruct": dict(lane="trajectories", mb=1, text_mb=8, ram_mb=200,
                          default=False, repo="zai-org/AgentInstruct",
                          files=["data/*.parquet"],
                          convert=from_agentinstruct, licence="none stated"),
    # ARC (corpora/arc.py). ARC-AGI-2 before ARC-AGI-1: 391 of ARC-AGI-1's
    # 400 training tasks are in ARC-AGI-2's training set as well, and the
    # copy kept is the first.
    "arc2": dict(lane="arc", make="arc", mb=10, text_mb=300, ram_mb=800,
                 url="https://github.com/arcprize/ARC-AGI-2/archive/refs/heads/"
                     "main.zip",
                 licence="Apache-2.0"),
    "arc1": dict(lane="arc", make="arc", mb=5, text_mb=110, ram_mb=800,
                 url="https://github.com/fchollet/ARC-AGI/archive/refs/heads/"
                     "master.zip",
                 licence="Apache-2.0"),
    "arc-community": dict(lane="arc", make="arc", mb=1000, text_mb=500,
                          ram_mb=2000,
                          url="https://github.com/neoneye/arc-dataset-"
                              "collection/archive/refs/heads/main.zip",
                          licence="MIT, Apache-2.0, DbCL (per dataset)"),
    "arc-community-unlicensed": dict(
        lane="arc", make="arc", mb=0, text_mb=100, ram_mb=2000, default=False,
        url="https://github.com/neoneye/arc-dataset-collection/archive/refs/"
            "heads/main.zip",
        licence="none stated"),
    "rearc": dict(lane="arc", make="arc", mb=51, text_mb=600, ram_mb=1000,
                  url="https://github.com/michaelhodel/re-arc/raw/main/"
                      "re_arc.zip",
                  licence="MIT"),
    "barc": dict(lane="arc-synthetic", make="arc", mb=11_900, text_mb=6000,
                 ram_mb=1500, licence="MIT",
                 repos=[("barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_"
                         "generated_problems", ["*.jsonl"]),
                        ("barc0/100k-gpt4-description-gpt4omini-code_"
                         "generated_problems", ["*.jsonl"]),
                        ("barc0/100k-gpt4omini-description-gpt4omini-code_"
                         "generated_problems", ["*.jsonl"])]),
    "nemotron-rl": dict(lane="arc-synthetic", make="arc", mb=465, text_mb=160,
                        ram_mb=600, repo="nvidia/Nemotron-RL-ARC-AGI-v1",
                        files=["data/*/train.jsonl", "data/*/validation.jsonl"],
                        licence="CC BY 4.0, Apache-2.0, MIT"),
    "nvarc": dict(lane="arc-synthetic", make="arc", mb=17_760, text_mb=9000,
                  ram_mb=4000, default=False, needs="kaggle",
                  kaggle=["sorokin/nvarc-synthetic-puzzles",
                          "sorokin/nvarc-augmented-puzzles"],
                  licence="none stated"),
    "nemotron-sft": dict(lane="arc-traces", make="arc", mb=18_432,
                         text_mb=16_000, ram_mb=2000,
                         repo="nvidia/Nemotron-SFT-ARC-AGI-v1",
                         files=["data/*/data.jsonl"],
                         licence="CC BY 4.0; research and development"),
    "nvarc-artifacts": dict(lane="arc-traces", make="arc", mb=47_514,
                            text_mb=40_000, ram_mb=3000, default=False,
                            needs="kaggle",
                            kaggle=["sorokin/nvarc-artifacts-puzzles"],
                            licence="none stated"),
    "arc3": dict(lane="arc-games", make="arc3", mb=400, text_mb=None,
                 ram_mb=2000, dedup=False,
                 licence="MIT toolkit; Apache-2.0 games (Kaggle)"),
}
LANES = ["tools", "function-calls", "trajectories", "arc", "arc-synthetic",
         "arc-traces", "arc-games"]


# ------------------------------------------------------------------ reading
def records(path):
    """Every record of a downloaded file: a JSON array, JSON lines, or
    Parquet read a batch at a time."""
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
            yield from batch.to_pylist()
        return
    with open(path, encoding="utf-8") as f:
        head = f.read(1)
        while head and head.isspace():
            head = f.read(1)
        f.seek(0)
        if head == "[":
            yield from json.load(f)
            return
        for line in f:
            if line.strip():
                yield json.loads(line)


def fetch_hf(name, repo, raw, patterns):
    """The files a source needs, from a Hugging Face dataset repository, into
    data/raw/agent/<name>/<repo's name>. Files already there are kept."""
    from huggingface_hub import snapshot_download
    where = os.path.join(raw, name, repo.split("/")[-1])
    print(f"  fetching {repo} ({', '.join(patterns)})", flush=True)
    snapshot_download(repo_id=repo, repo_type="dataset",
                      allow_patterns=patterns, local_dir=where)
    return where


def fetch_url(url, path):
    """One file, fetched once: half a download is kept under .part and never
    mistaken for the whole."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"  fetching {url}", flush=True)
        with urlopen(url, timeout=120) as r, open(path + ".part", "wb") as f:
            shutil.copyfileobj(r, f)
        os.replace(path + ".part", path)
    return path


def fetch_kaggle(slug, where):
    """A Kaggle dataset, unpacked into `where`, once."""
    done = os.path.join(where, ".fetched")
    if not os.path.exists(done):
        os.makedirs(where, exist_ok=True)
        print(f"  fetching kaggle:{slug}", flush=True)
        subprocess.run(["kaggle", "datasets", "download", "-d", slug, "-p",
                        where, "--unzip"], check=True)
        open(done, "w").close()
    return where


def _files(where, patterns):
    out = []
    for p in patterns:
        out.extend(sorted(glob.glob(os.path.join(where, p))))
    return out


def from_hub(name, spec, a):
    where = fetch_hf(name, spec["repo"], a.raw, spec["files"]
                     + spec.get("held_out", []) + spec.get("keep", []))
    for val, pats in ((None, spec["files"]), (True, spec.get("held_out", []))):
        for path in _files(where, pats):
            print(f"  reading {os.path.relpath(path, where)}", flush=True)
            for row in records(path):
                yield spec["convert"](row), val


def from_tldr_zip(name, spec, a):
    """Every page, in every language: pages/ is English, pages.<code>/ the
    translations."""
    path = fetch_url(spec["url"], os.path.join(a.raw, name, "tldr-main.zip"))
    with zipfile.ZipFile(path) as z:
        for info in sorted(z.infolist(), key=lambda i: i.filename):
            parts = info.filename.split("/")
            if len(parts) == 4 and parts[1].startswith("pages") \
                    and parts[3].endswith(".md"):
                language = parts[1][len("pages."):] or "en"
                page = z.read(info).decode("utf-8", "replace")
                yield from_tldr(page, parts[2], language), None


def from_toolbox(name, spec, a):
    """Training and held-out episodes from separate seeds, the held-out ones
    asking about pages training never looks up."""
    from corpora.toolbox import episodes
    wiki = os.path.join(a.out, "train", "wikipedia")
    for seed, val, chars in ((0, False, a.tools_chars),
                             (1, True, a.tools_chars * a.val_per_mille // 1000)):
        got, made = 0, 0
        for ep in episodes(seed, wiki, held_out=val,
                           pages=30_000 if not val else 3_000):
            if got >= chars or (a.limit and made >= (a.limit if not val else
                                                     max(1, a.limit // 100))):
                break
            text = render(ep)
            got += len(text or "")
            made += 1
            yield ep, val


def from_worlds(name, spec, a):
    from corpora import worlds
    if name == "textworld":
        yield from worlds.textworld(a.textworld_games if not a.limit
                                    else a.limit, a.workers,
                                    os.path.join(a.raw, name))
    else:
        yield from worlds.alfworld(os.path.join(a.raw, name),
                                   limit=a.limit)


_GUARD = {}


def guard(a):
    """The public evaluation tasks of ARC-AGI-1 and -2, copied as they are to
    data/eval/arc-agi-1 and data/eval/arc-agi-2 for the benchmark, and the
    guard that keeps them out of training - fetched and built once a run,
    whichever ARC sources are being built."""
    if "g" not in _GUARD:
        from corpora import arc
        zips = {}
        for src, ev in (("arc1", "arc-agi-1"), ("arc2", "arc-agi-2")):
            zips[ev] = fetch_url(SOURCES[src]["url"],
                                 os.path.join(a.raw, "arc", f"{src}.zip"))
            d = os.path.join(a.out, "eval", ev)
            os.makedirs(d, exist_ok=True)
            with zipfile.ZipFile(zips[ev]) as z:
                for n in z.namelist():
                    if "/data/evaluation/" in n and n.endswith(".json"):
                        with open(os.path.join(d, os.path.basename(n)), "wb") as f:
                            f.write(z.read(n))
        sets = arc.evaluation_sets(zips["arc-agi-1"], zips["arc-agi-2"])
        kept_out = dict(sets["arc-agi-2"])
        if not a.arc1_eval_for_training:
            kept_out.update(sets["arc-agi-1"])
        _GUARD["g"] = arc.Guard(kept_out)
        print(f"  guard: {len(_GUARD['g'].ids)} evaluation tasks, "
              f"{len(_GUARD['g'].sigs):,} pairs kept out of training",
              flush=True)
    return _GUARD["g"]


def _tasks(arc, items, g, copies, per_mille):
    """Records for (name, task or pool) items; a pool of pairs from one rule
    is cut into tasks first."""
    for n, t in items:
        if isinstance(t, list):
            for i, c in enumerate(arc.chunks(t, random.Random(n))):
                yield from arc.records(c, f"{n}/{i}", g, copies, per_mille,
                                       side=n)
        else:
            yield from arc.records(t, n, g, copies, per_mille)


def from_arc(name, spec, a):
    """The ARC sources: (episode, held out?, identity), or a skip with its
    reason - "evaluation overlap" for a task the guard refused."""
    from corpora import arc
    g, per = guard(a), a.val_per_mille
    if name in ("arc1", "arc2"):
        z = fetch_url(spec["url"], os.path.join(a.raw, "arc", f"{name}.zip"))
        splits = ["training"] + (["evaluation"] if name == "arc1"
                                 and a.arc1_eval_for_training else [])
        for split in splits:
            for tid, t in arc.split_tasks(z, split):
                yield from arc.records(t, f"{name}/{tid}", g, a.arc_augment,
                                       per)
    elif name in ("arc-community", "arc-community-unlicensed"):
        z = fetch_url(spec["url"], os.path.join(a.raw, "arc", "neoneye.zip"))
        which = arc.COMMUNITY if name == "arc-community" \
            else arc.COMMUNITY_UNLICENSED
        for n, t, copies in arc.community(z, which, skip=g.named):
            copies = a.arc_augment if copies == 64 else copies
            yield from _tasks(arc, [(n, t)], g, copies, per)
    elif name == "rearc":
        z = fetch_url(spec["url"], os.path.join(a.raw, "arc", "re_arc.zip"))
        yield from _tasks(arc, arc.walk(z), g, 1, per)
    elif name == "barc":
        for repo, pats in spec["repos"]:
            where = fetch_hf(name, repo, a.raw, pats)
            for path in _files(where, pats):
                print(f"  reading {os.path.relpath(path, a.raw)}", flush=True)
                for i, row in enumerate(records(path)):
                    ex = arc.pairs(row.get("examples"))
                    if not ex or len(ex) < 3:
                        yield "unusable", None, None
                        continue
                    nm = f"barc/{os.path.basename(path)}/{i}"
                    src = row.get("source")
                    for j, t in enumerate(arc.chunks(ex, random.Random(nm))):
                        yield from arc.records(
                            t, nm, g, 1, per, side=nm,
                            programs=(ex[:5], src) if j == 0 and
                            isinstance(src, str) else None)
    elif name == "nemotron-rl":
        where = fetch_hf(name, spec["repo"], a.raw, spec["files"])
        for path in _files(where, spec["files"]):
            print(f"  reading {os.path.relpath(path, a.raw)}", flush=True)
            held = True if path.endswith("validation.jsonl") else None
            for row in records(path):
                t = arc.task({"train": _loads(row.get("train")),
                              "test": [{"input": _loads(row.get("test_input")),
                                        "output": _loads(row.get(
                                            "expected_output"))}]})
                if not t:
                    yield "unusable", None, None
                    continue
                nm = f"{row.get('problem_id', '')}/{row.get('task_id', '')}"
                for ep, val, ident in arc.records(t, nm, g, 1, per):
                    yield ep, (held if held and val is not None else val), ident
    elif name == "nemotron-sft":
        where = fetch_hf(name, spec["repo"], a.raw, spec["files"])
        for path in _files(where, spec["files"]):
            print(f"  reading {os.path.relpath(path, a.raw)}", flush=True)
            for row in records(path):
                msgs = row.get("messages")
                asked = " ".join(m.get("content") or "" for m in msgs
                                 if isinstance(m, dict) and m.get("role")
                                 in ("system", "user")) \
                    if isinstance(msgs, list) else ""
                if g.bad({"train": arc.text_pairs(asked), "test": []},
                         str(row.get("problem_id", ""))):
                    yield "evaluation overlap", None
                    continue
                yield from_chat(msgs, row.get("tools"), keep_system=True), None
    elif name == "nvarc":
        for slug in spec["kaggle"]:
            where = fetch_kaggle(slug, os.path.join(a.raw, name,
                                                    slug.split("/")[-1]))
            yield from _tasks(arc, arc.walk(where, skip=g.named), g, 1, per)
    elif name == "nvarc-artifacts":
        for slug in spec["kaggle"]:
            where = fetch_kaggle(slug, os.path.join(a.raw, name,
                                                    slug.split("/")[-1]))
            for path in sorted(glob.glob(os.path.join(where, "**", "*.parquet"),
                                         recursive=True)):
                for row in records(path):
                    ask, said = row.get("prompt"), row.get("completion")
                    why = row.get("reasoning")
                    if not isinstance(ask, str) or not isinstance(said, str):
                        yield "unusable", None
                        continue
                    steps = ([("think", why)] if isinstance(why, str)
                             and why.strip() else []) + [("text", said)]
                    yield {"tools": [], "turns": [("user", ask),
                                                  ("bot", steps)]}, None


def arc3_python(a):
    """A Python 3.12 with the ARC-AGI toolkit: the one given, or one made
    under data/raw/agent/arc3 with uv the first time."""
    if a.arc3_python:
        return a.arc3_python
    venv = os.path.join(os.path.abspath(a.raw), "arc3", "venv")
    py = os.path.join(venv, "bin", "python")
    if not os.path.exists(py):
        uv = shutil.which("uv")
        if not uv:
            raise RuntimeError("ARC-AGI-3 needs Python 3.12 with arc-agi: "
                               "install uv, or pass --arc3-python")
        print(f"  making a Python 3.12 with arc-agi in {venv}", flush=True)
        subprocess.run([uv, "venv", "--python", "3.12", venv], check=True)
        subprocess.run([uv, "pip", "install", "--python", py, "arc-agi"],
                       check=True)
    return py


def from_arc3(name, spec, a):
    """ARC-AGI-3's public games, played by corpora/arc3_play.py under its own
    Python. The games it holds out are listed in data/eval/arc-agi-3."""
    from corpora import arc
    where = os.path.abspath(os.path.join(a.raw, "arc3"))
    out, held = os.path.join(where, "episodes.jsonl"), \
        os.path.join(where, "held_out.json")
    if a.force or not os.path.exists(out + ".done"):
        os.makedirs(where, exist_ok=True)
        subprocess.run([arc3_python(a), os.path.join(ROOT, "corpora",
                                                     "arc3_play.py"),
                        "--out", out, "--held", held,
                        "--environments", os.path.join(where,
                                                       "environment_files"),
                        "--steps", str(a.arc3_steps),
                        "--hold-every", str(a.arc3_hold_every)],
                       check=True, cwd=where)
        open(out + ".done", "w").close()
    d = os.path.join(a.out, "eval", "arc-agi-3")
    os.makedirs(d, exist_ok=True)
    shutil.copy(held, os.path.join(d, "held_out_games.json"))
    with open(out) as f:
        for line in f:
            ep = arc.game_record(json.loads(line))
            yield (ep, None) if ep else ("unusable", None)


MAKERS = {"toolbox": from_toolbox, "tldr": from_tldr_zip,
          "textworld": from_worlds, "alfworld": from_worlds,
          "arc3": from_arc3}


def maker(name, spec):
    return MAKERS.get(name) or (from_arc if spec.get("make") == "arc"
                                else from_hub)


# ------------------------------------------------------------------ writing
class Shelf:
    """Records grouped into files of about `group` characters, a record never
    split - a window must not span a seam that means nothing (fetch.py). The
    training files go in shards of `shard`; the held-out ones side by side,
    named for their source, so the held-out subject's blocks spread over
    every source in proportion."""

    def __init__(self, train_dir, val_dir, name, group=16_000, shard=2000):
        self.dirs, self.name = (train_dir, val_dir), name
        self.group, self.shard = group, shard
        self.buf = ([], [])
        self.size = [0, 0]
        self.files = [0, 0]
        self.chars = [0, 0]
        self.records = [0, 0]

    def add(self, text, val):
        i = int(bool(val))
        self.buf[i].append(text)
        self.size[i] += len(text)
        self.records[i] += 1
        if self.size[i] >= self.group:
            self._flush(i)

    def _flush(self, i):
        if not self.buf[i]:
            return
        n = self.files[i]
        if i == 0:
            d = os.path.join(self.dirs[0], f"{n // self.shard:04d}")
            path = os.path.join(d, f"part-{n:06d}.txt")
        else:
            d = self.dirs[1]
            path = os.path.join(d, f"{self.name}-{n:05d}.txt")
        os.makedirs(d, exist_ok=True)
        text = "".join(self.buf[i])
        with open(path, "w", encoding="utf-8", errors="replace") as f:
            f.write(text)
        self.chars[i] += len(text)
        self.files[i] += 1
        self.buf[i].clear()
        self.size[i] = 0

    def close(self):
        self._flush(0)
        self._flush(1)


def _norm(s):
    return " ".join(s.lower().split())


def keys(ep):
    """(the record's identity for finding duplicates, the number that decides
    its side). Identity is the conversation as rendered, tools left out: the
    same exchange from two sources is the same record however each wrote its
    tools. The side follows the first request when it is long enough to be
    distinctive, so a reworded copy of it cannot sit on the other side;
    a short one ("Hi") says nothing, and the whole record decides."""
    first = next((t[1] for t in ep["turns"] if t[0] == "user"), "")
    ident = hashlib.blake2b(_norm(render_turns(ep["turns"])).encode(),
                            digest_size=8).digest()
    side = _norm(first) if len(_norm(first)) >= 40 else ident
    if isinstance(side, str):
        side = hashlib.blake2b(side.encode(), digest_size=8).digest()
    return ident, int.from_bytes(side, "big")


def _built(raw, name):
    return os.path.join(raw, "built", f"{name}.json")


def _keys_file(raw, name):
    return os.path.join(raw, "built", f"{name}.keys")


def _load_keys(raw, name, seen):
    """What an earlier build of `name` kept, so later sources still drop
    their copies of it."""
    if os.path.exists(_built(raw, name)) and \
            os.path.exists(_keys_file(raw, name)):
        with open(_keys_file(raw, name), "rb") as f:
            blob = f.read()
        seen.update(blob[i:i + 8] for i in range(0, len(blob), 8))


def build(name, spec, a, seen):
    """One source, into data/train/<lane>/<name> and data/val/<lane>. Returns
    what was written, or None if it was already there.

    A maker yields (episode, held out?) or (episode, held out?, identity):
    None for the side or the identity leaves it to keys(), and a string in
    place of an episode is a record skipped, for that reason."""
    lane = spec["lane"]
    train_dir = os.path.join(a.out, "train", lane, name)
    val_dir = os.path.join(a.out, "val", lane)
    state = _built(a.raw, name)
    if os.path.exists(state) and os.path.isdir(train_dir) and not a.force:
        _load_keys(a.raw, name, seen)
        print(f"== {name}: already in {train_dir} - skipping (--force to "
              "rebuild)", flush=True)
        return None
    print(f"== {name} -> {train_dir}", flush=True)
    # written beside the downloads and moved in when complete: a session
    # starting mid-build sees neither half a source nor a held-out subject
    # with no files in it yet
    partial = os.path.join(a.raw, "partial", name)
    shutil.rmtree(partial, ignore_errors=True)
    tmp_train, tmp_val = os.path.join(partial, "train"), os.path.join(partial, "val")
    os.makedirs(tmp_train)
    os.makedirs(tmp_val)
    shelf = Shelf(tmp_train, tmp_val, name)
    mine, made, dup, bad, skipped, t0 = [], 0, 0, 0, {}, time.time()
    for got in maker(name, spec)(name, spec, a):
        ep, val = got[0], got[1]
        ident = got[2] if len(got) > 2 else None
        made += 1
        if isinstance(ep, str):
            skipped[ep] = skipped.get(ep, 0) + 1
            continue
        ep = tidy(ep)
        if ep is None:
            bad += 1
            continue
        if ident is None or val is None:
            ident2, side = keys(ep)
            ident = ident or ident2
            if val is None:
                val = side % 1000 < a.val_per_mille
        if spec.get("dedup", True):
            if ident in seen:
                dup += 1
                continue
            seen.add(ident)
            mine.append(ident)
        shelf.add(render(ep), val)
        kept = shelf.records[0] + shelf.records[1]
        if kept % 20_000 == 0:
            print(f"    {kept:,} records, "
                  f"{(shelf.chars[0] + shelf.chars[1]) / 1e6:,.0f}M characters "
                  f"({time.time() - t0:,.0f} s)", flush=True)
        if a.limit and name not in ("toolbox",) and kept >= a.limit:
            break
    shelf.close()
    if not shelf.records[0]:
        shutil.rmtree(partial, ignore_errors=True)
        print(f"!! {name}: nothing usable ({made:,} read, {bad:,} unusable"
              + "".join(f", {v:,} {k}" for k, v in skipped.items()) + ")",
              file=sys.stderr, flush=True)
        return {"failed": True}
    # into place: the old copy out, the new one in, the held-out files beside
    # the other sources'
    shutil.rmtree(train_dir, ignore_errors=True)
    for f in glob.glob(os.path.join(val_dir, f"{name}-*.txt")):
        os.remove(f)
    os.makedirs(os.path.dirname(train_dir), exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    shutil.move(tmp_train, train_dir)
    for f in sorted(os.listdir(tmp_val)):
        shutil.move(os.path.join(tmp_val, f), os.path.join(val_dir, f))
    shutil.rmtree(partial, ignore_errors=True)
    out = {"lane": lane, "licence": spec["licence"], "read": made,
           "unusable": bad, "duplicates": dup, "skipped": skipped,
           "train_records": shelf.records[0], "val_records": shelf.records[1],
           "train_chars": shelf.chars[0], "val_chars": shelf.chars[1],
           "built": time.strftime("%Y-%m-%d %H:%M"),
           "from": spec.get("repo") or spec.get("url") or spec.get("kaggle")
           or [r for r, _ in spec.get("repos", [])] or None}
    os.makedirs(os.path.dirname(state), exist_ok=True)
    with open(_keys_file(a.raw, name), "wb") as f:
        f.write(b"".join(mine))
    with open(state, "w") as f:
        json.dump(out, f, indent=1)
    print(f"  {name}: {shelf.records[0]:,} records to train "
          f"({shelf.chars[0] / 1e6:,.1f}M characters), "
          f"{shelf.records[1]:,} held out; {dup:,} duplicates, "
          f"{bad:,} unusable"
          + "".join(f", {v:,} {k}" for k, v in skipped.items())
          + f", of {made:,} ({time.time() - t0:,.0f} s)", flush=True)
    return out


# --------------------------------------------------------------- the plan
def _missing(spec, a):
    """What has to be installed or set up first, or None."""
    need = spec.get("needs")
    if need == "kaggle":
        if not shutil.which("kaggle"):
            return "pip install kaggle, and an API token in ~/.kaggle/kaggle.json"
        if not (os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json"))
                or "KAGGLE_KEY" in os.environ):
            return "a Kaggle API token in ~/.kaggle/kaggle.json"
    elif need:
        try:
            __import__(need)
        except ImportError:
            return f"pip install {need}"
    if spec.get("repo") or spec.get("repos"):
        try:
            import huggingface_hub                    # noqa: F401
            import pyarrow                            # noqa: F401
        except ImportError:
            return "pip install huggingface_hub pyarrow"
    if spec.get("make") == "arc3" and not a.arc3_python and \
            not shutil.which("uv") and not os.path.exists(
                os.path.join(a.raw, "arc3", "venv", "bin", "python")):
        return "uv (it makes a Python 3.12 for arc-agi), or --arc3-python"
    return None


def plan(a):
    """The sources this run would build, in order."""
    want = a.only or [n for n, s in SOURCES.items() if s.get("default", True)]
    want = [n for n in want if n not in (a.skip or [])]
    want += [n for n in (a.add or []) if n not in want]
    return [n for n in SOURCES if n in want]


def _need(n, a):
    """(MB kept under data/raw, MB of text written, MB of memory at most)."""
    s = SOURCES[n]
    text, ram = s.get("text_mb"), s.get("ram_mb")
    if n == "toolbox":
        text = a.tools_chars * 1.01 / 1e6
    elif n == "textworld":
        text, ram = a.textworld_games * 0.0025, a.workers * 350
    elif n == "arc3":
        text = 200                  # ~20 games' solutions and walks, 4 KB a screen
    elif n in ("arc1", "arc2"):
        text = text * a.arc_augment / 64
    return s.get("mb", 0), text or 0, ram or 500


def _gb(mb):
    return f"{mb / 1000:,.1f} GB"


def _fs(path):
    """(device, free bytes) of the filesystem a path is or will be on."""
    p = os.path.abspath(path)
    while not os.path.exists(p):
        p = os.path.dirname(p)
    return os.stat(p).st_dev, shutil.disk_usage(p).free


def report(a, names):
    """The plan, source by source, and what it needs in all: (disk MB, memory
    MB). Built sources need nothing more."""
    print(f"{'source':<25} {'lane':<15} {'kept':>9} {'text':>9}  "
          f"{'licence':<38} status")
    raw = text = ram = 0
    for n in names:
        s = SOURCES[n]
        done = os.path.exists(_built(a.raw, n)) and not a.force
        k, t, r = _need(n, a)
        status = "built" if done else "to build"
        miss = _missing(s, a)
        if miss:
            status = f"needs {miss}"
        elif s.get("gated") and not done:
            status += "; gated: accept its terms on its page, then " \
                      "`hf auth login`"
        if n == "arc3" and not done:
            status += "; ARC_API_KEY " + ("set" if "ARC_API_KEY" in os.environ
                                          else "not set: 3 games, not all "
                                               "the public ones")
        if not done:
            raw, text, ram = raw + k, text + t, max(ram, r)
        print(f"{n:<25} {s['lane']:<15} {_gb(k):>9} {_gb(t):>9}  "
              f"{s['licence']:<38} {status}")
    off = [n for n in SOURCES if n not in names]
    if off:
        print(f"\nnot in this run: {', '.join(off)} (add with --with)")
    print(f"\nneeds about {_gb(raw + text)} of disk: {_gb(raw)} kept under "
          f"{a.raw}, {_gb(text)} of text under {a.out}/train and {a.out}/val "
          "(estimated from each source's listing and rows sampled from it)")
    print(f"and about {_gb(ram)} of memory at most: sources are built one at "
          "a time")
    (dr, fr), (do, fo) = _fs(a.raw), _fs(a.out)
    if dr == do:
        short = (raw + text) * 1e6 - fr
        print(f"free there now: {fr / 1e9:,.1f} GB"
              + (f" - {short / 1e9:,.1f} GB short" if short > 0 else ""))
    else:
        print(f"free now: {fr / 1e9:,.1f} GB for {a.raw}, {fo / 1e9:,.1f} GB "
              f"for {a.out}")
    return raw + text, ram


def main():
    ap = argparse.ArgumentParser(
        prog="python3 -m corpora agent",
        description="build the next stage's lanes: tool use, agent "
                    "trajectories, ARC")
    ap.add_argument("--check", action="store_true",
                    help="say what would be downloaded and built, and what is "
                         "missing; download and write nothing")
    ap.add_argument("--yes", action="store_true",
                    help="start without asking whether the space it needs is "
                         "there to give")
    ap.add_argument("--only", nargs="+", metavar="SOURCE", choices=list(SOURCES),
                    help="build just these")
    ap.add_argument("--skip", nargs="+", metavar="SOURCE", choices=list(SOURCES))
    ap.add_argument("--with", dest="add", nargs="+", metavar="SOURCE",
                    choices=list(SOURCES),
                    help="add the sources that are off by default, the ones "
                         "that state no licence: agentinstruct, nvarc, "
                         "nvarc-artifacts, arc-community-unlicensed")
    ap.add_argument("--force", action="store_true",
                    help="rebuild sources that are already built")
    ap.add_argument("--out", default="data",
                    help="writes <out>/train/<lane>, <out>/val/<lane> and the "
                         "ARC evaluation sets under <out>/eval")
    ap.add_argument("--raw", default="data/raw/agent",
                    help="where downloads are kept")
    ap.add_argument("--tools-chars", type=int, default=200_000_000,
                    help="size of the generated tools lane")
    ap.add_argument("--textworld-games", type=int, default=20_000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                    help="processes generating TextWorld games")
    ap.add_argument("--arc-augment", type=int, default=64,
                    help="copies of each hand-made ARC task, turned, mirrored "
                         "and recolored")
    ap.add_argument("--arc1-eval-for-training", action="store_true",
                    help="train on ARC-AGI-1's 400 evaluation tasks - 376 of "
                         "them are in ARC-AGI-2's training set - and give up "
                         "ARC-AGI-1's evaluation as a clean benchmark; "
                         "ARC-AGI-2's stays out either way")
    ap.add_argument("--arc3-python", default=None,
                    help="a Python 3.12 with arc-agi installed; without it one "
                         "is made with uv")
    ap.add_argument("--arc3-steps", type=int, default=500_000,
                    help="game steps the ARC-AGI-3 search spends on each game")
    ap.add_argument("--arc3-hold-every", type=int, default=5,
                    help="every n-th public ARC-AGI-3 game is held out")
    ap.add_argument("--val-per-mille", type=int, default=10,
                    help="records per thousand held out")
    ap.add_argument("--limit", type=int, default=0, metavar="N",
                    help="records per source, for a trial run")
    a = ap.parse_args()
    os.chdir(ROOT)
    names = plan(a)
    disk, ram = report(a, names)
    if a.check:
        return 0
    if not a.yes:
        if not sys.stdin.isatty():
            print("\nnot started: run again with --yes once that space is "
                  "there", file=sys.stderr)
            return 1
        ask = (f"\nThis needs about {_gb(disk)} of disk and {_gb(ram)} of "
               "memory. Ready to provide it? [y/N] ")
        if input(ask).strip().lower() not in ("y", "yes"):
            print("not started")
            return 1

    # in SOURCES order, so each source is compared against every source
    # before it - built now, or in an earlier run and not rebuilt
    seen, done, failed = set(), {}, []
    for n in SOURCES:
        spec = SOURCES[n]
        if n not in names:
            _load_keys(a.raw, n, seen)
            continue
        miss = _missing(spec, a)
        if miss:
            print(f"!! {n}: needs {miss} - skipping", file=sys.stderr)
            failed.append(n)
            continue
        try:
            got = build(n, spec, a, seen)
        except Exception as e:                 # one source never stops the rest
            msg = f"{type(e).__name__}: {e}"
            if spec.get("gated") and ("401" in msg or "403" in msg
                                      or "Gated" in msg):
                msg += ("\n   this dataset is gated: accept its terms at "
                        f"https://huggingface.co/datasets/{spec['repo']} and "
                        "log in with `hf auth login`")
            print(f"!! {n} failed: {msg}", file=sys.stderr, flush=True)
            failed.append(n)
            continue
        if got and got.get("failed"):
            failed.append(n)
        elif got:
            done[n] = got

    manifest = os.path.join(a.raw, "manifest.json")
    every = {}
    for n in SOURCES:
        if os.path.exists(_built(a.raw, n)):
            with open(_built(a.raw, n)) as f:
                every[n] = json.load(f)
    os.makedirs(a.raw, exist_ok=True)
    with open(manifest, "w") as f:
        json.dump({"sources": every,
                   "kept_for_benchmarks": [
                       "when2call/test (MCQ and LLM-judge test sets)",
                       "alfworld json_2.1.1/valid_unseen",
                       f"{a.out}/eval/arc-agi-1 (ARC-AGI-1's 400 public "
                       "evaluation tasks"
                       + (": TRAINED ON, so not a clean benchmark)"
                          if a.arc1_eval_for_training else ")"),
                       f"{a.out}/eval/arc-agi-2 (ARC-AGI-2's 120)",
                       f"{a.out}/eval/arc-agi-3/held_out_games.json (public "
                       "ARC-AGI-3 games never played for training)"]},
                  f, indent=1)

    print("\nlanes:")
    for lane in LANES:
        rows = [(n, v) for n, v in every.items() if v["lane"] == lane]
        if not rows:
            continue
        tc = sum(v["train_chars"] for _, v in rows)
        vc = sum(v["val_chars"] for _, v in rows)
        print(f"  {lane:<15} {tc / 1e6:>8,.1f}M characters to train, "
              f"{vc / 1e6:>6,.1f}M held out  ({', '.join(n for n, _ in rows)})")
    print(f"\nmanifest, with every source's licence: {manifest}")
    if failed:
        print(f"\nnot built: {' '.join(failed)}")
        print(f"retry with: python3 -m corpora agent --only {' '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
