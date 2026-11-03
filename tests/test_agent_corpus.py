#!/usr/bin/env python3
"""
The next stage's corpus, built with every download stubbed out.

  every source comes out in the one protocol: tools as JSON Schema lines
  before the first turn, a call as JSON between <call> tags, its result
  inside the same bot turn, the markers balanced and each one a single
  symbol
  each source's own way of writing a call is read: Glaive's quoted
  arguments, ToolACE's Python-style calls with spaces in the names,
  Toucan's dict reprs and function_calls, When2Call's <TOOLCALL> lists, Hermes's
  <tool_response>s; the preference set's rejected reply is never written
  the tool dialects meet in one schema: xLAM's flat parameters and
  ", optional", ToolACE's "dict", When2Call's "required" beside the
  parameters, MCP's titles
  parallel calls come out call, result, call, result
  a marker inside a source's text cannot open a turn
  a record that stops on a result nobody read loses it; one with no answer
  is dropped
  every generated episode's calls give the same results when run again on
  its world, and the replies quote what the tools returned
  a build writes data/train/<lane>/<source> and data/val/<lane>, drops the
  duplicates a later source repeats, sends a source's own test split to
  held-out, leaves nothing half-built, and leaves a built source alone on a
  rerun; --check downloads and writes nothing, and a build says what it
  needs and starts only when that is agreed to
"""
import contextlib
import io
import json
import os
import random
import re
import shutil
import sys
import tempfile
import zipfile
from fractions import Fraction

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    extra = extra if (not cond or len(extra) < 100) else ""
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


from corpora import agent as A                                # noqa: E402
from corpora import toolbox as T                              # noqa: E402
from minagi.tokenizer import ByteTokenizer, SPECIAL_ID        # noqa: E402

TOK = ByteTokenizer()
J = json.dumps


def calls(text):
    return [json.loads(m) for m in re.findall(r"<call>(.*?)</call>", text)]


def balanced(text):
    ids = TOK.encode(text).ids
    n = {m: ids.count(SPECIAL_ID[m]) for m in ("<user>", "</user>", "<bot>",
                                               "</bot>")}
    return len(set(n.values())) == 1 and n["<user>"] > 0 \
        and ids.count(SPECIAL_ID["<|endoftext|>"]) == 1


# --------------------------------------------------- one record per source
XLAM = {"id": 7, "query": "What's the weather in Lisbon and in Porto?",
        "tools": J([{"name": "weather.now", "description": "Weather now.",
                     "parameters": {
                         "city": {"type": "str", "description": "A city",
                                  "default": ""},
                         "units": {"type": "str, optional",
                                   "description": "metric or imperial",
                                   "default": "metric"}}}]),
        "answers": J([{"name": "weather.now", "arguments": {"city": "Lisbon"}},
                      {"name": "weather.now", "arguments": {"city": "Porto"}}])}
text = A.render(A.from_xlam(XLAM))
tool = json.loads(text.split("<tools>\n", 1)[1].split("\n", 1)[0])
check("xLAM: flat parameters become a JSON Schema object",
      tool["parameters"]["type"] == "object"
      and tool["parameters"]["properties"]["city"] == {"type": "string",
                                                       "description": "A city"}
      and tool["parameters"]["properties"]["units"]["default"] == "metric"
      and tool["parameters"]["required"] == ["city"], J(tool))
check("xLAM: both calls, as JSON, in order",
      calls(text) == [{"name": "weather.now", "arguments": {"city": "Lisbon"}},
                      {"name": "weather.now", "arguments": {"city": "Porto"}}])
check("xLAM: tools before the first turn, markers balanced",
      text.startswith("<tools>\n") and text.index("</tools>") <
      text.index("<user>") and balanced(text))

GLAIVE = {"system": "SYSTEM: You are a helpful assistant with access to the "
                    "following functions. Use them if required -\n{\n    "
                    "\"name\": \"rate\",\n    \"description\": \"Exchange rate\","
                    "\n    \"parameters\": {\"type\": \"object\", \"properties\":"
                    " {\"base\": {\"type\": \"string\"}}, \"required\": "
                    "[\"base\"]}\n}\n",
          "chat": "USER: How many euros is a dollar?\n\n\nASSISTANT: "
                  "<functioncall> {\"name\": \"rate\", \"arguments\": "
                  "'{\"base\": \"USD\"}'} <|endoftext|>\n\n\nFUNCTION RESPONSE: "
                  "{\"EUR\": 0.92}\n\n\nASSISTANT: One dollar is 0.92 euros. "
                  "<|endoftext|>\n\n\nUSER: Thanks!\n\n\nASSISTANT: You're "
                  "welcome! <|endoftext|>\n\n\n"}
text = A.render(A.from_glaive(GLAIVE))
check("Glaive: the quoted arguments are read as JSON",
      calls(text) == [{"name": "rate", "arguments": {"base": "USD"}}])
check("Glaive: the result sits between the call and the reply, in one bot turn",
      re.search(r"<bot>\n<call>.*</call>\n<result>\n\{\"EUR\": 0.92\}\n</result>"
                r"\nOne dollar is 0.92 euros.\n</bot>", text) is not None, text)
check("Glaive: its end-of-text markers are gone, the record's own is last",
      text.count("<|endoftext|>") == 1 and text.endswith("<|endoftext|>\n")
      and balanced(text))

HERMES = {"id": "x", "tools": J([{"type": "function", "function": {
              "name": "light", "description": "Switch a light.",
              "parameters": {"type": "object", "properties": {
                  "room": {"type": "string"}}, "required": ["room"]}}}]),
          "conversations": [
              {"from": "system", "value": "You are a function calling AI. "
                                          "<tools>[...]</tools>"},
              {"from": "human", "value": "Lights on in the hall and the den."},
              {"from": "gpt", "value": "<tool_call>\n{\"name\": \"light\", "
                                       "\"arguments\": {\"room\": \"hall\"}}\n"
                                       "</tool_call>\n<tool_call>\n{\"name\": "
                                       "\"light\", \"arguments\": {\"room\": "
                                       "\"den\"}}\n</tool_call>"},
              {"from": "tool", "value": "<tool_response>\n{\"name\": \"light\", "
                                        "\"content\": {\"hall\": \"on\"}}\n"
                                        "</tool_response>\n<tool_response>\n"
                                        "{\"name\": \"light\", \"content\": "
                                        "{\"den\": \"on <user> ok\"}}\n"
                                        "</tool_response>"},
              {"from": "gpt", "value": "Both are on."}]}
text = A.render(A.from_hermes(HERMES))
order = re.findall(r"<(call|result)>", text)
check("Hermes: parallel calls come out call, result, call, result",
      order == ["call", "result", "call", "result"], str(order))
check("Hermes: a result's content, not its wrapper",
      "<result>\n{\"hall\": \"on\"}\n</result>" in text)
check("a <user> inside a result is spelled out, not a turn",
      "on &lt;user&gt; ok" in text and balanced(text))
check("Hermes: OpenAI-style tool wrapper unwrapped",
      json.loads(text.split("\n")[1])["name"] == "light")

JSONMODE = {"id": "y", "schema": "{\"Answer\": {\"type\": \"object\"}}",
            "conversations": [
                {"from": "system", "value": "Answer in JSON.\n<schema>\n...\n"
                                            "</schema>"},
                {"from": "human", "value": "Give me an answer."},
                {"from": "gpt", "value": "{\"Answer\": {}}"}]}
text = A.render(A.from_hermes(JSONMODE))
check("Hermes json-mode: the schema comes first, no tools",
      text.startswith("<schema>\n{\"Answer\"") and "<tools>" not in text
      and balanced(text))

TOOLACE = {"system": "You are an expert in composing functions.\nHere is a "
                     "list of functions in JSON format that you can invoke:\n"
                     + J([{"name": "Market Trends API",
                           "description": "Market trends.",
                           "parameters": {"type": "dict", "properties": {
                               "trend_type": {"type": "string"},
                               "country": {"type": "string"}},
                               "required": ["trend_type"]},
                           "required": None}])
                     + "\nShould you decide to return the function call(s)...",
           "conversations": [
               {"from": "user", "value": "What's moving in US markets?"},
               {"from": "assistant", "value": "[Market Trends API(trend_type="
                                              "\"MARKET_INDEXES\", country="
                                              "\"us\")]"},
               {"from": "tool", "value": J([{"name": "Market Trends API",
                                             "results": {"up": ["S&P 500"]}}])},
               {"from": "assistant", "value": "The S&P 500 is up."},
               {"from": "user", "value": "And in Japan, with flags?"},
               {"from": "assistant", "value": "[Market Trends API(trend_type="
                                              "\"MOST_ACTIVE\", country=\"jp\", "
                                              "extra=[1, 2], strict=true, "
                                              "note=\"a, b) c\")]"}]}
text = A.render(A.from_toolace(TOOLACE))
check("ToolACE: Python-style calls with a spaced name, lists, JSON words, "
      "and a string holding ') ,'",
      calls(text) == [
          {"name": "Market Trends API",
           "arguments": {"trend_type": "MARKET_INDEXES", "country": "us"}},
          {"name": "Market Trends API",
           "arguments": {"trend_type": "MOST_ACTIVE", "country": "jp",
                         "extra": [1, 2], "strict": True, "note": "a, b) c"}}],
      J(calls(text)))
tool = json.loads(text.split("\n")[1])
check("ToolACE: \"dict\" becomes object; the JSON list read out of the prompt",
      tool["parameters"]["type"] == "object"
      and tool["parameters"]["required"] == ["trend_type"])
check("ToolACE: the result is its \"results\"",
      "<result>\n{\"up\": [\"S&P 500\"]}\n</result>" in text)
check("text that only looks like a list is left as text",
      A._pycalls("[1] First, check the plug.") is None
      and A._pycalls("[Get Time()]") == [("Get Time", {})])

W2C_SFT = {"tools": [J({"name": "orders.get", "description": "An order.",
                        "parameters": {"type": "dict", "properties": {
                            "order_id": {"type": "int",
                                         "description": "Its number"}}},
                        "required": ["order_id"]})],
           "messages": [{"role": "user", "content": "Where is order 12?"},
                        {"role": "assistant", "content": "<TOOLCALL>[{\"name\":"
                         " \"orders.get\", \"arguments\": {\"order_id\": 12}}]"
                         "</TOOLCALL>"}]}
text = A.render(A.from_when2call(W2C_SFT))
tool = json.loads(text.split("\n")[1])
check("When2Call: <TOOLCALL> list read; \"required\" beside the parameters kept",
      calls(text) == [{"name": "orders.get", "arguments": {"order_id": 12}}]
      and tool["parameters"]["required"] == ["order_id"]
      and tool["parameters"]["properties"]["order_id"]["type"] == "integer")
W2C_PREF = {"tools": W2C_SFT["tools"],
            "messages": [{"role": "user", "content": "Where is my order?"}],
            "chosen_response": {"role": "assistant",
                                "content": "Which order number is it?"},
            "rejected_response": {"role": "assistant", "content":
                                  "<TOOLCALL>[{\"name\": \"orders.get\", "
                                  "\"arguments\": {\"order_id\": 1}}]"
                                  "</TOOLCALL>"}}
text = A.render(A.from_when2call(W2C_PREF))
check("When2Call preference: the chosen reply, never the rejected one",
      "Which order number is it?" in text and "<call>" not in text)

SMOL = {"messages": [
    {"role": "system", "content": "You are an expert in composing functions."
                                  "\n<tools>" + J([{"name": "is_prime",
                                                    "description": "Prime?",
                                                    "parameters": {"n": {
                                                        "type": "int",
                                                        "description": "n"}}}])
                                  + "</tools>\nThe output MUST ..."},
    {"role": "user", "content": "Are 97 and 91 prime?"},
    {"role": "assistant", "content": "<tool_call>[{\"name\": \"is_prime\", "
                                     "\"arguments\": {\"n\": 97}}, {\"name\": "
                                     "\"is_prime\", \"arguments\": {\"n\": 91}}]"
                                     "</tool_call>"}]}
text = A.render(A.from_smoltalk(SMOL))
check("SmolTalk: a JSON list inside <tool_call>, tools out of <tools>",
      [c["arguments"]["n"] for c in calls(text)] == [97, 91]
      and json.loads(text.split("\n")[1])["name"] == "is_prime")

TOUCAN = {"uuid": "u", "subset_name": "single-turn-original",
          "question": "q", "target_tools": "rhymes",
          "tools": J([{"type": "function", "function": {
              "name": "lyric-rhymes", "description": "\n   Rhymes for a word. \n",
              "parameters": {"properties": {"word": {
                  "title": "Word", "type": "string"}}, "required": ["word"],
                  "title": "rhymesArguments", "type": "object"}}}]),
          "messages": J([
              {"role": "user", "content": "Rhymes for 'smile' and 'light'?"},
              {"role": "assistant", "content": "Let me look both up."},
              {"role": "tool_call", "content": "{'name': 'lyric-rhymes', "
                                               "'arguments': '{\"word\": "
                                               "\"smile\"}'}"},
              {"role": "tool_call", "content": "{'name': 'lyric-rhymes', "
                                               "'arguments': '{\"word\": "
                                               "\"light\"}'}"},
              {"role": "tool_response", "content": "[\"aisle\", \"while\"]"},
              {"role": "tool_response", "content": "[\"night\", \"kite\"]"},
              {"role": "assistant", "content": "Smile: aisle, while. Light: "
                                               "night, kite."}])}
text = A.render(A.from_toucan(TOUCAN))
tool = json.loads(text.split("\n")[1])
check("Toucan: dict-repr calls read, results paired with them",
      [c["arguments"]["word"] for c in calls(text)] == ["smile", "light"]
      and re.findall(r"<(call|result)>", text) == ["call", "result", "call",
                                                   "result"])
check("Toucan: MCP titles dropped, the description's whitespace folded",
      "title" not in J(tool) and tool["description"] == "Rhymes for a word.")

TOUCAN_FULL = {"uuid": "v", "subset_name": "single-turn-original",
               "available_tools": J([{"type": "function", "function": {
                   "name": "geo-find", "description": "A place.",
                   "parameters": {"type": "object", "properties": {
                       "q": {"type": "string"}}, "required": ["q"]}}}]),
               "messages": J([
                   {"role": "system", "content": "<|im_system|>tool_declare..."},
                   {"role": "user", "content": "Where is Porto?"},
                   {"role": "assistant", "content": "",
                    "reasoning_content": "Look it up."},
                   {"role": "assistant", "content": "", "function_call":
                    "{'name': 'geo-find', 'arguments': '{\"q\": \"Porto\"}', "
                    "'call_id': 'call_1'}"},
                   {"role": "function", "name": "geo-find",
                    "content": "{\"country\": \"Portugal\"}"},
                   {"role": "assistant", "content": "Porto is in Portugal."}])}
text = A.render(A.from_toucan(TOUCAN_FULL))
check("Toucan's full sets: function_call read, `function` results kept, "
      "reasoning_content is the think step, tools from available_tools",
      calls(text) == [{"name": "geo-find", "arguments": {"q": "Porto"}}]
      and "<result>\n{\"country\": \"Portugal\"}\n</result>" in text
      and "<think>\nLook it up.\n</think>" in text
      and json.loads(text.split("\n")[1])["name"] == "geo-find"
      and "im_system" not in text and balanced(text), text[-400:])

AGENT = {"id": "os_0", "conversations": [
    {"from": "human", "value": "You are in a shell. Count the files in /etc."},
    {"from": "gpt", "value": "Think: list them.\nAct: bash\n```bash\nls /etc | "
                             "wc -l\n```"},
    {"from": "human", "value": "The output of the OS:\n220\n\x1b]0;root@x: /"
                               "\x07root@x:/# \x1b[K\r"},
    {"from": "gpt", "value": "Think: done.\nAct: answer(220)"}]}
text = A.render(A.from_agentinstruct(AGENT))
check("AgentInstruct: the environment speaks as the user, no tools",
      text.count("<user>") == 2 and "<tools>" not in text and balanced(text))
check("terminal escapes and control characters are taken out",
      "220\nroot@x:/#" in text and not re.search(r"[\x00-\x08\x0b-\x1f]", text),
      repr(text[-120:]))

PAGE = ("# frob\n\n> Frobnicate things.\n> More information: <https://example."
        "org/frob>.\n\n- [l]ist every frob:\n\n`frob {{[-l|--list]}}`\n\n- "
        "SSH into a frob host:\n\n`frob ssh {{host}}`\n")
text = A.render(A.from_tldr(PAGE, "linux"))
check("tldr: a page becomes what-is, then how-do-I per example",
      "What is `frob` for on Linux?" in text
      and "With `frob`, how do I list every frob?" in text
      and "`frob [-l|--list]`" in text
      and "how do I SSH into a frob host?" in text and balanced(text), text)

from corpora import worlds as W                              # noqa: E402

GAME = ("\n\n      $$$$$$\\  $$\\   $$\\\n      \\_$$  _|$$ |  $$ |\n\n"
        "Welcome! Your task: put the key on the table.\n\n-= Hall =-\nYou "
        "are in a hall. A key sits here.\n\n>  ")
END = ("You put the key on the table.\n\n\n                *** The End ***\n\n"
       "You scored 1 out of a possible 1, in 2 turn(s).\n\n\nWould you like to "
       "RESTART, RESTORE a saved game, QUIT or UNDO the last command?\n> ")
text = A.render(W.transcript(W.clean(GAME), [("take key", "Taken."),
                                             ("put key on table", W.clean(END))],
                             W.ACT))
check("a game's banner, prompt and offer to restart are cut; the commands "
      "are calls to act",
      "$$" not in text and "RESTART" not in text and "\n>" not in text
      and "-= Hall =-" in text and "*** The End ***" in text
      and [c["arguments"]["command"] for c in calls(text)]
      == ["take key", "put key on table"] and balanced(text), text[-300:])

text = A.render(A.from_tldr(PAGE, "linux", "de"))
check("tldr: a translation is asked for whole, in its language",
      "Show me the tldr page for `frob` on Linux, in German." in text
      and "- list every frob:\n`frob [-l|--list]`" in text and balanced(text))

# ------------------------------------------------------------- the edges
cut = {"tools": [], "turns": [("user", "Do it."),
                              ("bot", [("call", "f", {}), ("result", "ok")])]}
text = A.render(cut)
check("a record ending on an unread result keeps the call, drops the result",
      text is not None and "<call>" in text and "<result>" not in text)
check("a record with no answer is dropped",
      A.render({"tools": [], "turns": [("user", "Hello?")]}) is None
      and A.render(None) is None)
check("an unreadable call drops the record",
      A.from_xlam({**XLAM, "answers": "[{\"name\": \"f\", \"arguments\": "
                                      "\"not json\"}]"}) is None)


# --------------------------------------------------------- the generator
def pages(where, n):
    os.makedirs(where, exist_ok=True)
    rng = random.Random(0)
    for i in range(n):
        name = f"Lake {chr(65 + i % 26)}{i}"
        with open(os.path.join(where, f"{name}.txt"), "w") as f:
            f.write(f"<page>\n    <title>{name}</title>\n    <revision><text "
                    f"xml:space=\"preserve\">'''{name}''' is a lake in the "
                    f"[[North|north]] of a country, with {rng.randint(2, 99)} "
                    f"islands.{{{{cite|x}}}} It is &amp;ndash; deep.\n\n"
                    f"== History ==\nSettled early.</text></revision>\n"
                    f"</page>\n")


tmp = tempfile.mkdtemp(prefix="agent-corpus-")
try:
    pages(os.path.join(tmp, "wiki"), 300)
    lib = T.Library.from_lane(os.path.join(tmp, "wiki"), 1000, False,
                              random.Random(0))
    lead = lib.lead(lib.titles[0])
    check("wikitext comes out as plain text",
          not re.search(r"\[\[|'''|\{\{|&ndash;|&amp;", lead) and "–" in lead,
          lead)
    rng = random.Random(3)
    mismatch, n, unquoted, looked = 0, 0, [], 0
    for _ in range(1500):
        got = T.episode(rng, lib)
        if got is None:
            continue
        ep, world, _ = got
        n += 1
        mismatch += bool(T.replay(ep, world))
        for role, body in ep["turns"]:
            if role != "bot":
                continue
            reply = " ".join(s[1] for s in body if s[0] == "text")
            for i, s in enumerate(body):
                if s[0] != "call":
                    continue
                looked += s[1] in ("search", "open_page")
                r = body[i + 1][1]
                if s[1] == "calculator" and not r.startswith("Error"):
                    v = Fraction(r)
                    if not (r in reply or T._money(r) in reply
                            or T.number(abs(v)) in reply or v == 0):
                        unquoted.append((r, reply))
                if s[1] == "convert_units":
                    if r.split(" = ")[1].rsplit(" ", 1)[0] not in reply:
                        unquoted.append((r, reply))
    check("every generated call gives the same result run again on its world",
          n > 1000 and mismatch == 0, f"{mismatch} of {n}")
    check("replies quote what the calculator and the converter returned",
          not unquoted, str(unquoted[:2]))
    check("look-ups are generated when an encyclopedia is there", looked > 50,
          str(looked))
    check("the calculator is exact and refuses what it cannot do",
          T.calc("123456789 * 987654321") == "121932631112635269"
          and T.calc("1 / 3") == "0.3333333333" and T.calc("0.1 + 0.2") == "0.3"
          and T.calc("round(2.675, 2)") == "2.68"
          and T.World().run("calculator", {"expression": "2 ** 1000"})
          .startswith("Error")
          and T.World().run("calculator", {"expression": "__import__('os')"})
          .startswith("Error"))

    # ------------------------------------------------- a build, offline
    raw, out = os.path.join(tmp, "raw"), os.path.join(tmp, "data")
    pages(os.path.join(out, "train", "wikipedia"), 300)
    dup_of_xlam = {"messages": [
        {"role": "system", "content": "<tools>" + XLAM["tools"] + "</tools>"},
        {"role": "user", "content": XLAM["query"]},
        {"role": "assistant", "content": "<tool_call>" + XLAM["answers"]
                                         + "</tool_call>"}]}
    FILES = {
        "xlam": {"xlam_function_calling_60k.json": [XLAM]},
        "toolace": {"data.json": [TOOLACE]},
        "hermes": {"func-calling.json": [HERMES],
                   "json-mode-singleturn.json": [JSONMODE]},
        "when2call": {"train/when2call_train_sft.jsonl": [W2C_SFT],
                      "train/when2call_train_pref.jsonl": [W2C_PREF],
                      "test/when2call_test_mcq.jsonl": [{"uuid": "t"}]},
        "smoltalk": {"data/apigen-80k/train-00000-of-00001.parquet":
                     [SMOL, dup_of_xlam],
                     "data/apigen-80k/test-00000-of-00001.parquet":
                     [{"messages": SMOL["messages"][:2] + [
                         {"role": "assistant", "content": "I can't tell."}]}]},
        "glaive": {"glaive-function-calling-v2.json": [GLAIVE]},
        "toucan": {"SFT/train-00000-of-00003.parquet": [TOUCAN],
                   "OSS/train-00000-of-00047.parquet": [TOUCAN_FULL]},
        "agentinstruct": {"data/os-00000-of-00001-a.parquet": [AGENT]},
    }
    fetched = []

    def fake_fetch(name, repo, raw_dir, patterns):
        import pyarrow as pa
        import pyarrow.parquet as pq
        fetched.append(name)
        where = os.path.join(raw_dir, name, repo.split("/")[-1])
        for rel, rows in FILES[name].items():
            path = os.path.join(where, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if rel.endswith(".parquet"):
                pq.write_table(pa.Table.from_pylist(rows), path)
            elif rel.endswith(".jsonl"):
                with open(path, "w") as f:
                    f.writelines(J(r) + "\n" for r in rows)
            else:
                with open(path, "w") as f:
                    json.dump(rows, f)
        return where

    os.makedirs(os.path.join(raw, "tldr"))
    with zipfile.ZipFile(os.path.join(raw, "tldr", "tldr-main.zip"), "w") as z:
        z.writestr("tldr-main/pages/linux/frob.md", PAGE)
        z.writestr("tldr-main/pages.de/linux/frob.md", PAGE)   # a translation
    A.fetch_hf = fake_fetch
    A.shutil.disk_usage = lambda p: shutil._ntuple_diskusage(10**13, 0, 10**13)

    def run(*args, agree=True):
        sys.argv = ["agent", "--out", out, "--raw", raw,
                    "--tools-chars", "40000"] + list(args) \
            + (["--yes"] if agree and "--check" not in args else [])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            rc = A.main()
        return rc, buf.getvalue()

    rc, log = run("--check")
    check("--check fetches and writes nothing", not fetched
          and not os.path.exists(raw + "/built") and "xlam" in log
          and "agentinstruct" in log.split("not in this run:")[-1], log[-300:])

    rc, log = run("--only", "xlam", agree=False)
    check("without --yes and with no one at a terminal to ask, nothing starts",
          rc == 1 and not fetched and "--yes" in log
          and "needs about" in log and "of memory" in log, log[-300:])

    names = ["toolbox", "xlam", "toolace", "hermes", "when2call", "smoltalk",
             "glaive", "tldr", "toucan"]
    rc, log = run("--only", *names, "--with", "agentinstruct")
    check("an offline build of every downloaded source succeeds", rc == 0,
          log[-800:])
    texts = {}
    for lane in ("tools", "function-calls", "trajectories"):
        for dp, dirs, fs in os.walk(os.path.join(out, "train", lane)):
            for f in fs:
                texts[os.path.join(dp, f)] = open(os.path.join(dp, f)).read()
    check("the layout: train/<lane>/<source>/<shard>/part-*.txt",
          os.path.exists(os.path.join(out, "train", "function-calls", "xlam",
                                      "0000", "part-000000.txt"))
          and os.path.isdir(os.path.join(out, "train", "trajectories",
                                         "toucan"))
          and os.path.isdir(os.path.join(out, "train", "tools", "toolbox")))
    held = sorted(os.listdir(os.path.join(out, "val", "function-calls")))
    check("SmolTalk's test split went to held-out, named for its source",
          any(f.startswith("smoltalk-") for f in held), str(held))
    check("the tools lane has a held-out side of its own",
          any(f.startswith("toolbox-") for f in
              os.listdir(os.path.join(out, "val", "tools"))))
    state = json.load(open(os.path.join(raw, "manifest.json")))["sources"]
    check("SmolTalk's copy of the xLAM record is dropped as a duplicate",
          state["smoltalk"]["duplicates"] == 1, J(state["smoltalk"]))
    every = "".join(texts.values())
    records = [r + "<|endoftext|>\n" for r in every.split("<|endoftext|>\n")
               if r.strip()]
    check("every record written is balanced and ends the same way",
          all(balanced(r) for r in records), str(len(records)))
    check("tldr's English page and its translation are both read",
          every.count("What is `frob` for") == 1
          and every.count("in German.") == 1)
    check("When2Call's test set is downloaded but never written to train",
          os.path.exists(os.path.join(raw, "when2call", "When2Call", "test",
                                      "when2call_test_mcq.jsonl"))
          and "\"uuid\"" not in every)
    from minagi.ingest import collect
    seen = collect([os.path.join(out, "train")], cache=None, quiet=True)
    check("the reader takes every file as text; nothing is left half-built",
          sorted(seen) == sorted(p for p in texts) + sorted(
              p for p in seen if "/wikipedia/" in p)
          and not os.listdir(os.path.join(raw, "partial")), str(len(seen)))

    before = {p: os.path.getmtime(p) for p in texts}
    fetched.clear()
    rc, log = run("--only", "xlam", "smoltalk")
    check("a rerun leaves built sources alone and fetches nothing",
          not fetched and "already in" in log
          and all(os.path.getmtime(p) == t for p, t in before.items()))
    rc, log = run("--only", "smoltalk", "--force")
    state = json.load(open(os.path.join(raw, "manifest.json")))["sources"]
    check("a forced rebuild still knows xLAM's records, built before it",
          rc == 0 and state["smoltalk"]["duplicates"] == 1, log[-400:])
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n  all agent corpus tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
