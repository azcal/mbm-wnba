"""
Board delivery. Discord if a webhook is configured, Telegram if a token is, both if both.

Rendering notes, because the obvious approach breaks.

**Entries arrive as multi-line strings.** A board row and its live-entry notes are joined
with newlines before they reach here, so `lines` is a list of blocks, not a list of lines.
Anything that wraps an entry in single backticks produces a code span containing newlines,
which Discord cannot render: the backtick shows up literally and the block collapses. Every
entry is split into real lines first.

**Alignment needs monospace, structure needs markdown.** One big code fence keeps the columns
lined up and reads badly on a phone, where it will not wrap. So the price sits in a short
inline code span, the pick and matchup are bold, and live-entry notes use subtext. Short
lines, no horizontal scroll.

Anything unrecognised passes through rather than being dropped, so a format change degrades
instead of silently vanishing.

Set in the workflow:
    DISCORD_WEBHOOK   channel webhook URL
    TG_TOKEN, TG_CHAT the existing pair

Neither set means nothing is sent and the board still runs and grades.
"""
import os, re, requests

GREEN = 0x2E8B57      # normal board
GREY = 0x8A8A8A       # calibration window, predictions only
RED = 0xC0392B        # tripwire

DISCORD_LIMIT = 3800  # embed description cap is 4096; leave room
TG_LIMIT = 3900       # message cap is 4096

# "2026-07-31  SEA @ ATL   ATL   find 1.32 / -312 or better"
BOARD = re.compile(r"^(\d{4}-\d{2}-\d{2})\s+(.+?)\s{2,}(\S.*?)\s+find\s+(.+?)\s+or better\s*$")
# "2026-10-22  Anthony Edwards  MIN vs DEN  forecast 33"  and the MMA fight header
HEADER = re.compile(r"^(\d{4}-\d{2}-\d{2})\s{2,}(.+)$")
# indented rows carrying a probability and a price: the props ladder and the MMA markets
#   "    over  29.5   62.6%  find 1.68 / -147 or better"
#   "  both           34.3%  find 3.06 / +206 or better   [indep 3.16, +8%]"
ROW = re.compile(r"^\s+(.+?)\s+([\d.]+%)\s+find\s+(.+?)\s+or better\s*(\[[^\]]*\])?\s*$")
# "      or live 1.69-1.91 (-145 to -110) after Q1 if down 5-9"
LIVE = re.compile(r"^\s+or (live .+)$")


def _flatten(lines):
    """Entries can be multi-line blocks. Split them into real lines."""
    return [ln.rstrip() for raw in lines for ln in str(raw).split("\n")]


def _render(lines):
    out, prev_blank = [], True
    for ln in _flatten(lines):
        if not ln.strip():
            if not prev_blank:
                out.append("")
                prev_blank = True
            continue
        prev_blank = False
        m = BOARD.match(ln)
        if m:
            _, game, pick, price = m.groups()
            if out and out[-1] != "":
                out.append("")
            out.append(f"**{pick.strip()}**  ·  {game.strip()}")
            out.append(f"`{price.strip()}` or better")
            continue
        m = HEADER.match(ln)
        if m:
            if out and out[-1] != "":
                out.append("")
            out.append(f"**{m.group(2).strip()}**")
            continue
        m = ROW.match(ln)
        if m:
            label, pct, price, note = m.groups()
            tail = f"  -# {note}" if note else ""
            out.append(f"{label.strip()}  **{pct}**  ·  `{price}` or better{tail}")
            continue
        m = LIVE.match(ln)
        if m:
            out.append(f"-# or {m.group(1)}")
            continue
        if ln.strip().startswith("TRIPWIRE"):
            out.append(f"**{ln.strip()}**")
            continue
        out.append(f"`{ln.strip()}`" if ln.startswith(" ") else ln.strip())
    while out and out[-1] == "":
        out.pop()
    return out


def _chunks(lines, limit):
    out, cur, n = [], [], 0
    for ln in lines:
        if n + len(ln) + 1 > limit and cur:
            out.append(cur)
            cur, n = [], 0
        cur.append(ln)
        n += len(ln) + 1
    if cur:
        out.append(cur)
    return out


def _discord(url, title, lines, footer, color):
    body = _render(lines)
    if not body and not footer:
        print("discord: nothing to render after formatting, not posting")
        return
    parts = _chunks(body, DISCORD_LIMIT) or [[]]
    for i, part in enumerate(parts):
        desc = "\n".join(part).strip()
        embed = {"color": color, "description": desc or "\u200b"}
        if title:
            embed["title"] = title if i == 0 else f"{title} (cont.)"
        if footer and i == len(parts) - 1:
            embed["footer"] = {"text": footer}
        try:
            r = requests.post(url, json={"embeds": [embed]}, timeout=20)
        except Exception as ex:
            print(f"discord post error: {type(ex).__name__}: {ex}")
            return
        if r.status_code >= 300:
            print(f"discord post failed: HTTP {r.status_code} {r.text[:200]}")
            print(f"  description was {len(desc)} chars, part {i+1} of {len(parts)}")
            return
    print(f"discord: posted {len(body)} lines in {len(parts)} embed(s)")


def _telegram(token, chat, title, lines, footer):
    body = [title] + _flatten(lines) + ([footer] if footer else [])
    for part in _chunks(body, TG_LIMIT):
        try:
            r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                              data={"chat_id": chat, "text": "\n".join(part)}, timeout=20)
            if r.status_code >= 300:
                print(f"telegram post failed: HTTP {r.status_code} {r.text[:200]}")
                return
        except Exception as ex:
            print(f"telegram post error: {type(ex).__name__}: {ex}")
            return
    print("telegram: posted")


def send(title, lines, footer=None, color=GREEN):
    """Deliver a board. `lines` may contain multi-line entries; they get split."""
    if not lines and not footer:
        print("notify: nothing to send")
        return
    url = os.environ.get("DISCORD_WEBHOOK")
    tok, chat = os.environ.get("TG_TOKEN"), os.environ.get("TG_CHAT")
    if not url and not (tok and chat):
        print("no notifier configured, board not sent")
        return
    if url:
        _discord(url, title, lines, footer, color)
    if tok and chat:
        _telegram(tok, chat, title, lines, footer)
