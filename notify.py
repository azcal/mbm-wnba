"""
Board delivery to Discord.

Rendering notes, because the obvious approach breaks.

**Entries arrive as multi-line strings.** A board row and its live-entry notes are joined
with newlines before they reach here, so `lines` is a list of blocks, not a list of lines.
Anything that wraps an entry in single backticks produces a code span containing newlines,
which Discord cannot render: the backtick shows up literally and the block collapses. Every
entry is split into real lines first.

**Alignment needs monospace, structure needs markdown.** One big code fence keeps the columns
lined up and reads badly on a phone, where it will not wrap. So the price sits in a short
inline code span, the game and the pick are bold, and live-entry notes use subtext. Short
lines, no horizontal scroll.

**Game first, then the instruction.** A row reads as the fixture, then what to do about it.
Leading with the pick puts the team name twice in two lines and reads backwards to anyone who
is not already holding the context.

Anything unrecognised passes through rather than being dropped, so a format change degrades
instead of silently vanishing.

**Mentions only work in `content`.** Discord does not parse pings inside an embed, so an
`@everyone` in the description renders as plain text and notifies nobody. It goes in the
top-level `content` field with an explicit `allowed_mentions`, and only on the first embed of
a board so a long slate does not ping three times.

Set in the workflow:
    DISCORD_WEBHOOK   channel webhook URL
    DISCORD_MENTION   optional. "@everyone", "@here", or "<@&ROLE_ID>" for a role.
                      Unset means no ping.

No webhook means nothing is sent, and the board still runs and grades.
"""
import os, re, requests

VERSION = "2026-08-03"

GREEN = 0x2E8B57      # normal board
GREY = 0x8A8A8A       # calibration window, predictions only
RED = 0xC0392B        # tripwire

DISCORD_LIMIT = 3800  # embed description cap is 4096; leave room

# "2026-07-31  SEA @ ATL   ATL   find 1.32 / -312 or better"
BOARD = re.compile(r"^(\d{4}-\d{2}-\d{2})\s+(.+?)\s{2,}(\S.*?)\s+find\s+(.+?)\s+or better\s*$")
# "2026-10-22  Anthony Edwards  MIN vs DEN  forecast 33"  and the MMA fight header
HEADER = re.compile(r"^(\d{4}-\d{2}-\d{2})\s{2,}(.+)$")
# indented rows carrying a probability and a price: the props ladder and the MMA markets
#   "    over  29.5   62.6%  find 1.68 / -147 or better"
#   "  both           34.3%  find 3.06 / +206 or better   [indep 3.16, +8%]"
ROW = re.compile(r"^\s+(.+?)\s+([\d.]+%)\s+find\s+(.+?)\s+or better\s*(\[[^\]]*\])?\s*$")
# "      or live 1.69-1.91 (-145 to -110) after Q1 if down 5-9"
LIVE = re.compile(r"^\s+or live\s+([\d.]+)-([\d.]+)\s+\((.+?)\)\s+after\s+(\S+)\s+if\s+(.+)$")
LIVE_ANY = re.compile(r"^\s+or (live .+)$")
# "      changed: price was 1.55"  /  "      new: not on the last board"
# The delta gate emits these on their own line. Appending a marker to the end of a board
# row instead would put a second bracket after "or better", which BOARD and ROW both
# reject, and the row would fall through to the raw handler.
CHANGE = re.compile(r"^\s+(changed|new):\s*(.+)$")


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
            out.append(f"**{game.strip()}**")
            sides = [x.strip() for x in pick.split(" / ")] if " / " in pick else [pick.strip()]
            for j, side in enumerate(sides):
                lead = "Back" if j == 0 else "Or"
                out.append(f"{lead} **{side}** at `{price.strip()}` or better")
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
            lo, hi, am, cp, cond = m.groups()
            out.append(f"-# If {cond} after {cp}, back live at `{lo}` to `{hi}` ({am})")
            continue
        m = LIVE_ANY.match(ln)
        if m:
            out.append(f"-# or {m.group(1)}")
            continue
        m = CHANGE.match(ln)
        if m:
            out.append(f"-# {m.group(1)}: {m.group(2)}")
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
        payload = {"embeds": [embed]}
        mention = os.environ.get("DISCORD_MENTION", "").strip()
        if mention and i == 0:
            payload["content"] = mention
            payload["allowed_mentions"] = {"parse": ["everyone", "roles", "users"]}
        try:
            r = requests.post(url, json=payload, timeout=20)
        except Exception as ex:
            print(f"discord post error: {type(ex).__name__}: {ex}")
            return
        if r.status_code >= 300:
            print(f"discord post failed: HTTP {r.status_code} {r.text[:200]}")
            print(f"  description was {len(desc)} chars, part {i+1} of {len(parts)}")
            return
    print(f"discord: posted {len(body)} lines in {len(parts)} embed(s)")


def send(title, lines, footer=None, color=GREEN):
    """Deliver a board. `lines` may contain multi-line entries; they get split."""
    if not lines and not footer:
        print("notify: nothing to send")
        return
    url = os.environ.get("DISCORD_WEBHOOK")
    if not url:
        print("no notifier configured, board not sent")
        return
    _discord(url, title, lines, footer, color)
