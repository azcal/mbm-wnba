"""
Board delivery. Discord if a webhook is configured, Telegram if a token is, both if both.

Discord is the better home for this: a webhook is a single URL with no bot to register, one
channel per sport stops boards interleaving, and an embed gives a monospace body with a
coloured border and a footer without the message becoming a wall of text.

Set in the workflow:
    DISCORD_WEBHOOK   channel webhook URL
    TG_TOKEN, TG_CHAT the existing pair

Neither set means nothing is sent and the board still runs and grades.
"""
import json, os, requests

GREEN = 0x2E8B57      # normal board
GREY = 0x8A8A8A       # calibration window, predictions only
RED = 0xC0392B        # tripwire

DISCORD_LIMIT = 3800  # embed description cap is 4096; leave room for the code fence
TG_LIMIT = 3900       # message cap is 4096


def _chunks(lines, limit):
    out, cur, n = [], [], 0
    for ln in lines:
        if n + len(ln) + 1 > limit and cur:
            out.append(cur); cur, n = [], 0
        cur.append(ln); n += len(ln) + 1
    if cur:
        out.append(cur)
    return out


def _discord(url, title, lines, footer, color):
    for i, part in enumerate(_chunks(lines, DISCORD_LIMIT)):
        embed = {"title": title if i == 0 else f"{title} (cont.)",
                 "description": "```\n" + "\n".join(part) + "\n```",
                 "color": color}
        if footer and i == len(_chunks(lines, DISCORD_LIMIT)) - 1:
            embed["footer"] = {"text": footer}
        r = requests.post(url, json={"embeds": [embed]}, timeout=20)
        if r.status_code >= 300:
            print(f"discord post failed: {r.status_code} {r.text[:120]}")


def _telegram(token, chat, title, lines, footer):
    body = [title] + lines + ([footer] if footer else [])
    for part in _chunks(body, TG_LIMIT):
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data={"chat_id": chat, "text": "\n".join(part)}, timeout=20)


def send(title, lines, footer=None, color=GREEN):
    """Deliver a board. lines is the list of already-formatted rows."""
    if not lines and not footer:
        return
    url = os.environ.get("DISCORD_WEBHOOK")
    tok, chat = os.environ.get("TG_TOKEN"), os.environ.get("TG_CHAT")
    if url:
        _discord(url, title, lines, footer, color)
    if tok and chat:
        _telegram(tok, chat, title, lines, footer)
    if not url and not (tok and chat):
        print("no notifier configured, board not sent")
