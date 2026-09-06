"""Decide whether a message needs the web, before generating anything.

The alternative - offering the model a tool and trusting it to call it - costs
a whole generation turn to find out, and a ~30B local model is not reliably
good at the judgement. This is cruder but it is free, it is inspectable, and
it is the baseline the model-driven tier has to actually beat.
"""
from __future__ import annotations

import re
from datetime import date

# "what is the latest ...", "who won ... today"
TEMPORAL = re.compile(
    r"\b("
    r"today|tonight|yesterday|tomorrow|right now|just now|currently|"
    r"latest|newest|most recent|recently|this week|this month|this year|"
    r"so far|as of|up to date|up-to-date|these days|nowadays|"
    r"news|headlines|breaking"
    r")\b",
    re.IGNORECASE,
)

# "look it up", "search for x", "google x"
IMPERATIVE = re.compile(
    r"\b(search (?:the )?(?:web|internet|online)|look (?:this |that |it )?up|"
    r"google|web search|check online|find out online)\b",
    re.IGNORECASE,
)

# Facts that are wrong the moment they are memorised.
VOLATILE = re.compile(
    r"\b(price|prices|stock price|share price|exchange rate|weather|forecast|"
    r"score|scores|standings|release date|version|changelog|who is the current|"
    r"population of|status of)\b",
    re.IGNORECASE,
)

URL = re.compile(r"https?://\S+", re.IGNORECASE)

# A year at or after the model's training horizon. Anything older is history
# and the model most likely has it.
YEAR = re.compile(r"\b(20[2-9]\d)\b")

# Things that look topical but are the user talking about their own material.
LOCAL_CONTEXT = re.compile(
    r"\b(this (?:code|file|function|repo|error|test|diff)|my (?:code|file|project)|"
    r"the (?:code|file|snippet) (?:above|below))\b",
    re.IGNORECASE,
)


def should_search(text: str, *, today: date | None = None) -> tuple[bool, str]:
    """Return (decision, reason). The reason is logged and shown in /status.

    Kept deliberately conservative. A false positive costs several seconds and
    risks displacing knowledge the model already had with an SEO-spam snippet,
    which is a worse answer than not searching at all - so the triggers are
    ones that are hard to fire by accident.
    """
    probe = (text or "").strip()
    if not probe:
        return False, "empty"
    if LOCAL_CONTEXT.search(probe):
        return False, "about the user's own material"

    if match := URL.search(probe):
        return True, f"message contains a URL ({match.group(0)[:40]})"
    if match := IMPERATIVE.search(probe):
        return True, f"asked to look it up ({match.group(0).lower()})"
    if match := TEMPORAL.search(probe):
        return True, f"time-sensitive wording ({match.group(0).lower()})"
    if match := VOLATILE.search(probe):
        return True, f"fact that changes over time ({match.group(0).lower()})"

    this_year = (today or date.today()).year
    for candidate in YEAR.findall(probe):
        if int(candidate) >= this_year - 1:
            return True, f"asks about {candidate}"

    return False, "no trigger"
