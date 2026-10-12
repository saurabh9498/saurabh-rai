"""
PHI minimisation: replace identifiers with typed placeholders before text leaves the
agent's trusted core (logs, model prompts, analytics).

What this is, and is not:
  * It REDUCES what reaches models and logs. It is not HIPAA de-identification. Safe Harbor
    lists 18 identifier types, and appointment dates tied to a person are themselves PHI.
    So the model endpoint must sit inside the HIPAA boundary anyway (on-prem or covered by
    a BAA); redaction is defence in depth, not a substitute.
  * It is pattern-based. It covers the identifiers a scheduling conversation carries -
    names, date of birth, phone, email, SSN, MRN, street address, ZIP - in the formats
    people actually type, and is tested on them. Free-text clinical notes need a trained
    de-identification model.

Two levels:
  prompt  identifiers removed; appointment dates and relative dates ("next Tuesday") kept,
          because the model needs them to understand the request
  log     everything above plus every explicit date

Verify mode: while the agent is asking for name and date of birth, the message is expected
to contain them, so it is redacted more aggressively - any date counts as a date of birth,
names may be lower case, and a bare "Maria Lopez 03/14/1985" is recognised without cue
words.

The Vault maps placeholders back to real values for one conversation, so a reply to the
patient can say their own name while the model and the logs never saw it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

MONTHS = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
          r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
_NUM_DATE = r"\d{1,2}[/.-]\d{1,2}[/.-](?:\d{4}|\d{2})"
_ISO_DATE = r"\d{4}-\d{1,2}-\d{1,2}"
_MD_DATE = rf"{MONTHS}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?"
_DM_DATE = rf"\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{MONTHS}\.?(?:,?\s+\d{{4}})?"
ANY_DATE = rf"(?:{_ISO_DATE}|{_NUM_DATE}|{_DM_DATE}|{_MD_DATE})"

WORD = r"[^\W\d_]+(?:['’-][^\W\d_]+)*"          # unicode letters; O'Brien, Mary-Ann, José
_WORD_RE = re.compile(WORD)
STOP = {
    "i", "im", "i'm", "me", "my", "mine", "name", "names", "is", "was", "born", "on", "dob", "date",
    "of", "birth", "birthday", "and", "the", "a", "an", "to", "for", "need", "needs", "want", "wants",
    "would", "like", "book", "booking", "cancel", "appointment", "hi", "hello", "hey", "this", "it's",
    "its", "here", "calling", "patient", "yes", "no", "ok", "okay", "please", "with", "at", "in",
    "next", "tomorrow", "today", "can", "could", "you", "help", "thanks", "thank", "sure", "so",
    "um", "uh", "well", "just", "also", "again", "sorry", "good", "morning", "afternoon", "evening",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "not", "fine",
    "have", "had", "has", "am", "are", "be", "been", "see", "get", "make", "new", "visit", "doctor",
}
NAME_CUE = re.compile(r"\b(?:my name is|my name's|name is|name:|this is|i am|i'm|im|it's|its|"
                      r"patient is|patient name is|calling for|calling about)\s+", re.I)

# Order matters: more specific patterns first, so an SSN is not also read as a phone number.
PATTERNS: list[tuple[str, re.Pattern]] = [
    ("DOB", re.compile(rf"\b(?:born(?: on)?|dob|d\.o\.b\.?|date of birth|birth ?date|birthday)\b[:\s]*(?:is\s+)?({ANY_DATE})", re.I)),
    ("SSN", re.compile(r"\b(\d{3}-\d{2}-\d{4})\b")),
    ("SSN", re.compile(r"\b(?:ssn|social security(?: number| no\.?)?)[:#\s]*(?:is\s+)?(\d{3}[-\s]?\d{2}[-\s]?\d{4})\b", re.I)),
    ("MRN", re.compile(r"\b(?:mrn|medical record(?: number| no\.?)?)[:#\s]*(?:is\s+)?([a-z]{0,3}\d{4,10})\b", re.I)),
    ("EMAIL", re.compile(r"\b([\w.+-]+@[\w-]+(?:\.[\w-]+)+)\b")),
    ("PHONE", re.compile(r"(?<!\d)((?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})(?!\d)")),
    ("ADDRESS", re.compile(
        r"\b(\d{1,5}\s+(?:[nsew]\.?\s+)?(?:[a-z]+\s+){1,3}"
        r"(?:st|street|ave|avenue|rd|road|blvd|boulevard|ln|lane|dr|drive|ct|court|way|pl|place|ter|terrace)\b\.?"
        r"(?:,?\s*(?:apt|apartment|unit|suite|ste|#)\.?\s*[\w-]+)?)", re.I)),
    ("ZIP", re.compile(r"\b(?:zip(?: code)?|postal code)[:\s]*(?:is\s+)?(\d{5}(?:-\d{4})?)\b", re.I)),
]
ANY_DATE_RE = re.compile(rf"\b({ANY_DATE})\b", re.I)
LOG_ONLY = [("DATE", ANY_DATE_RE)]


@dataclass
class Vault:
    """Placeholder <-> value mapping for one conversation. Never logged, never sent to a model."""
    _by_value: dict[tuple[str, str], str] = field(default_factory=dict)
    _by_token: dict[str, str] = field(default_factory=dict)
    _counts: dict[str, int] = field(default_factory=dict)

    def token_for(self, kind: str, value: str) -> str:
        key = (kind, value.strip().lower())
        if key not in self._by_value:
            self._counts[kind] = self._counts.get(kind, 0) + 1
            tok = f"[{kind}_{self._counts[kind]}]"
            self._by_value[key] = tok
            self._by_token[tok] = value.strip()
        return self._by_value[key]

    def value(self, token: str) -> str | None:
        return self._by_token.get(token)

    def restore(self, text: str) -> str:
        return re.sub(r"\[[A-Z]+_\d+\]", lambda m: self._by_token.get(m.group(0), m.group(0)), text)


def redact(text: str, vault: Vault, level: str = "prompt", known_names: tuple[str, ...] = (),
           verify: bool = False) -> str:
    if level not in ("prompt", "log"):
        raise ValueError(level)
    out = text
    for kind, pat in PATTERNS:
        out = _sub_group(pat, out, lambda v, k=kind: vault.token_for(k, v))
    if verify:                                   # asked for DOB: any remaining date is one
        out = _sub_group(ANY_DATE_RE, out, lambda v: vault.token_for("DOB", v))
    if level == "log":
        for kind, pat in LOG_ONLY:
            out = _sub_group(pat, out, lambda v, k=kind: vault.token_for(k, v))
    out = _redact_names(out, vault, verify)
    # Names already known for this conversation are removed wherever they appear. This runs
    # last so a name inside an email address is not split out of it.
    for name in sorted({n for n in known_names if n}, key=len, reverse=True):
        out = re.sub(rf"(?<![\w\[]){re.escape(name)}(?!\w)", lambda m: vault.token_for("NAME", m.group(0)),
                     out, flags=re.I)
    return out


def _redact_names(text: str, vault: Vault, verify: bool) -> str:
    spans = []
    for m in NAME_CUE.finditer(text):
        span = _name_after(text, m.end(), verify, max_words=3)
        if span:
            spans.append(span)
    # Verify mode: a message that leads with two or three words and carries a date of birth
    # ("Maria Lopez 03/14/1985") leads with the name.
    if verify and "[DOB_" in text and not spans:
        lead = re.match(r"^\W*(?:(?:hi|hello|hey|yes|ok|okay|sure)\b\W*)*", text, re.I)
        span = _name_after(text, lead.end(), True, max_words=3, min_words=2)
        if span:
            spans.append(span)
    for s, e in sorted(set(spans), reverse=True):
        text = text[:s] + vault.token_for("NAME", text[s:e]) + text[e:]
    return text


def _name_after(text: str, pos: int, verify: bool, max_words: int, min_words: int = 1):
    """Span of up to max_words name-like words starting at pos, or None.

    Outside verify mode every word must be capitalised ("I am fine" is not a name).
    Punctuation or a stop word ends the name.
    """
    words, i = [], pos
    while len(words) < max_words:
        j = i
        while j < len(text) and text[j] in " \t":
            j += 1
        m = _WORD_RE.match(text, j)
        if not m or (words and j == i):
            break
        w = m.group(0)
        if w.lower().replace("’", "'") in STOP or (not verify and not w[0].isupper()):
            break
        words.append(m.span())
        i = m.end()
    if len(words) < max(min_words, 1):
        return None
    return words[0][0], words[-1][1]


def _sub_group(pat: re.Pattern, text: str, repl) -> str:
    """Replace only capture group 1, keeping the cue words ("born on") around it."""
    parts, last = [], 0
    for m in pat.finditer(text):
        s, e = m.span(1)
        if s < last or text[s:e].startswith("["):   # overlapping or already a placeholder
            continue
        parts.append(text[last:s])
        parts.append(repl(text[s:e]))
        last = e
    parts.append(text[last:])
    return "".join(parts)
