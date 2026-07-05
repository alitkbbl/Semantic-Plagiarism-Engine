"""
preprocessing.py
=================

Text preprocessing pipeline shared by both detection backends
(Shingling + MinHash + LSH, and TF-IDF weighted SimHash).

Pipeline stages (in order):
    1. Character normalization (unicode + Persian character folding)
    2. Punctuation stripping
    3. Whitespace collapsing
    4. Stop word removal (English + Persian)
    5. Tokenization
    6. Word shingle ("k-gram") construction

The pipeline is intentionally dependency-free (stdlib only) so that it can
run on any machine without extra installs, and so behaviour is fully
transparent for the technical report.
"""

from __future__ import annotations

import re
import string
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Set, Tuple

# --------------------------------------------------------------------------- #
# Stop words
# --------------------------------------------------------------------------- #

# A small, classic English stop word list. This is intentionally not
# exhaustive -- the goal is a transparent, easily-audited list rather than a
# maximal one imported from a third-party corpus.
STOPWORDS_EN: frozenset = frozenset(
    """
    a an and are as at be by for from has have he her hers him his i in
    is it its of on or our ours she that the their theirs them they this
    to was we were will with you your yours
    """.split()
)

# Persian (Farsi) equivalents of the same closed-class function words,
# as explicitly requested by the project specification ("and", "is",
# "the", "or" and their Persian equivalents).
STOPWORDS_FA: frozenset = frozenset(
    """
    و است هست را در به از که این آن یا با برای بر تا هم یک آنها ایشان
    من تو او ما شما نیز اگر اما ولی هر هیچ چون چرا کدام همه بی
    """.split()
)

STOPWORDS: frozenset = STOPWORDS_EN | STOPWORDS_FA

# --------------------------------------------------------------------------- #
# Character normalization
# --------------------------------------------------------------------------- #

# Arabic-script look-alikes that should be folded to their canonical
# Persian forms before comparison (very common source of false negatives
# in Persian text, since many keyboards/input methods produce the Arabic
# variants interchangeably).
_PERSIAN_CHAR_MAP = {
    "\u064a": "\u06cc",  # ARABIC YEH -> PERSIAN YEH (ی)
    "\u0643": "\u06a9",  # ARABIC KAF -> PERSIAN KEHEH (ک)
    "\u0629": "\u0647",  # ARABIC TEH MARBUTA -> HEH (ة -> ه)
    "\u06c0": "\u0647",  # HEH DOACHASHMEE -> HEH
    "\u0654": "",        # ARABIC HAMZA ABOVE (combining) -> drop
    "\u0640": "",        # ARABIC TATWEEL (kashida) -> drop
}

# Arabic/Persian diacritics (harakat) - purely phonetic marks that should
# not affect duplicate detection.
_ARABIC_DIACRITICS_RE = re.compile(
    "[" + "".join(chr(c) for c in range(0x064B, 0x0653)) + "]"
)

# Punctuation to strip: standard ASCII punctuation plus common
# Arabic/Persian punctuation marks that do not appear in `string.punctuation`.
_PERSIAN_PUNCTUATION = "،؛؟٪«»…ـ"
_PUNCTUATION_TABLE = str.maketrans(
    {ch: " " for ch in (string.punctuation + _PERSIAN_PUNCTUATION)}
)

_WHITESPACE_RE = re.compile(r"\s+", flags=re.UNICODE)


def _strip_control_characters(text: str) -> str:
    """Drop unicode control/format/surrogate characters.

    Guards against "documents containing unusual characters" (stray control
    bytes, zero-width joiners, unpaired surrogates from bad encodings, etc.)
    causing crashes or corrupting shingle boundaries downstream. Whitespace
    characters such as tab/newline/carriage-return are technically in the
    "Cc" (control) unicode category too, but must be preserved here so the
    later whitespace-collapsing step can turn them into normal spaces
    instead of having words silently glued together.
    """
    return "".join(
        ch
        for ch in text
        if ch.isspace() or unicodedata.category(ch) not in ("Cc", "Cf", "Cs", "Co")
    )


def normalize_text(text: str) -> str:
    """Normalize raw text prior to tokenization.

    Steps: unicode NFKC normalization, control-character removal, Persian
    character folding, diacritic removal, lower-casing, punctuation removal
    and whitespace collapsing. Safe to call on empty strings.
    """
    if text is None:
        return ""

    # Guard against non-string input (e.g. NaN from a pandas cell).
    if not isinstance(text, str):
        text = str(text)

    text = unicodedata.normalize("NFKC", text)
    text = _strip_control_characters(text)

    for src, dst in _PERSIAN_CHAR_MAP.items():
        text = text.replace(src, dst)
    text = _ARABIC_DIACRITICS_RE.sub("", text)

    text = text.lower()
    text = text.translate(_PUNCTUATION_TABLE)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Whitespace tokenization on already-normalized text."""
    normalized = normalize_text(text)
    if not normalized:
        return []
    return normalized.split(" ")


def remove_stopwords(tokens: Sequence[str], stopwords: Iterable[str] = STOPWORDS) -> List[str]:
    """Filter stop words out of a token sequence."""
    sw = stopwords if isinstance(stopwords, (set, frozenset)) else set(stopwords)
    return [t for t in tokens if t and t not in sw]


def build_shingles(tokens: Sequence[str], k: int = 3) -> Set[str]:
    """Build word-level shingles (k-grams) from a token sequence.

    Edge cases handled explicitly:
      * Empty document (no tokens) -> empty shingle set.
      * Very short document (fewer tokens than k) -> the whole document is
        treated as a single shingle, so short documents are not silently
        dropped from comparison entirely.
    """
    if k < 1:
        raise ValueError("Shingle size k must be >= 1")
    n = len(tokens)
    if n == 0:
        return set()
    if n < k:
        return {" ".join(tokens)}
    return {" ".join(tokens[i : i + k]) for i in range(n - k + 1)}


@dataclass
class PreprocessedDocument:
    """Container for the result of running the full preprocessing pipeline
    on a single document."""

    raw_text: str
    normalized_text: str
    tokens: List[str] = field(default_factory=list)
    shingles: Set[str] = field(default_factory=set)

    @property
    def is_empty(self) -> bool:
        return len(self.tokens) == 0


def preprocess_document(
    text: str,
    shingle_size: int = 3,
    stopwords: Iterable[str] = STOPWORDS,
    drop_stopwords: bool = True,
) -> PreprocessedDocument:
    """Run the complete preprocessing pipeline on a single document.

    Returns a :class:`PreprocessedDocument` holding the normalized text,
    the (optionally stop-word-filtered) token list, and the shingle set
    used for Jaccard / MinHash similarity.
    """
    normalized = normalize_text(text)
    tokens = tokenize(text)
    if drop_stopwords:
        tokens = remove_stopwords(tokens, stopwords)
    shingles = build_shingles(tokens, k=shingle_size)
    return PreprocessedDocument(
        raw_text=text if isinstance(text, str) else str(text),
        normalized_text=normalized,
        tokens=tokens,
        shingles=shingles,
    )
