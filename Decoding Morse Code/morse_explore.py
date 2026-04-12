#!/usr/bin/env python3
"""
Explore morse_code.txt under several hypotheses.

Assumption: ciphertext starts with Morse for THE. Candidates use only words from
project dictionary.txt (default --dict). Zipf ranking is a heuristic; a real
decode must still read as a meaningful English sentence (human check).
Requires: pip install wordfreq
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from wordfreq import zipf_frequency
except ImportError:
    print("Install dependency: pip install wordfreq", file=sys.stderr)
    sys.exit(1)

MORSE_LETTER = {
    "A": ".-",
    "B": "-...",
    "C": "-.-.",
    "D": "-..",
    "E": ".",
    "F": "..-.",
    "G": "--.",
    "H": "....",
    "I": "..",
    "J": ".---",
    "K": "-.-",
    "L": ".-..",
    "M": "--",
    "N": "-.",
    "O": "---",
    "P": ".--.",
    "Q": "--.-",
    "R": ".-.",
    "S": "...",
    "T": "-",
    "U": "..-",
    "V": "...-",
    "W": ".--",
    "X": "-..-",
    "Y": "-.--",
    "Z": "--..",
}

THE_MORSE = "".join(MORSE_LETTER[c] for c in "THE")

SHORT_OK = {
    "A",
    "I",
    "O",
    "AN",
    "AS",
    "AT",
    "BE",
    "DO",
    "GO",
    "HE",
    "IF",
    "IN",
    "IS",
    "IT",
    "ME",
    "MY",
    "NO",
    "OF",
    "ON",
    "OR",
    "SO",
    "TO",
    "UP",
    "US",
    "WE",
    "AM",
    "ID",
}


def enc(word: str) -> str:
    return "".join(MORSE_LETTER[c] for c in word.upper())


def load_words(dict_path: Path) -> dict[str, list[str]]:
    words_by_morse: dict[str, list[str]] = {}
    with dict_path.open() as f:
        for line in f:
            w = line.strip().upper()
            if not w.isalpha() or len(w) > 25:
                continue
            words_by_morse.setdefault(enc(w), []).append(w)
    return words_by_morse


def viterbi_concat(
    s: str,
    words_by_morse: dict[str, list[str]],
    ok_word,
) -> tuple[list[str], float] | None:
    """Maximize sum of zipf scores; words tile s with no delimiters."""
    n = len(s)
    neg = -1e300
    best_score: list[float] = [neg] * (n + 1)
    back: list[tuple[int, str] | None] = [None] * (n + 1)
    best_score[n] = 0.0

    for i in range(n - 1, -1, -1):
        best_sc = neg
        best_bp: tuple[int, str] | None = None
        for j in range(i + 1, min(i + 100, n + 1)):
            chunk = s[i:j]
            if chunk not in words_by_morse:
                continue
            for w in words_by_morse[chunk]:
                if not ok_word(w):
                    continue
                z = zipf_frequency(w.lower(), "en")
                tot = z + best_score[j]
                if tot > best_sc:
                    best_sc = tot
                    best_bp = (j, w)
        best_score[i] = best_sc
        back[i] = best_bp

    if best_score[0] <= neg + 1:
        return None
    out: list[str] = []
    i = 0
    while i < n:
        j, w = back[i]  # type: ignore[misc]
        out.append(w)
        i = j
    return out, best_score[0]


def viterbi_delims(
    s: str,
    words_by_morse: dict[str, list[str]],
    next_delims,
    ok_word,
) -> tuple[list[str], float] | None:
    n = len(s)
    neg = -1e300
    best_score: list[float] = [neg] * (n + 1)
    back: list[tuple[int, str, int] | None] = [None] * (n + 1)
    best_score[n] = 0.0

    for i in range(n - 1, -1, -1):
        best_sc = neg
        best_bp: tuple[int, str, int] | None = None
        for j in range(i + 1, min(i + 100, n + 1)):
            chunk = s[i:j]
            if chunk not in words_by_morse:
                continue
            for w in words_by_morse[chunk]:
                if not ok_word(w):
                    continue
                z = zipf_frequency(w.lower(), "en")
                if j == n:
                    tot = z + best_score[n]
                    if tot > best_sc:
                        best_sc = tot
                        best_bp = (j, w, n)
                else:
                    for nxt in next_delims(j, s, n):
                        if best_score[nxt] > neg + 1:
                            tot = z + best_score[nxt]
                            if tot > best_sc:
                                best_sc = tot
                                best_bp = (j, w, nxt)
        best_score[i] = best_sc
        back[i] = best_bp

    if best_score[0] <= neg + 1:
        return None
    out: list[str] = []
    i = 0
    while i < n:
        j, w, nxt = back[i]  # type: ignore[misc]
        out.append(w)
        i = nxt
    return out, best_score[0]


def next_delims_a(j: int, s: str, n: int) -> list[int]:
    out: list[int] = []
    if j >= n:
        return out
    if s[j] == ".":
        out.append(j + 1)
    if j + 2 <= n and s[j : j + 2] == "--":
        out.append(j + 2)
    return out


def next_delims_b(j: int, s: str, n: int) -> list[int]:
    out: list[int] = []
    if j >= n:
        return out
    if j + 2 <= n and s[j : j + 2] == "..":
        out.append(j + 2)
    if s[j] == "-":
        out.append(j + 1)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Explore Morse hypotheses for morse_code.txt")
    p.add_argument(
        "--file",
        type=Path,
        default=Path(__file__).resolve().parent / "morse_code.txt",
        help="Path to ciphertext file",
    )
    p.add_argument(
        "--dict",
        type=Path,
        default=Path(__file__).resolve().parent / "dictionary.txt",
        help="Word list (default: dictionary.txt next to this script)",
    )
    p.add_argument(
        "--mode",
        choices=("remainder", "full", "delims-a", "delims-b"),
        default="remainder",
        help="remainder=THE+decode rest; full=decode whole string; delims-*=word-space rules",
    )
    p.add_argument(
        "--min-len",
        type=int,
        default=4,
        metavar="N",
        help="Min word length on decoded segment (THE always allowed when mode=remainder)",
    )
    p.add_argument("--no-short-ok", action="store_true", help="Do not allow SHORT_OK for short words")
    p.add_argument(
        "--min-zipf",
        type=float,
        default=None,
        metavar="Z",
        help=(
            "If set, only allow words whose English Zipf score (wordfreq) is >= Z. "
            "Typical scale: ~7 very common (THE), ~4 common, ~2 rare. "
            "Narrows to 'more commonly occurring' words in English corpora."
        ),
    )

    args = p.parse_args()
    if not args.dict.is_file():
        print(f"Dictionary not found: {args.dict}", file=sys.stderr)
        sys.exit(1)

    raw = args.file.read_text().strip()
    if not raw.startswith(THE_MORSE):
        print("File does not start with standard Morse for THE.", file=sys.stderr)
        sys.exit(1)

    rest = raw[len(THE_MORSE) :]
    print(f"Assumed THE morse length: {len(THE_MORSE)}")
    print(f"Remainder length: {len(rest)}")
    if args.min_zipf is not None:
        print(f"English Zipf floor (wordfreq): >= {args.min_zipf} (more common words only)")
    print(f"Remainder:\n{rest}\n")

    words_by_morse = load_words(args.dict)

    def ok_word(w: str) -> bool:
        if args.no_short_ok:
            ok = len(w) >= args.min_len
        else:
            ok = len(w) >= args.min_len or w in SHORT_OK
        if not ok:
            return False
        if args.min_zipf is not None:
            if zipf_frequency(w.lower(), "en") < args.min_zipf:
                return False
        return True

    if args.mode == "remainder":
        r = viterbi_concat(rest, words_by_morse, ok_word)
        if not r:
            print("No tiling found for remainder with current filters.")
            sys.exit(2)
        words, score = r
        print("Decode: THE +" , " ".join(words))
        print(f"Zipf sum (approx): {score:.3f}")
        cat = "".join(enc(w) for w in words)
        print(f"Re-encode rest matches: {cat == rest}")
        print(
            "Note: All tokens are from the given dictionary; still verify the line reads as a meaningful sentence.",
        )
        return

    if args.mode == "full":
        r = viterbi_concat(raw, words_by_morse, ok_word)
        if not r:
            print("No tiling found for full string with current filters.")
            sys.exit(2)
        words, score = r
        print("Decode:", " ".join(words))
        print(f"Zipf sum (approx): {score:.3f}")
        cat = "".join(enc(w) for w in words)
        print(f"Re-encode full matches: {cat == raw}")
        print(
            "Note: All tokens are from the given dictionary; still verify the line reads as a meaningful sentence.",
        )
        return

    if args.mode == "delims-a":
        nd = next_delims_a
        label = "Scheme A: space = . OR --"
    else:
        nd = next_delims_b
        label = "Scheme B: space = .. OR -"

    r = viterbi_delims(raw, words_by_morse, nd, ok_word)
    print(label)
    if not r:
        print("No tiling found with current filters.")
        sys.exit(2)
    words, score = r
    print("Decode:", " ".join(words))
    print(f"Zipf sum (approx): {score:.3f}")
    return


if __name__ == "__main__":
    main()
