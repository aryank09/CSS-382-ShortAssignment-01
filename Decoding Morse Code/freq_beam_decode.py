#!/usr/bin/env python3
"""
Frequency-guided Morse decode after fixed THE.

1) **Local score** (word + phrase bigram, wordfreq Zipf):
     sum_w [ zipf(w) + bigram_weight * zipf(prev + " " + w) ]
   starting with prev = "THE".

2) **Full-line Zipf**: zipf_frequency("the " + words..., "en") — strong signal for
   natural English; penalizes high-unigram junk (TEST, TWAT, ...).

3) **N-best DP** over reachable (position, last_word) states (graph is a DAG).

Requires: pip install wordfreq
"""

from __future__ import annotations

import argparse
import heapq
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


def word_zipf(w: str) -> float:
    return zipf_frequency(w.lower(), "en")


def bigram_zipf(prev: str, w: str) -> float:
    return zipf_frequency(f"{prev.lower()} {w.lower()}", "en")


def full_line_zipf(words: list[str]) -> float:
    return zipf_frequency("the " + " ".join(w.lower() for w in words), "en")


def nbest_dp(
    rest: str,
    words_by_morse: dict[str, list[str]],
    *,
    prev0: str,
    bigram_w: float,
    min_len: int,
    short_ok: set[str],
    allow_extra: set[str],
    ban: set[str],
    topk: int,
) -> list[tuple[float, list[str]]]:
    """Return up to `topk` complete paths (local_score, words), best local first."""
    n = len(rest)

    def ok_word(w: str) -> bool:
        if w in ban:
            return False
        return len(w) >= min_len or w in short_ok or w in allow_extra

    # best_partial[(pos, last)] = min-heap of (score, words_tuple) keeping K largest scores
    heaps: dict[tuple[int, str], list] = {}

    def push(pos: int, last: str, sc: float, words: tuple[str, ...]) -> None:
        key = (pos, last)
        h = heaps.setdefault(key, [])
        item = (sc, words)
        if len(h) < topk:
            heapq.heappush(h, item)
        elif sc > h[0][0]:
            heapq.heapreplace(h, item)

    push(0, prev0, 0.0, ())

    # Process positions in order (DAG by increasing pos)
    for i in range(n + 1):
        keys_at_i = [k for k in heaps if k[0] == i]
        for key in keys_at_i:
            h = heaps[key]
            if not h:
                continue
            _pos, last = key
            for sc, words_t in list(h):
                for j in range(i + 1, min(i + 100, n + 1)):
                    chunk = rest[i:j]
                    if chunk not in words_by_morse:
                        continue
                    for w in words_by_morse[chunk]:
                        if not ok_word(w):
                            continue
                        z = word_zipf(w)
                        b = bigram_zipf(last, w)
                        nsc = sc + z + bigram_w * b
                        push(j, w, nsc, words_t + (w,))

    out: list[tuple[float, list[str]]] = []
    for (pos, last), h in heaps.items():
        if pos != n or not h:
            continue
        for sc, words_t in h:
            out.append((sc, list(words_t)))

    out.sort(key=lambda x: -x[0])
    # dedupe (keep all distinct complete paths — needed when sorting by full-line Zipf)
    seen: set[tuple[str, ...]] = set()
    deduped: list[tuple[float, list[str]]] = []
    for sc, ws in out:
        t = tuple(ws)
        if t in seen:
            continue
        seen.add(t)
        deduped.append((sc, ws))
    return deduped


def main() -> None:
    p = argparse.ArgumentParser(description="N-best Morse decode with Zipf + full-line rescoring")
    p.add_argument("--file", type=Path, default=Path(__file__).resolve().parent / "morse_code.txt")
    p.add_argument("--dict", type=Path, default=Path(__file__).resolve().parent / "dictionary.txt")
    p.add_argument(
        "--topk",
        type=int,
        default=40,
        help="Per-state beam: keep this many best partial paths at each (position, last_word)",
    )
    p.add_argument(
        "--show",
        type=int,
        default=25,
        help="How many lines to print after sorting",
    )
    p.add_argument("--bigram-weight", type=float, default=1.0, metavar="W", help="Weight on phrase Zipf")
    p.add_argument("--min-len", type=int, default=4)
    p.add_argument("--no-short-ok", action="store_true")
    p.add_argument(
        "--allow-extra",
        default="",
        help="Comma-separated words allowed even if shorter than --min-len (e.g. IS,FAR)",
    )
    p.add_argument(
        "--ban",
        default="",
        help="Comma-separated words to forbid (e.g. IT to reduce Morse spam)",
    )
    p.add_argument(
        "--sort",
        choices=("local", "full"),
        default="full",
        help="Rank paths by local DP score or full-line phrase Zipf",
    )
    p.add_argument(
        "--reference-sentence",
        help='Optional uppercase words after THE, e.g. "IMPRESSION IS FAR MORE IMPORTANT BEACH REALITY" — print Zipf scores and exit',
    )
    args = p.parse_args()

    raw = args.file.read_text().strip()
    if not raw.startswith(THE_MORSE):
        print("File does not start with Morse for THE.", file=sys.stderr)
        sys.exit(1)

    rest = raw[len(THE_MORSE) :]

    if args.reference_sentence:
        ref_words = [w.strip().upper() for w in args.reference_sentence.split() if w.strip()]
        cat = "".join(enc(w) for w in ref_words)
        if cat != rest:
            print("Reference sentence does not re-encode the remainder (wrong tiling).", file=sys.stderr)
            print(f"  expected len {len(rest)} got {len(cat)}", file=sys.stderr)
            sys.exit(2)
        prev = "THE"
        loc = 0.0
        for w in ref_words:
            loc += word_zipf(w) + args.bigram_weight * bigram_zipf(prev, w)
            prev = w
        fz = full_line_zipf(ref_words)
        print("Reference (fixed THE):")
        print("  THE " + " ".join(ref_words))
        print(f"  local (word + bigram phrase Zipf): {loc:.3f}")
        print(f"  full-line phrase Zipf: {fz:.3f}")
        return
    words_by_morse = load_words(args.dict)

    short_ok = set() if args.no_short_ok else {
        "A", "I", "O", "AN", "AS", "AT", "BE", "DO", "GO", "HE", "IF", "IN", "IS", "IT",
        "ME", "MY", "NO", "OF", "ON", "OR", "SO", "TO", "UP", "US", "WE", "AM", "ID",
    }
    allow_extra = {x.strip().upper() for x in args.allow_extra.split(",") if x.strip()}
    ban = {x.strip().upper() for x in args.ban.split(",") if x.strip()}

    paths = nbest_dp(
        rest,
        words_by_morse,
        prev0="THE",
        bigram_w=args.bigram_weight,
        min_len=args.min_len,
        short_ok=short_ok,
        allow_extra=allow_extra,
        ban=ban,
        topk=args.topk,
    )

    enriched: list[tuple[float, float, float, list[str]]] = []
    for sc, ws in paths:
        fz = full_line_zipf(ws)
        combined = 0.5 * sc + 0.5 * fz
        enriched.append((sc, fz, combined, ws))

    if args.sort == "full":
        enriched.sort(key=lambda x: -x[1])
    elif args.sort == "local":
        enriched.sort(key=lambda x: -x[0])
    else:
        enriched.sort(key=lambda x: -x[2])

    print(f"Remainder length: {len(rest)}")
    print(
        f"bigram_weight={args.bigram_weight} min_len={args.min_len} "
        f"short_ok={not args.no_short_ok} sort={args.sort}\n"
    )
    print("Columns: local = sum word Zipf + bigram_weight * phrase Zipf; full = full-line phrase Zipf")
    print()

    for rank, (loc, fz, _comb, ws) in enumerate(enriched[: args.show], 1):
        line = "THE " + " ".join(ws)
        cat = "".join(enc(w) for w in ws)
        ok = cat == rest
        print(f"{rank:2d}. local={loc:.3f}  full={fz:.3f}  match={ok}")
        print(f"    {line}")
        print()


if __name__ == "__main__":
    main()
