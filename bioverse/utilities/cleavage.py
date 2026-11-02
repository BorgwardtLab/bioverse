"""P4–P4′ (and optional P6–P5) cleavage-window helpers for protease tasks."""

from __future__ import annotations

import re

from .constants import PROTEIN_ALPHABET, THREE_TO_ONE

WINDOW_SITES = ("P6", "P5", "P4", "P3", "P2", "P1", "P1'", "P2'", "P3'", "P4'")
WINDOW_LEN = len(WINDOW_SITES)
P4P4_SITES = WINDOW_SITES[2:]
CANONICAL = set(PROTEIN_ALPHABET)
GAP = "-"
UNK_INDEX = len(PROTEIN_ALPHABET)
AA_INDEX = {aa: i for i, aa in enumerate(PROTEIN_ALPHABET)}
SWISS_PROT = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9])$"
)


def residue_letter(token) -> str:
    """Map a MEROPS / PDB residue token to a canonical 1-letter code or gap."""
    if token is None:
        return GAP
    try:
        if token != token:  # NaN from unmapped PDB residue names
            return GAP
    except (TypeError, ValueError):
        pass
    text = str(token).strip().strip("'").strip('"')
    if text.upper() in {"", "NULL", "NONE", "NAN", "-", "X", "XXX"}:
        return GAP
    if len(text) == 1:
        letter = text.upper()
        return letter if letter in CANONICAL else GAP
    code = text[:3].upper()
    if code in THREE_TO_ONE:
        return THREE_TO_ONE[code]
    return GAP


def pad_window(window: str, length: int = WINDOW_LEN, align: str = "p4") -> str:
    """Pad a cleavage motif to ``length`` sites.

    ``align="p4"`` left-pads so a P4–P4′ octamer occupies P4–P4′ of a
    P6–P4′ window. ``align="left"`` right-pads (Huber P6–P1′ heptamers).
    """
    letters = [residue_letter(ch) for ch in str(window)]
    letters = [ch if ch in CANONICAL else GAP for ch in letters]
    if len(letters) > length:
        letters = letters[:length]
    if len(letters) < length:
        pad = [GAP] * (length - len(letters))
        letters = pad + letters if align == "p4" else letters + pad
    return "".join(letters)


def merops_p4p4_window(residues) -> str:
    """Build a P6–P4′ window from eight P4–P4′ MEROPS residue tokens."""
    core = "".join(residue_letter(token) for token in list(residues)[:8])
    return pad_window(core, align="p4")


def huber_window(tevs: str, length: int = WINDOW_LEN) -> str:
    """Pad a TEV heptamer (P6–P1′) into the shared P6–P4′ window."""
    return pad_window("".join(residue_letter(ch) for ch in tevs), length, align="left")


def tokenize_window(window: str, length: int = WINDOW_LEN) -> list[int]:
    padded = pad_window(window, length, align="left") if len(window) != length else window
    if len(padded) != length:
        padded = pad_window(window, length, align="left")
    return [AA_INDEX.get(aa, UNK_INDEX) for aa in padded]


def canonical_count(window: str) -> int:
    return sum(ch in CANONICAL for ch in window)


def prefer_uniprot(ids) -> str | None:
    """Pick a Swiss-Prot-like accession, otherwise the first non-empty id."""
    cleaned = []
    for item in ids:
        text = str(item).strip().strip('"').strip("'")
        if text and text not in {"-", "NULL", "None"}:
            cleaned.append(text)
    if not cleaned:
        return None
    swiss = [item for item in cleaned if SWISS_PROT.match(item)]
    short = [item for item in swiss if len(item) == 6]
    return (short or swiss or cleaned)[0]


def allowed_letters(token: str) -> list[str] | None:
    """Parse a combinatorial-profile cell (``L/Y``, ``-`` = unrestricted)."""
    text = str(token).strip().strip("'").strip('"')
    if text in {"", "-", "NULL", "None"}:
        return None
    letters = []
    for part in text.split("/"):
        letter = residue_letter(part)
        if letter in CANONICAL:
            letters.append(letter)
    return letters or None
