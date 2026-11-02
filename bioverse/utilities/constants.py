import numpy as np

MASKING_TOKEN = "?"
SOS_TOKEN = "<"
EOS_TOKEN = ">"
UNK_TOKEN = "%"
PADDING_TOKEN = "-"
TOKENS = MASKING_TOKEN + SOS_TOKEN + EOS_TOKEN + UNK_TOKEN + PADDING_TOKEN
PROTEIN_ALPHABET = "ARNDCQEGHILKMFPSTWYV"
CHAIN_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DNA_ALPHABET = "ACGT"
RNA_ALPHABET = "ACGU"
SMALL_MOLECULE_ALPHABET = "HCNOFS"
ION_ALPHABET = "K"
SHARD_SIZE = 1000
SHARD_BUFFER_SIZE = 4
HARTREE_TO_EV = 27.211386246
THREE_TO_ONE = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}
ONE_TO_THREE = {k: v for v, k in THREE_TO_ONE.items()}
CHEMICAL_GROUPS = {
    aa: g
    for aa, g in zip(
        PROTEIN_ALPHABET,
        [
            "hydrophobic",
            "charged",
            "polar",
            "charged",
            "polar",
            "polar",
            "charged",
            "polar",
            "charged",
            "hydrophobic",
            "hydrophobic",
            "charged",
            "hydrophobic",
            "hydrophobic",
            "hydrophobic",
            "polar",
            "polar",
            "hydrophobic",
            "polar",
            "hydrophobic",
        ],
    )
}
ATOM_ALPHABET = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]
_ELEMENT_BY_UPPER = {el.upper(): el for el in ATOM_ALPHABET}


def normalize_element(symbol: str) -> str:
    """Canonical element symbol (``FE`` / ``fe`` → ``Fe``)."""
    text = str(symbol).strip()
    if not text:
        return "X"
    return _ELEMENT_BY_UPPER.get(text.upper(), text[0].upper() + text[1:].lower())


def element_from_atom_name(name: str) -> str:
    """Element symbol from a PDB atom name or element label.

    Two-letter symbols are accepted only when the second character is
    lowercase (``Fe``), so ``CA`` is carbon rather than calcium.
    """
    text = str(name).strip()
    i = 0
    while i < len(text) and text[i].isdigit():
        i += 1
    text = text[i:]
    if not text:
        return "X"
    if len(text) >= 2 and text[0].isalpha() and text[1].islower():
        pair = text[0].upper() + text[1].lower()
        if pair.upper() in _ELEMENT_BY_UPPER:
            return _ELEMENT_BY_UPPER[pair.upper()]
    first = text[0].upper()
    return _ELEMENT_BY_UPPER.get(first, first)


BIOCHEMICAL_ATOM_ALPHABET = [
    # Major elements (bulk of biomolecules)
    "H",  # Hydrogen
    "C",  # Carbon
    "N",  # Nitrogen
    "O",  # Oxygen
    "P",  # Phosphorus
    "S",  # Sulfur
    # Major ions / Electrolytes
    "Na",  # Sodium
    "K",  # Potassium
    "Mg",  # Magnesium
    "Ca",  # Calcium
    "Cl",  # Chlorine
    # Transition metals (enzyme cofactors)
    "Fe",  # Iron
    "Zn",  # Zinc
    "Cu",  # Copper
    "Mn",  # Manganese
    "Co",  # Cobalt
    "Mo",  # Molybdenum
    "Ni",  # Nickel
    # Trace elements
    "Se",  # Selenium
    "I",  # Iodine
    # Occasionally relevant or context-specific
    "B",  # Boron (plant metabolism, some signaling)
    "F",  # Fluorine (teeth, not essential)
    "Cr",  # Chromium (glucose metabolism – debated)
    "V",  # Vanadium (some marine or lower organisms)
]

# fmt: off
BLOSUM62 = np.array(
    [
        # A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V
        [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
        [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],
        [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
        [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
        [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
        [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],
        [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
        [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
        [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
        [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],
        [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],
        [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],
        [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],
        [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],
        [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],
        [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],
        [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
        [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],
        [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],
        [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],
    ],
    dtype=float,
)
# fmt: on
