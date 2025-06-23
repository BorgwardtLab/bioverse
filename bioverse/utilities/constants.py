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
