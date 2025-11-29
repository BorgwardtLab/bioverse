import random
import struct
import zlib

import awkward as ak
import numpy as np
import pandas as pd
from rdkit import Chem

from ..adapter import Adapter
from ..data import Assets, Split
from ..utilities import IteratorWithLength, batched, config, download


class BetaLactamAdapter(Adapter):

    @classmethod
    def download(cls):
        from chembl_webresource_client.new_client import new_client

        molecules = fetch_antibiotic_molecules()

        assert (
            len(molecules) > 0
        ), "No molecules retrieved. Check network connectivity or ChEMBL availability."

        pos, neg = split_by_beta_lactam(molecules)
        rng = np.random.default_rng(0)
        rng.shuffle(pos)
        rng.shuffle(neg)

        total = len(pos) + len(neg)
        n_val, n_test = total // 20, total // 20
        n_train_pos, n_train_neg = len(pos) - n_val - n_test, len(neg) - n_val - n_test

        df = make_df(pos, neg)

        split = np.array(
            [[1]] * n_val
            + [[2]] * n_test
            + [[0]] * n_train_pos
            + [[1]] * n_val
            + [[2]] * n_test
            + [[0]] * n_train_neg
        )
        perm = np.random.permutation(len(split))
        split, df = split[perm], df.iloc[perm]

        # create records
        def generator():
            for smiles, id, name, beta_lactam in zip(
                df["smiles"], df["chembl_id"], df["name"], df["beta_lactam"]
            ):
                data = {
                    "molecule_smiles": [[smiles]],
                    "molecule_id": [[id]],
                    "molecule_name": [[name]],
                    "molecule_label": [[beta_lactam]],
                }
                yield ak.Record(data)

        batches = batched(IteratorWithLength(generator(), len(df)))
        return batches, Split(split, names=["train", "val", "test"]), Assets({})


def iter_unique_molecules(records):
    seen: set = set()
    for rec in records:
        chembl_id = rec.get("molecule_chembl_id")
        if not chembl_id:
            continue
        if chembl_id in seen:
            continue
        seen.add(chembl_id)
        yield rec


def fetch_antibiotic_molecules():
    """
    Fetch molecules classified as antibiotics from ChEMBL using multiple strategies:
    1. ATC J01 (Antibacterials for systemic use)
    2. Molecules with synonyms containing antibiotic-related terms
    3. Molecules with mechanism of action related to antibiotics
    4. Molecules indicated for bacterial infections
    """
    from chembl_webresource_client.new_client import new_client

    molecule = new_client.molecule
    atc = new_client.atc_class
    fields = [
        "molecule_chembl_id",
        "pref_name",
        "atc_classifications",
        "molecule_structures",
    ]
    results = []
    seen_mol_ids = set()

    def add_molecules(mol_list):
        """Helper to add molecules while tracking seen IDs."""
        for rec in mol_list:
            mol_id = rec.get("molecule_chembl_id")
            if mol_id and mol_id not in seen_mol_ids:
                seen_mol_ids.add(mol_id)
                results.append(rec)

    # Strategy 1: ATC J01 (Antibacterials for systemic use)
    try:
        level5_codes = [
            rec["level5"] for rec in atc.filter(level2="J01") if rec.get("level5")
        ]
        for code in level5_codes:
            try:
                res = molecule.filter(
                    atc_classifications=code,
                    molecule_structures__isnull=False,
                ).only(fields)
                add_molecules(list(res))
            except Exception:
                continue
    except Exception:
        pass

    # Strategy 2: Molecules with synonyms containing antibiotic-related terms
    synonym_terms = [
        "antibiotic",
        "antibacterial",
        "antimicrobial",
        "bactericidal",
        "bacteriostatic",
    ]
    try:
        synonym = new_client.molecule_synonym
        synonym_mol_ids = set()
        for term in synonym_terms:
            try:
                synonym_results = synonym.filter(synonym__icontains=term)
                for syn_rec in synonym_results:
                    mol_id = syn_rec.get("molecule_chembl_id")
                    if mol_id:
                        synonym_mol_ids.add(mol_id)
            except Exception:
                continue

        # Fetch full molecule records for synonym matches
        for mol_id in synonym_mol_ids:
            if mol_id in seen_mol_ids:
                continue
            try:
                mol_rec = molecule.filter(
                    molecule_chembl_id=mol_id,
                    molecule_structures__isnull=False,
                ).only(fields)
                add_molecules(list(mol_rec))
            except Exception:
                continue
    except Exception:
        pass

    # Strategy 3: Molecules with mechanism of action related to antibiotics
    try:
        mechanism = new_client.mechanism
        # Search for mechanisms with antibiotic-related terms
        moa_terms = ["antibiotic", "antibacterial", "bacterial", "bactericidal"]
        mechanism_mol_ids = set()
        for term in moa_terms:
            try:
                mech_results = mechanism.filter(mechanism_of_action__icontains=term)
                for mech_rec in mech_results:
                    mol_id = mech_rec.get("molecule_chembl_id")
                    if mol_id:
                        mechanism_mol_ids.add(mol_id)
            except Exception:
                continue

        # Fetch full molecule records
        for mol_id in mechanism_mol_ids:
            if mol_id in seen_mol_ids:
                continue
            try:
                mol_rec = molecule.filter(
                    molecule_chembl_id=mol_id,
                    molecule_structures__isnull=False,
                ).only(fields)
                add_molecules(list(mol_rec))
            except Exception:
                continue
    except Exception:
        pass

    # Strategy 4: Molecules indicated for bacterial infections
    try:
        indication = new_client.drug_indication
        # Search for indications related to bacterial infections
        indication_terms = ["bacterial infection", "bacteremia", "sepsis", "pneumonia"]
        indication_mol_ids = set()
        for term in indication_terms:
            try:
                ind_results = indication.filter(efo_term__icontains=term)
                for ind_rec in ind_results:
                    mol_id = ind_rec.get("molecule_chembl_id")
                    if mol_id:
                        indication_mol_ids.add(mol_id)
            except Exception:
                continue

        # Fetch full molecule records
        for mol_id in indication_mol_ids:
            if mol_id in seen_mol_ids:
                continue
            try:
                mol_rec = molecule.filter(
                    molecule_chembl_id=mol_id,
                    molecule_structures__isnull=False,
                ).only(fields)
                add_molecules(list(mol_rec))
            except Exception:
                continue
    except Exception:
        pass

    # Keep only those with a canonical SMILES
    filtered = []
    for rec in results:
        structs = rec.get("molecule_structures") or {}
        smi = structs.get("canonical_smiles")
        if smi:
            filtered.append(rec)
    return filtered


def get_smiles(rec):
    structs = rec.get("molecule_structures") or {}
    return structs.get("canonical_smiles")


def has_beta_lactam(smiles, motif):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return mol.HasSubstructMatch(motif)


def split_by_beta_lactam(molecules):
    # SMARTS for β-lactam: 4-membered cyclic amide core
    # We use a permissive core: N1C(=O)CC1 to capture substitutions
    beta_lactam_smarts = "N1C(=O)CC1"
    motif = Chem.MolFromSmarts(beta_lactam_smarts)
    if motif is None:
        raise RuntimeError("Failed to compile β-lactam SMARTS pattern.")
    positives = []
    negatives = []
    for rec in molecules:
        smi = get_smiles(rec)
        if not smi:
            continue
        if has_beta_lactam(smi, motif):
            positives.append(rec)
        else:
            negatives.append(rec)
    return positives, negatives


def sample_balanced(positives, negatives, n_pos, n_neg, seed):
    random.Random(seed).shuffle(positives)
    random.Random(seed + 1).shuffle(negatives)
    n_pos_avail = len(positives)
    n_neg_avail = len(negatives)
    if n_pos_avail < n_pos or n_neg_avail < n_neg:
        raise RuntimeError(
            f"Not enough molecules to satisfy requested counts. Requested pos={n_pos}, neg={n_neg}, available pos={n_pos_avail}, neg={n_neg_avail}"
        )
    return positives[: min(n_pos, n_pos_avail)], negatives[: min(n_neg, n_neg_avail)]


def make_df(positives, negatives):
    rows = []
    for rec in positives:
        rows.append(
            {
                "chembl_id": rec.get("molecule_chembl_id"),
                "name": rec.get("pref_name"),
                "smiles": get_smiles(rec),
                "beta_lactam": 1,
            }
        )
    for rec in negatives:
        rows.append(
            {
                "chembl_id": rec.get("molecule_chembl_id"),
                "name": rec.get("pref_name"),
                "smiles": get_smiles(rec),
                "beta_lactam": 0,
            }
        )
    df = pd.DataFrame(rows, columns=["chembl_id", "name", "smiles", "beta_lactam"])
    return df
