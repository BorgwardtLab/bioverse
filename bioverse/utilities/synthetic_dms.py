"""Compose synthetic protease DMS labels from ProteinGym, BioLiP, and MEROPS priors.

The generator is modular: mutation / pocket / substrate priors are independent
and a registered composer turns them into ``Δactivity``. Huber 2025 is never
read here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from .constants import BLOSUM62, PROTEIN_ALPHABET
from .cleavage import AA_INDEX, WINDOW_LEN, residue_letter

AA_CLASS = {
    "A": "hydrophobic",
    "I": "hydrophobic",
    "L": "hydrophobic",
    "M": "hydrophobic",
    "V": "hydrophobic",
    "F": "aromatic",
    "W": "aromatic",
    "Y": "aromatic",
    "S": "polar",
    "T": "polar",
    "N": "polar",
    "Q": "polar",
    "D": "negative",
    "E": "negative",
    "K": "positive",
    "R": "positive",
    "H": "positive",
    "C": "special",
    "G": "special",
    "P": "special",
}
BURIAL_BINS = ("buried", "partial", "exposed")
BLOSUM_BINS = ("conservative", "moderate", "radical")
FORMULATIONS = (
    "mutation",
    "mutation_pocket",
    "mutation_pocket_substrate",
    "attribution_mutation",
    "pocket_attribution_mutation",
)
# Additive Huber-like ablations. The product q*a*e is retained only for
# comparison; it is not the default predictor.
ZERO_SHOT_FORMULATIONS = (
    "activity",
    "activity_rewire",
    "rewire",
)
PRODUCT_FORMULATIONS = (
    "mutation",
    "attribution_mutation",
    "pocket_attribution_mutation",
)
N_STRATUM_KEEP = 512
CHARGE = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.5}
HELD_OUT_FAMILIES = frozenset({"C04", "M10"})


def chemistry_class(aa: str) -> str:
    return AA_CLASS.get(str(aa).upper(), "special")


def chemistry_pair_score(pocket: str, substrate: str) -> float:
    """Signed pocket–peptide chemistry. Opposite charge and like-with-like win."""
    pocket = str(pocket).upper()
    substrate = str(substrate).upper()
    score = 0.0
    cq = CHARGE.get(pocket, 0.0)
    cs = CHARGE.get(substrate, 0.0)
    if cq and cs:
        score -= 1.25 * cq * cs
    pc = chemistry_class(pocket)
    sc = chemistry_class(substrate)
    if pc == sc and pc in {"hydrophobic", "aromatic", "polar"}:
        score += 0.75
    if {pc, sc} == {"hydrophobic", "aromatic"}:
        score += 0.5
    if pc in {"hydrophobic", "aromatic"} and sc in {"positive", "negative"}:
        score -= 0.4
    if pc in {"positive", "negative"} and sc in {"hydrophobic", "aromatic"}:
        score -= 0.4
    return float(score)


def softplus_gate(values, axis=None) -> np.ndarray:
    """Non-negative attribution gate. Does not z-score (that flips signs)."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    gated = np.log1p(np.exp(np.clip(arr, -40.0, 40.0)))
    if axis is None:
        center = float(np.nanmean(gated))
        scale = center if center > 1e-8 else 1.0
        return gated / scale
    center = np.nanmean(gated, axis=axis, keepdims=True)
    scale = np.where(center < 1e-8, 1.0, center)
    return gated / scale


def sigmoid_auc(latent, temperature: float = 1.0) -> np.ndarray:
    """Map a latent competence/recognition score to a Huber-like AUC."""
    z = np.asarray(latent, dtype=np.float64) / max(float(temperature), 1e-6)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -12.0, 12.0)))


def blosum_score(wt: str, mut: str) -> float:
    i = AA_INDEX.get(str(wt).upper())
    j = AA_INDEX.get(str(mut).upper())
    if i is None or j is None:
        return 0.0
    return float(BLOSUM62[i, j])


def blosum_bin(score: float) -> str:
    if score >= 0:
        return "conservative"
    if score >= -2:
        return "moderate"
    return "radical"


def neighbor_counts(coords: np.ndarray, radius: float = 10.0) -> np.ndarray:
    xyz = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    if xyz.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    delta = xyz[:, None, :] - xyz[None, :, :]
    dist = np.sqrt(np.sum(delta * delta, axis=-1))
    np.fill_diagonal(dist, np.inf)
    return np.sum(dist < radius, axis=1).astype(np.float64)


def burial_bin(count: float, buried: float = 20.0, exposed: float = 10.0) -> str:
    if count >= buried:
        return "buried"
    if count >= exposed:
        return "partial"
    return "exposed"


def burial_bins(coords: np.ndarray, radius: float = 10.0) -> list[str]:
    return [burial_bin(float(n)) for n in neighbor_counts(coords, radius=radius)]


def cleft_pocket_prob(coords: np.ndarray) -> np.ndarray:
    """Unsupervised pocket-likeness: mid-shell neighbors, moderate burial."""
    xyz = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    n = int(xyz.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    delta = xyz[:, None, :] - xyz[None, :, :]
    dist = np.sqrt(np.sum(delta * delta, axis=-1))
    np.fill_diagonal(dist, np.inf)
    near = np.sum(dist < 8.0, axis=1)
    shell = np.sum((dist >= 8.0) & (dist < 14.0), axis=1)
    near_z = (near - near.mean()) / max(float(near.std()), 1e-6)
    shell_z = (shell - shell.mean()) / max(float(shell.std()), 1e-6)
    # Pocket residues sit in a cleft: not the most buried core, rich mid-shell.
    logits = 0.85 * shell_z - 0.35 * near_z
    return 1.0 / (1.0 + np.exp(-logits))


def robust_scale(values) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.std(values))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    return center, scale


def _unit_interval(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -12.0, 12.0))))


@dataclass
class MutationContext:
    burial: str
    wt_class: str
    mut_class: str
    blosum: str

    @classmethod
    def from_mutation(cls, wt: str, mut: str, burial: str) -> "MutationContext":
        return cls(
            burial=str(burial),
            wt_class=chemistry_class(wt),
            mut_class=chemistry_class(mut),
            blosum=blosum_bin(blosum_score(wt, mut)),
        )

    def keys(self) -> tuple[tuple, ...]:
        return (
            (self.burial, self.wt_class, self.mut_class, self.blosum),
            (self.burial, self.wt_class, self.mut_class),
            (self.burial, self.blosum),
            (self.burial,),
            (),
        )


@dataclass
class MutationEffectPrior:
    """Empirical ProteinGym effect distributions, stratified by context."""

    strata: dict = field(default_factory=dict)
    global_center: float = 0.0
    global_scale: float = 1.0
    n_fit: int = 0

    def fit(self, contexts: list[MutationContext], effects) -> "MutationEffectPrior":
        effects = np.asarray(effects, dtype=np.float64).reshape(-1)
        buckets: dict[tuple, list[float]] = {(): []}
        for ctx, effect in zip(contexts, effects):
            if not np.isfinite(effect):
                continue
            for key in ctx.keys():
                buckets.setdefault(key, []).append(float(effect))
        self.strata = {}
        for key, values in buckets.items():
            if not values:
                continue
            kept = values[:N_STRATUM_KEEP]
            center, scale = robust_scale(kept)
            self.strata[key] = {
                "values": np.asarray(kept, dtype=np.float64),
                "center": center,
                "scale": scale,
                "n": int(len(values)),
            }
        global_stats = self.strata.get((), {"center": 0.0, "scale": 1.0, "n": 0})
        self.global_center = float(global_stats["center"])
        self.global_scale = float(global_stats["scale"])
        self.n_fit = int(global_stats["n"])
        return self

    def _lookup(self, ctx: MutationContext) -> dict:
        for key in ctx.keys():
            stats = self.strata.get(key)
            if stats is not None and stats["n"] >= 8:
                return stats
        return self.strata.get(
            (),
            {
                "values": np.zeros(0, dtype=np.float64),
                "center": self.global_center,
                "scale": self.global_scale,
                "n": 0,
            },
        )

    def expect(self, ctx: MutationContext) -> float:
        """Deterministic stratum location (median). Used as the mutation prior."""
        return float(self._lookup(ctx).get("center", self.global_center))

    def sample(self, ctx: MutationContext, rng: np.random.Generator) -> float:
        stats = self._lookup(ctx)
        values = np.asarray(stats.get("values", []), dtype=np.float64)
        if values.size >= 8 and rng.random() < 0.7:
            return float(rng.choice(values))
        center = float(stats.get("center", self.global_center))
        scale = float(stats.get("scale", self.global_scale))
        draw = float(center + scale * rng.normal())
        lo = center - 3.0 * scale
        hi = center + 3.0 * scale
        return float(np.clip(draw, lo, hi))


@dataclass
class SubstratePwmPrior:
    """Per-protease PWM compatibility on P6–P4′ windows."""

    pwms: dict = field(default_factory=dict)
    background: np.ndarray | None = None

    def fit(self, windows_by_id: dict[str, list[str]]) -> "SubstratePwmPrior":
        all_windows = [w for windows in windows_by_id.values() for w in windows]
        self.background = _pwm_probs(all_windows) if all_windows else None
        self.pwms = {}
        for code, windows in windows_by_id.items():
            if not windows:
                continue
            self.pwms[str(code)] = _pwm_logodds(windows, self.background)
        return self

    def score(self, protease_id: str, window: str) -> float:
        pwm = self.pwms.get(str(protease_id))
        if pwm is None:
            return 0.0
        return _score_pwm(pwm, window)

    def zscore(self, protease_id: str, window: str, pool: list[str]) -> float:
        scores = np.asarray([self.score(protease_id, item) for item in pool], dtype=np.float64)
        center, scale = robust_scale(scores)
        return float((self.score(protease_id, window) - center) / scale)


def _pwm_probs(windows, length: int = WINDOW_LEN) -> np.ndarray:
    counts = np.ones((length, len(PROTEIN_ALPHABET)), dtype=np.float64)
    for window in windows:
        for i, aa in enumerate(str(window)[:length]):
            idx = AA_INDEX.get(aa)
            if idx is not None:
                counts[i, idx] += 1.0
    return counts / counts.sum(axis=1, keepdims=True)


def _pwm_logodds(windows, background=None, length: int = WINDOW_LEN) -> np.ndarray:
    probs = _pwm_probs(windows, length=length)
    if background is None:
        background = np.full_like(probs, 1.0 / probs.shape[1])
    return np.log(np.clip(probs, 1e-8, 1.0)) - np.log(np.clip(background, 1e-8, 1.0))


def _score_pwm(pwm: np.ndarray, window: str) -> float:
    total = 0.0
    for i, aa in enumerate(str(window)[: pwm.shape[0]]):
        idx = AA_INDEX.get(residue_letter(aa) if len(aa) != 1 else aa)
        if idx is None:
            idx = AA_INDEX.get(str(aa).upper())
        if idx is not None and idx < pwm.shape[1]:
            total += float(pwm[i, idx])
    return total


@dataclass
class RewirePrior:
    """MEROPS-informed pocket × substrate substitution operator.

    ``affinity(pocket_AA, S_p, p)`` combines chemistry pairing with a
    family-held-out PWM that only contributes when the pocket residue
    matches the preferred chemistry at position ``p``. Huber / C4 / M10
    windows are never used to fit this table.
    """

    logodds: np.ndarray | None = None
    preferred_class: tuple = ()
    pwm_weight: float = 0.5
    n_fit: int = 0

    def fit(self, windows: list[str], pwm_weight: float = 0.5) -> "RewirePrior":
        kept = [str(window) for window in windows if window]
        self.logodds = _pwm_logodds(kept) if kept else np.zeros(
            (WINDOW_LEN, len(PROTEIN_ALPHABET)), dtype=np.float64
        )
        preferred = []
        for pos in range(self.logodds.shape[0]):
            idx = int(np.argmax(self.logodds[pos]))
            preferred.append(chemistry_class(PROTEIN_ALPHABET[idx]))
        self.preferred_class = tuple(preferred)
        self.pwm_weight = float(pwm_weight)
        self.n_fit = int(len(kept))
        return self

    def affinity(self, pocket: str, substrate: str, pos: int) -> float:
        chem = chemistry_pair_score(pocket, substrate)
        idx = AA_INDEX.get(str(substrate).upper())
        pwm = 0.0
        if (
            self.logodds is not None
            and idx is not None
            and 0 <= int(pos) < self.logodds.shape[0]
        ):
            pwm = float(self.logodds[int(pos), idx])
        match = 0.0
        if 0 <= int(pos) < len(self.preferred_class):
            if chemistry_class(pocket) == self.preferred_class[int(pos)]:
                match = 1.0
        return chem + self.pwm_weight * pwm * match

    def delta_psi(self, wt: str, mut: str, window: str) -> np.ndarray:
        """``ψ(MUT, S_p, p) − ψ(WT, S_p, p)`` for each window position."""
        letters = [residue_letter(ch) if len(ch) != 1 else str(ch).upper() for ch in str(window)]
        n = WINDOW_LEN if self.logodds is None else int(self.logodds.shape[0])
        out = np.zeros(n, dtype=np.float64)
        for pos in range(min(n, len(letters))):
            aa = letters[pos]
            if aa not in AA_INDEX:
                continue
            out[pos] = self.affinity(mut, aa, pos) - self.affinity(wt, aa, pos)
        return out


def gaussian_contacts(distances, sigma: float = 3.5) -> np.ndarray:
    dist = np.asarray(distances, dtype=np.float64)
    return np.exp(-0.5 * np.square(dist / max(float(sigma), 1e-6)))


def contact_from_distance(distances, radius: float = 8.0, sharpness: float = 1.5) -> np.ndarray:
    """Soft contact occupancy from Cα–Cα distance. 0.5 at ``radius`` Å."""
    dist = np.asarray(distances, dtype=np.float64)
    return 1.0 / (1.0 + np.exp((dist - float(radius)) / max(float(sharpness), 1e-6)))


def pocket_from_contacts(contact) -> np.ndarray:
    """Residue pocket probability: max occupancy over peptide positions."""
    arr = np.asarray(contact, dtype=np.float64)
    if arr.ndim == 1:
        return np.clip(arr, 0.0, 1.0)
    if arr.size == 0:
        return np.zeros(0, dtype=np.float64)
    return np.clip(np.max(arr, axis=1), 0.0, 1.0)


def literature_contact_matrix(n_res: int, mapping: dict[int, list[int]], n_pos: int = WINDOW_LEN):
    """Sparse protease-site → peptide-position contacts from literature."""
    contact = np.zeros((int(n_res), int(n_pos)), dtype=np.float64)
    for site, positions in mapping.items():
        if 0 <= int(site) < contact.shape[0]:
            for pos in positions:
                if 0 <= int(pos) < contact.shape[1]:
                    contact[int(site), int(pos)] = 1.0
    return contact


@dataclass
class ComposerConfig:
    formulation: str = "mutation_pocket_substrate"
    pocket_floor: float = 0.15
    substrate_gain: float = 0.75
    noise_scale: float = 0.15

    def __post_init__(self):
        if self.formulation not in FORMULATIONS:
            raise ValueError(
                f"Unknown formulation {self.formulation!r}; expected one of {FORMULATIONS}."
            )


def compose_mutation(mutation_effect, pocket_weight, substrate_z, cfg: ComposerConfig):
    return float(mutation_effect)


def compose_mutation_pocket(mutation_effect, pocket_weight, substrate_z, cfg: ComposerConfig):
    scale = cfg.pocket_floor + (1.0 - cfg.pocket_floor) * float(pocket_weight)
    return float(mutation_effect) * scale


def compose_mutation_pocket_substrate(
    mutation_effect, pocket_weight, substrate_z, cfg: ComposerConfig
):
    pocket = cfg.pocket_floor + (1.0 - cfg.pocket_floor) * float(pocket_weight)
    substrate = 1.0 + cfg.substrate_gain * float(np.tanh(substrate_z))
    return float(mutation_effect) * pocket * substrate


COMPOSERS: dict[str, Callable] = {
    "mutation": compose_mutation,
    "mutation_pocket": compose_mutation_pocket,
    "mutation_pocket_substrate": compose_mutation_pocket_substrate,
}


def register_composer(name: str, fn: Callable) -> None:
    COMPOSERS[name] = fn


def normalize_attribution(values, mode: str = "zscore") -> np.ndarray:
    """Calibrate residue attributions so windows are comparable."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    axis = 1 if arr.ndim > 1 else None
    if mode in (None, "raw", "none"):
        return arr
    if mode == "zscore":
        center = np.nanmean(arr, axis=axis, keepdims=True)
        scale = np.nanstd(arr, axis=axis, keepdims=True)
        scale = np.where(scale < 1e-6, 1.0, scale)
        return (arr - center) / scale
    if mode == "l1":
        denom = np.nansum(np.abs(arr), axis=axis, keepdims=True)
        denom = np.where(denom < 1e-8, 1.0, denom)
        return arr / denom
    raise ValueError(f"Unknown attribution normalization {mode!r}.")


def compose_huber_like(
    activity,
    rewire,
    formulation: str = "activity_rewire",
    *,
    catalytic: float = 0.0,
    temperature: float = 1.0,
    as_auc: bool = True,
) -> float:
    """Additive Huber-like latent: activity + gated rewire + catalytic tax.

    Ablations:
    * ``activity`` — ProteinGym competence, plus catalytic knockout
    * ``activity_rewire`` — activity + pocket-gated substitution operator
    * ``rewire`` — substitution operator only (specificity residual)
    """
    if formulation not in ZERO_SHOT_FORMULATIONS:
        raise ValueError(
            f"Unknown zero-shot formulation {formulation!r}; "
            f"expected one of {ZERO_SHOT_FORMULATIONS}."
        )
    act = float(activity) + float(catalytic)
    spec = float(rewire)
    if formulation == "activity":
        latent = act
    elif formulation == "rewire":
        latent = spec
    else:
        latent = act + spec
    if as_auc:
        return float(sigmoid_auc(latent, temperature=temperature))
    return float(latent)


def compose_zero_shot(
    pocket,
    attribution,
    mutation_effect,
    formulation: str = "activity_rewire",
    *,
    attr: str = "signed",
    effect: str = "signed",
    rewire: float = 0.0,
    catalytic: float = 0.0,
    temperature: float = 1.0,
    as_auc: bool = True,
) -> float:
    """Zero-shot score. Additive Huber-like by default; product forms remain.

    Additive (default):
    * ``activity`` — ``e``
    * ``activity_rewire`` — ``e + q-gated rewire``
    * ``rewire`` — substitution operator only

    Legacy product (for comparison only):
    * ``mutation`` — ``e``
    * ``attribution_mutation`` — ``a * e``
    * ``pocket_attribution_mutation`` — ``q * a * e``
    """
    if formulation in ZERO_SHOT_FORMULATIONS:
        return compose_huber_like(
            mutation_effect,
            rewire,
            formulation,
            catalytic=catalytic,
            temperature=temperature,
            as_auc=as_auc,
        )
    if formulation not in PRODUCT_FORMULATIONS:
        raise ValueError(
            f"Unknown zero-shot formulation {formulation!r}; "
            f"expected one of {ZERO_SHOT_FORMULATIONS + PRODUCT_FORMULATIONS}."
        )
    q = float(pocket)
    a = float(attribution)
    e = float(mutation_effect)
    if attr == "abs":
        a = abs(a)
    if effect == "abs":
        e = abs(e)
    if formulation == "mutation":
        return e
    if formulation == "attribution_mutation":
        return a * e
    return q * a * e


def compose_delta(
    mutation_effect,
    pocket_weight,
    substrate_z,
    cfg: ComposerConfig,
    rng: np.random.Generator | None = None,
    noise_mad: float = 1.0,
) -> float:
    fn = COMPOSERS[cfg.formulation]
    value = float(fn(mutation_effect, pocket_weight, substrate_z, cfg))
    if rng is not None and cfg.noise_scale > 0:
        value = value + float(cfg.noise_scale * noise_mad * rng.normal())
    return value


@dataclass
class SyntheticExample:
    pos: int
    wt: str
    mut: str
    window: str
    delta: float
    pocket: float
    mutation_effect: float
    substrate_z: float
    kind: str


def select_mutation_sites(
    pocket_q: np.ndarray,
    sequence: str,
    rng: np.random.Generator,
    n_pocket: int,
    n_offpocket: int,
) -> list[tuple[int, str]]:
    """Return ``(index, kind)`` mutation sites, mixing pocket and off-pocket."""
    q = np.asarray(pocket_q, dtype=np.float64).reshape(-1)
    n = min(int(q.size), len(sequence))
    if n <= 1:
        return []
    letters = [str(sequence[i]).upper() for i in range(n)]
    valid = np.array([aa in PROTEIN_ALPHABET for aa in letters], dtype=bool)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return []
    q = np.clip(q[:n][idx], 1e-6, 1.0)
    inv = np.clip(1.0 - q, 1e-6, 1.0)
    n_pocket = min(int(n_pocket), int(idx.size))
    n_off = min(int(n_offpocket), max(int(idx.size) - n_pocket, 0))
    pocket_pick = _weighted_choice(idx, q, n_pocket, rng)
    remain = np.setdiff1d(idx, pocket_pick, assume_unique=False)
    if remain.size and n_off:
        remain_w = inv[np.isin(idx, remain)]
        off_pick = _weighted_choice(remain, remain_w, n_off, rng)
    else:
        off_pick = np.zeros(0, dtype=int)
    out = [(int(i), "pocket") for i in pocket_pick] + [
        (int(i), "offpocket") for i in off_pick
    ]
    rng.shuffle(out)
    return out


def _weighted_choice(items, weights, k, rng):
    items = np.asarray(items)
    weights = np.asarray(weights, dtype=np.float64)
    k = min(int(k), int(items.size))
    if k <= 0 or items.size == 0:
        return np.zeros(0, dtype=int)
    weights = np.clip(weights, 1e-8, None)
    weights = weights / weights.sum()
    return rng.choice(items, size=k, replace=False, p=weights)


def choose_mutant_aa(wt: str, rng: np.random.Generator) -> str:
    wt = str(wt).upper()
    choices = [aa for aa in PROTEIN_ALPHABET if aa != wt]
    return str(rng.choice(choices))


def generate_matrix(
    *,
    sequence: str,
    pocket_q: np.ndarray,
    windows: list[str],
    decoys: list[str],
    mutation_prior: MutationEffectPrior,
    substrate_prior: SubstratePwmPrior,
    protease_id: str,
    cfg: ComposerConfig,
    rng: np.random.Generator,
    coords: np.ndarray | None = None,
    n_pocket: int = 28,
    n_offpocket: int = 12,
    n_cognate: int = 8,
    n_decoy: int = 4,
) -> list[SyntheticExample]:
    """Build a protease × mutation × substrate synthetic DMS matrix."""
    sites = select_mutation_sites(pocket_q, sequence, rng, n_pocket, n_offpocket)
    if not sites:
        return []
    cognate = list(dict.fromkeys(str(w) for w in windows if w))
    rng.shuffle(cognate)
    cognate = cognate[: max(int(n_cognate), 1)]
    decoy = [str(w) for w in decoys if w and w not in cognate]
    rng.shuffle(decoy)
    decoy = decoy[: max(int(n_decoy), 0)]
    pool = cognate + decoy
    if not pool:
        return []
    q = np.asarray(pocket_q, dtype=np.float64).reshape(-1)
    if coords is not None and len(np.asarray(coords)) == len(sequence):
        burials = burial_bins(coords)
    else:
        burials = [
            "partial" if (float(q[i]) if i < q.size else 0.0) >= 0.35 else "exposed"
            for i in range(len(sequence))
        ]
    rows: list[SyntheticExample] = []
    noise_mad = float(mutation_prior.global_scale)
    for pos, kind in sites:
        wt = str(sequence[pos]).upper()
        if wt not in PROTEIN_ALPHABET:
            continue
        mut = choose_mutant_aa(wt, rng)
        pocket = float(q[pos]) if pos < q.size else 0.0
        ctx = MutationContext.from_mutation(wt, mut, burials[pos])
        mutation_effect = mutation_prior.sample(ctx, rng)
        for window in pool:
            substrate_z = substrate_prior.zscore(protease_id, window, pool)
            delta = compose_delta(
                mutation_effect,
                pocket,
                substrate_z,
                cfg,
                rng=rng,
                noise_mad=noise_mad,
            )
            rows.append(
                SyntheticExample(
                    pos=int(pos),
                    wt=wt,
                    mut=mut,
                    window=str(window),
                    delta=float(delta),
                    pocket=pocket,
                    mutation_effect=float(mutation_effect),
                    substrate_z=float(substrate_z),
                    kind=kind,
                )
            )
    return rows


def scale_rows(rows: list[SyntheticExample]) -> list[SyntheticExample]:
    """Robust-center synthetic labels so A/B/C share a comparable target scale."""
    if not rows:
        return rows
    values = np.asarray([row.delta for row in rows], dtype=np.float64)
    center, scale = robust_scale(values)
    for row in rows:
        row.delta = float((row.delta - center) / scale)
    return rows


def composer_payload(cfg: ComposerConfig) -> dict:
    return asdict(cfg)
