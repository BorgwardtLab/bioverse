import awkward as ak
import numpy as np

from bioverse.collaters.long import LongCollater
from bioverse.data import Assets
from bioverse.tasks.protease_mutation_delta import ProteaseMutationDeltaTask
from bioverse.utilities.synthetic_dms import (
    ComposerConfig,
    MutationContext,
    MutationEffectPrior,
    RewirePrior,
    SubstratePwmPrior,
    burial_bin,
    chemistry_class,
    chemistry_pair_score,
    compose_delta,
    compose_huber_like,
    compose_zero_shot,
    contact_from_distance,
    generate_matrix,
    neighbor_counts,
    normalize_attribution,
    pocket_from_contacts,
    scale_rows,
    select_mutation_sites,
    sigmoid_auc,
    softplus_gate,
)
from bioverse.adapters.synthetic_dms import build_synthetic_records, pocket_probability


def test_chemistry_and_burial_bins():
    assert chemistry_class("D") == "negative"
    assert chemistry_class("W") == "aromatic"
    assert burial_bin(25) == "buried"
    assert burial_bin(12) == "partial"
    assert burial_bin(3) == "exposed"


def test_mutation_prior_samples_stratum_not_absolute_score():
    rng = np.random.default_rng(0)
    contexts = [
        MutationContext.from_mutation("A", "D", "buried"),
        MutationContext.from_mutation("A", "E", "buried"),
        MutationContext.from_mutation("S", "T", "exposed"),
    ] * 20
    effects = ([-1.5, -1.2, 0.1] * 20)
    prior = MutationEffectPrior().fit(contexts, effects)
    ctx = MutationContext.from_mutation("A", "N", "buried")
    draws = [prior.sample(ctx, rng) for _ in range(32)]
    assert np.mean(draws) < 0.0
    assert prior.n_fit == 60


def test_mutation_prior_expect_is_deterministic():
    contexts = [MutationContext.from_mutation("A", "D", "buried")] * 20
    prior = MutationEffectPrior().fit(contexts, [-1.5] * 20)
    ctx = MutationContext.from_mutation("A", "N", "buried")
    assert prior.expect(ctx) == prior.expect(ctx)
    assert prior.expect(ctx) < 0.0


def test_zero_shot_ablations():
    assert compose_zero_shot(0.2, 3.0, -1.5, "mutation") == -1.5
    assert compose_zero_shot(0.2, 3.0, -1.5, "attribution_mutation") == -4.5
    assert abs(compose_zero_shot(0.2, 3.0, -1.5, "pocket_attribution_mutation") + 0.9) < 1e-9
    assert compose_zero_shot(0.2, -3.0, -1.5, "attribution_mutation", attr="abs") == -4.5
    assert compose_zero_shot(0.2, -3.0, -1.5, "attribution_mutation", attr="abs", effect="abs") == 4.5


def test_complex_contact_is_half_at_radius():
    assert abs(float(contact_from_distance(8.0)) - 0.5) < 1e-6
    assert float(contact_from_distance(4.0)) > 0.8
    assert float(contact_from_distance(14.0)) < 0.05
    q = pocket_from_contacts(np.array([[0.1, 0.8], [0.2, 0.0]]))
    assert abs(q[0] - 0.8) < 1e-9
    assert abs(q[1] - 0.2) < 1e-9


def test_chemistry_pair_prefers_complementary_charge():
    assert chemistry_pair_score("D", "K") > chemistry_pair_score("D", "E")
    assert chemistry_pair_score("D", "K") > chemistry_pair_score("A", "K")
    assert chemistry_pair_score("I", "L") > 0


def test_softplus_gate_is_nonnegative_and_not_zscored():
    values = np.array([[-3.0, 0.0, 3.0]])
    gated = softplus_gate(values, axis=1)
    assert np.all(gated >= 0)
    assert gated[0, 2] > gated[0, 0]


def test_rewire_flips_when_pocket_charge_matches_peptide():
    prior = RewirePrior().fit(["ENLYFQSG--", "ENLYFESG--", "AAAAKAAA--"], pwm_weight=0.0)
    toward = prior.delta_psi("D", "K", "ENLYFESG--")
    away = prior.delta_psi("K", "D", "ENLYFESG--")
    # Huber-style P6–P4′ window: index 5 is P1 = E.
    assert toward[5] > 0
    assert away[5] < 0
    assert np.allclose(toward, -away)


def test_additive_ablations_do_not_multiply():
    a = compose_huber_like(-1.0, 0.8, "activity", as_auc=False)
    b = compose_huber_like(-1.0, 0.8, "activity_rewire", as_auc=False)
    c = compose_huber_like(-1.0, 0.8, "rewire", as_auc=False)
    assert a == -1.0
    assert abs(b - (-0.2)) < 1e-9
    assert c == 0.8
    auc = compose_huber_like(-1.0, 0.8, "activity_rewire", as_auc=True)
    assert 0.0 < auc < 0.5
    assert abs(auc - float(sigmoid_auc(-0.2))) < 1e-9
    dead = compose_huber_like(-0.2, 0.0, "activity", catalytic=-2.0, as_auc=False)
    assert dead == -2.2


def test_normalize_attribution_zscore():
    values = np.array([[1.0, 2.0, 3.0], [10.0, 10.0, 10.0]])
    out = normalize_attribution(values, "zscore")
    assert abs(out[0].mean()) < 1e-9
    assert np.allclose(out[1], 0.0)


def test_composers_are_ablatable():
    cfg_a = ComposerConfig(formulation="mutation", noise_scale=0.0)
    cfg_b = ComposerConfig(formulation="mutation_pocket", pocket_floor=0.0, noise_scale=0.0)
    cfg_c = ComposerConfig(
        formulation="mutation_pocket_substrate",
        pocket_floor=0.0,
        substrate_gain=1.0,
        noise_scale=0.0,
    )
    a = compose_delta(2.0, 0.1, 2.0, cfg_a)
    b = compose_delta(2.0, 0.1, 2.0, cfg_b)
    c = compose_delta(2.0, 0.1, 2.0, cfg_c)
    assert a == 2.0
    assert abs(b - 0.2) < 1e-6
    assert c > b
    off = compose_delta(2.0, 0.1, -2.0, cfg_c)
    assert off < c


def test_generate_matrix_includes_offpocket_and_is_deterministic():
    sequence = "ARNDCEQGHILKMFPSTWYV" * 3
    coords = np.stack(
        [np.arange(len(sequence)), np.zeros(len(sequence)), np.zeros(len(sequence))],
        axis=1,
    ).astype(float)
    q = np.linspace(0.05, 0.95, len(sequence))
    prior = MutationEffectPrior().fit(
        [MutationContext.from_mutation("A", "D", "partial")] * 16,
        np.linspace(-1, 1, 16),
    )
    pwm = SubstratePwmPrior().fit({"A01.001": ["--ENLYFQSG", "--RTIAVNYT"]})
    cfg = ComposerConfig(formulation="mutation_pocket_substrate", noise_scale=0.0)
    rows_a = generate_matrix(
        sequence=sequence,
        pocket_q=q,
        windows=["--ENLYFQSG", "--RTIAVNYT"],
        decoys=["--AAAAAAAA"],
        mutation_prior=prior,
        substrate_prior=pwm,
        protease_id="A01.001",
        cfg=cfg,
        rng=np.random.default_rng(1),
        coords=coords,
        n_pocket=6,
        n_offpocket=4,
        n_cognate=2,
        n_decoy=1,
    )
    rows_b = generate_matrix(
        sequence=sequence,
        pocket_q=q,
        windows=["--ENLYFQSG", "--RTIAVNYT"],
        decoys=["--AAAAAAAA"],
        mutation_prior=prior,
        substrate_prior=pwm,
        protease_id="A01.001",
        cfg=cfg,
        rng=np.random.default_rng(1),
        coords=coords,
        n_pocket=6,
        n_offpocket=4,
        n_cognate=2,
        n_decoy=1,
    )
    assert rows_a
    assert [row.kind for row in rows_a].count("offpocket") > 0
    assert {row.window for row in rows_a} >= {"--ENLYFQSG"}
    assert [(r.pos, r.mut, r.window, r.delta) for r in rows_a] == [
        (r.pos, r.mut, r.window, r.delta) for r in rows_b
    ]


def test_select_sites_mixes_pocket_and_background():
    rng = np.random.default_rng(2)
    q = np.array([0.9, 0.85, 0.1, 0.05, 0.8, 0.02])
    seq = "ARNDCE"
    picked = select_mutation_sites(q, seq, rng, n_pocket=2, n_offpocket=2)
    kinds = {kind for _i, kind in picked}
    assert kinds == {"pocket", "offpocket"}
    assert len(picked) == 4


def test_neighbor_counts_self_excluded():
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [30.0, 0.0, 0.0]])
    counts = neighbor_counts(coords, radius=2.0)
    assert counts[0] == 1
    assert counts[2] == 0


def test_pocket_probability_uses_biolip_hits():
    coords = np.stack([np.arange(6), np.zeros(6), np.zeros(6)], axis=1).astype(float)
    q = pocket_probability(coords, [10, 11, 12, 13, 14, 15], "P12345", {"P12345": {12}})
    assert q[2] >= 0.9


def test_scale_rows_centers_labels():
    from bioverse.utilities.synthetic_dms import SyntheticExample

    rows = [
        SyntheticExample(0, "A", "D", "--ENLYFQSG", 4.0, 0.8, 1.0, 0.0, "pocket"),
        SyntheticExample(1, "R", "K", "--ENLYFQSG", 0.0, 0.2, 0.0, 0.0, "offpocket"),
        SyntheticExample(2, "N", "T", "--ENLYFQSG", -4.0, 0.1, -1.0, 0.0, "offpocket"),
    ]
    scaled = scale_rows(rows)
    values = np.array([row.delta for row in scaled])
    assert abs(np.median(values)) < 1e-6


def _dummy_entry(code="A01.001", sequence="ARNDCEQGHILKMFPSTWYV" * 2):
    n = len(sequence)
    coords = np.stack([np.arange(n) * 3.8, np.zeros(n), np.zeros(n)], axis=1)
    record = ak.Record(
        {
            "chain_label": ak.Array(["A"])[np.newaxis, np.newaxis],
            "residue_label": ak.Array([list(sequence)])[np.newaxis, np.newaxis],
            "residue_number": ak.Array([np.arange(1, n + 1)])[np.newaxis, np.newaxis],
            "atom_label": ak.Array([[["CA"] for _ in range(n)]])[np.newaxis, np.newaxis],
            "atom_pos": ak.Array([[[[float(i) * 3.8, 0.0, 0.0]] for i in range(n)]])[
                np.newaxis, np.newaxis
            ],
            "molecule_id": ak.Array([[code]]),
        }
    )
    return {
        "code": code,
        "uniprot": "P00001",
        "source": "dummy",
        "record": record,
        "sequence": sequence,
        "coords": coords,
        "numbers": list(range(1, n + 1)),
        "windows": ["--ENLYFQSG", "--RTIAVNYT"],
    }


def test_task_emits_mutation_and_substrate_on_wt_graph():
    prior = MutationEffectPrior().fit(
        [MutationContext.from_mutation("A", "D", "partial")] * 12,
        np.linspace(-0.5, 0.5, 12),
    )
    pwm = SubstratePwmPrior().fit({"A01.001": ["--ENLYFQSG"]})
    cfg = ComposerConfig(formulation="mutation", noise_scale=0.0)
    records, _used, groups = build_synthetic_records(
        [_dummy_entry()],
        mutation_prior=prior,
        substrate_prior=pwm,
        biolip_sites={},
        cfg=cfg,
        rng=np.random.default_rng(0),
        n_pocket=3,
        n_offpocket=2,
        n_cognate=2,
        n_decoy=0,
    )
    assert records
    assert groups and groups[0]
    batch = ak.Array([records[0]])
    from bioverse.data import Batch

    vbatch = Batch(batch)
    n = len(groups[0])
    index = ak.Array(
        {
            "scene": np.zeros(min(4, n), dtype=int),
            "frame": ak.Array([[0] for _ in range(min(4, n))]),
            "molecule": ak.Array([[0] for _ in range(min(4, n))]),
            "mutation": np.arange(min(4, n)),
        }
    )
    X, y = ProteaseMutationDeltaTask()(vbatch, Assets(), index)
    labels = "".join(ak.to_list(ak.ravel(X.molecules.residue_label))[:20])
    assert labels.startswith("ARND")
    assert "substrate" in y.fields
    assert "position" in y.fields
    assert "wt" in y.fields
    assert "mut" in y.fields
    assert "graph" in y.fields
    assert len(y["target"]) == min(4, n)
    data = LongCollater.collate(X, y, assets=Assets())
    assert data.substrate_token is not None
    assert data.mutation_pos is not None
    assert np.asarray(data.y).shape[0] == min(4, n)


def test_a_vs_c_labels_differ_when_substrates_differ():
    prior = MutationEffectPrior().fit(
        [MutationContext.from_mutation("A", "D", "partial")] * 12,
        np.ones(12),
    )
    pwm = SubstratePwmPrior().fit(
        {"A01.001": ["--ENLYFQSG", "--RTIAVNYT"], "S08.001": ["--AAAAAAAA"]}
    )
    entry = _dummy_entry()
    kwargs = dict(
        mutation_prior=prior,
        substrate_prior=pwm,
        biolip_sites={},
        rng=np.random.default_rng(0),
        n_pocket=4,
        n_offpocket=2,
        n_cognate=2,
        n_decoy=0,
    )
    _, _, a = build_synthetic_records(
        [entry],
        cfg=ComposerConfig(formulation="mutation", noise_scale=0.0),
        **kwargs,
    )
    _, _, c = build_synthetic_records(
        [entry],
        cfg=ComposerConfig(
            formulation="mutation_pocket_substrate", noise_scale=0.0, pocket_floor=0.2
        ),
        rng=np.random.default_rng(0),
        mutation_prior=prior,
        substrate_prior=pwm,
        biolip_sites={},
        n_pocket=4,
        n_offpocket=2,
        n_cognate=2,
        n_decoy=0,
    )
    a_by_mut = {}
    for row in a[0]:
        a_by_mut.setdefault((row.pos, row.mut), set()).add(round(row.delta, 6))
    c_by_mut = {}
    for row in c[0]:
        c_by_mut.setdefault((row.pos, row.mut), set()).add(round(row.delta, 6))
    assert all(len(values) == 1 for values in a_by_mut.values())
    assert any(len(values) > 1 for values in c_by_mut.values())
