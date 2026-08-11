import awkward as ak

from ..task import Task


class PropertyPredictionTask(Task):
    """Predict a scalar or vector property at a chosen hierarchy level.

    Loads a molecule (or sub-molecule level) from the indexed batch and
    extracts ``{level}_{property}`` as the supervision target. Set
    ``resolution`` to control which atoms/residues are exposed as features.

    Parameters
    ----------
    property : str or tuple
        Target column name, or ``(name, index)`` to select one element of a
        vector-valued property.
    level : str
        Hierarchy level of the target (``"molecule"``, ``"residue"``, etc.).
    resolution : str
        Feature resolution passed to the returned batch (``"atom"`` or
        ``"residue"``).
    """

    def __init__(self, property="label", level="molecule", resolution="atom") -> None:
        super().__init__()
        self.property = property
        self.level = level
        self.resolution = resolution

    def __call__(self, vbatch, assets, index):
        X = vbatch[index["scene"], index["frame"], index["molecule"]]
        X.resolution = self.resolution
        if isinstance(self.property, tuple) or isinstance(self.property, list):
            prop, prop_idx = self.property
            targets = X.molecules.__getattr__(f"{self.level}_{prop}")[:, prop_idx]
        else:
            targets = X.molecules.__getattr__(f"{self.level}_{self.property}")
        if self.level != "molecule" and targets.ndim >= 2:
            y = ak.Array({"target": targets, "sizes": ak.num(targets, axis=1)})
        elif self.level == "molecule":
            y = ak.Array({"target": ak.flatten(targets, axis=None)})
        else:
            y = ak.Array({"target": targets})
        return X, y
