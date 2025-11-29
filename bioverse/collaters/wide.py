import awkward as ak
import numpy as np

from ..collater import Collater


class Data(object):

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if not value is None:
                setattr(self, key, value)

    def uncollate(self, y):
        if hasattr(self, "_sizes"):
            y = ak.unflatten(y, self._sizes, axis=0)
        y = ak.Array({"target": y})
        return y

    # def uncollate(self, y):
    #     if hasattr(self, "_sizes"):
    #         val, pad = self._sizes, y.shape[1] - self._sizes
    #         ones = ak.unflatten(np.ones(ak.sum(val)), val, axis=0)
    #         zeros = ak.unflatten(np.zeros(ak.sum(pad)), pad, axis=0)
    #         y = y[np.where(ak.concatenate([ones, zeros], axis=1))]
    #         y = ak.unflatten(y, self._sizes, axis=0)
    #     y = ak.Array({"target": y})
    #     return y


class WideCollater(Collater):

    @classmethod
    def collate(cls, X, y=None, attr=[], assets=None) -> Data:
        assert (
            "_pad_max_length" in assets
        ), "For the WideCollater, a padding transform must be applied to achieve rectangular tensors."
        num_molecules = ak.sum(X.toc["molecule"].ravel())
        if X.resolution == "atom":
            num_vertices = np.clip(
                X.toc["atom"].sum(axis=-1).sum(axis=-1).ravel(),
                0,
                assets["_pad_max_length"],
            )
            vertex2molecule = np.arange(num_molecules).repeat(num_vertices)
        else:
            num_vertices = np.clip(
                X.toc["residue"].sum(axis=-1).ravel(), 0, assets["_pad_max_length"]
            )
            vertex2molecule = np.arange(num_molecules).repeat(num_vertices)
        attr = [
            "vertex_features",
            "vertex_token",
            "vertex_pos",
            "vertex_mask",
            "vertex_frame_R",
            "residue_backbone",
            "residue_mask",
            "residue_token",
        ] + attr
        return Data(
            num_vertices=num_vertices,
            num_molecules=num_molecules,
            vertex2molecule=vertex2molecule,
            y=(
                (
                    ak.flatten(y["target"], axis=1)
                    if "sizes" in y.fields
                    else y["target"]
                )
                if not y is None
                else None
            ),
            # y=(
            #     (
            #         ak.fill_none(
            #             ak.pad_none(
            #                 y["target"],
            #                 assets["_pad_max_length"],
            #                 clip=assets["_pad_clip"],
            #                 axis=1,
            #             ),
            #             np.full(
            #                 y["target"][0].to_numpy().shape[1:],
            #                 assets["_pad_token"],
            #                 dtype=y["target"][0].to_numpy().dtype,
            #             ),
            #             axis=1,
            #         )
            #         if "sizes" in y.fields
            #         else y["target"]
            #     )
            #     if not y is None
            #     else None
            # ),
            _sizes=(
                (
                    np.clip(y["sizes"], 0, assets["_pad_max_length"])
                    if assets["_pad_clip"]
                    else y["sizes"]
                )
                if not y is None and "sizes" in y.fields
                else None
            ),
            **{
                name: ak.fill_none(
                    ak.pad_none(
                        X.molecules.__getattr__(name),
                        assets["_pad_max_length"],
                        clip=assets["_pad_clip"],
                        axis=1,
                    ),
                    np.full(
                        X.molecules.__getattr__(name)[0].to_numpy().shape[1:],
                        np.nan,  # assets["_pad_token"],
                        # dtype=X.molecules.__getattr__(name)[0].to_numpy().dtype,
                    ),
                    axis=1,
                )
                for name in attr
                if name in X
            },
        )
