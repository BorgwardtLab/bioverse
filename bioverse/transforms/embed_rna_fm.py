import awkward as ak
import fm
import torch

from ..transform import Transform


class EmbedRnaFm(Transform):

    def __init__(self, device="cpu", batch_size=None):
        self.device = device
        self.batch_size = batch_size
        self.model, self.alphabet = fm.pretrained.rna_fm_t12()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()
        self.model.to(self.device)

    def transform_batch(self, batch):
        sequences = ak.str.join(batch.molecules.residue_label, "")
        names = [f"RNA{i}" for i in range(len(sequences))]
        data = list(zip(names, sequences))
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        embeddings = []
        batch_size = self.batch_size or len(batch_tokens)
        for i in range(0, len(batch_tokens), batch_size):
            batch_tokens_ = batch_tokens[i : i + batch_size]
            with torch.no_grad():
                results = self.model(batch_tokens_.to(self.device), repr_layers=[12])
            embs = ak.Array(results["representations"][12].cpu().numpy())
            mask = batch_tokens_.ne(self.alphabet.padding_idx).numpy()
            embs = ak.drop_none(ak.mask(embs, mask))[:, 1:-1]
            embeddings.append(embs)
        embeddings = ak.concatenate(embeddings, axis=0)
        batch.molecules.residue_features = embeddings
        return batch
