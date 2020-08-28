# classes inspired by the annotated transformer
# http://nlp.seas.harvard.edu/2018/04/03/

import torch
from torchtext import data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BasicIterator(data.Iterator):
    """
    Primarily an object to ensure dynamic batching searches far enough ahead to get tightly grouped batches
    """

    def create_batches(self):
        if self.train:

            def pool_batch(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size,
                        self.batch_size_fn,
                    )
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool_batch(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size, self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


class Batch:
    def __init__(self, src, trg=None, pad_idx=1):
        """
        Object for realtime data preperation, masking, and generating targets for transformers.
        Args:
            src: input indices, [seq_len, bs]
            trg: target indices, [seq_len, bs]
            pad_idx: padding index
        """
        self.src = src
        self.src_mask = (
            src.float()
            .masked_fill(src != pad_idx, False)
            .masked_fill(src == pad_idx, True)
            .transpose(0, 1)
        ).bool()
        if trg is not None:
            self.raw_mask = self.make_std_mask(trg)  # raw mask for generation
            self.trg = trg[:-1, :]  # slice offset for proper teacher forcing
            self.trg_y = trg[1:, :]  # ground truth labels for loss
            self.trg_mask = self.make_std_mask(self.trg)

    def make_std_mask(self, tgt):
        # attn mask for dec-dec attn that prevents seeing future words
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(-2))
        return tgt_mask

    @staticmethod
    def _generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, 0)
            .to(device)
        )
        return mask
