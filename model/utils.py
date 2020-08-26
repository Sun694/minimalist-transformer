# class inspired by the annotated transformer
# http://nlp.seas.harvard.edu/2018/04/03/

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
