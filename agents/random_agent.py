import numpy as np

import sentencepiece as spm

class RandomAgent():
    def __init__(self, args):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(args.spm_path)

    def act(self, valid_ids):
        """
        """
        # print(valid_ids)
        return [np.random.randint(len(valid_acts)) for valid_acts in valid_ids]

    def act_topk(self, valid_ids):
        """
        """
        return [np.random.permutation(len(valid_acts)) for valid_acts in valid_ids]

    def encode(self, obs:[str]):
        """ 
        Encode a list of strings
        """
        return [self.sp.EncodeAsIds(o) for o in obs]