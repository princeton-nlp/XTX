# Built-in Imports
from typing import Dict, Union, Callable, List

# Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom Imports
from utils.vec_env import VecEnv
from utils.memory import State, StateWithActs
import utils.logger as logger
import utils.ngram as Ngram
import utils.inv_dyn as InvDyn

from models.drrn.drrn import DrrnQNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DrrnInvDynQNetwork(DrrnQNetwork):
    """
        Deep Reinforcement Relevance Network - He et al. '16
    """

    def __init__(
        self,
        vocab_size: int,
        args: Dict[str, Union[str, int, float]],
        envs: VecEnv,
        tokenizer,
        action_models,
        tb: logger.Logger = None,
        log: Callable[..., None] = None
    ):
        super().__init__(tb, log, vocab_size, envs, action_models, tokenizer, args)
        self.embedding = nn.Embedding(vocab_size, args.drrn_embedding_dim)
        self.obs_encoder = nn.GRU(
            args.drrn_embedding_dim, args.drrn_hidden_dim)
        self.look_encoder = nn.GRU(
            args.drrn_embedding_dim, args.drrn_hidden_dim)
        self.inv_encoder = nn.GRU(
            args.drrn_embedding_dim, args.drrn_hidden_dim)
        self.act_encoder = nn.GRU(
            args.drrn_embedding_dim, args.drrn_hidden_dim)
        self.act_scorer = nn.Linear(args.drrn_hidden_dim, 1)

        self.drrn_hidden_dim = args.drrn_hidden_dim
        self.hidden = nn.Linear(2 * args.drrn_hidden_dim, args.drrn_hidden_dim)
        # self.hidden       = nn.Sequential(nn.Linear(2 * args.drrn_hidden_dim, 2 * args.drrn_hidden_dim), nn.Linear(2 * args.drrn_hidden_dim, args.drrn_hidden_dim), nn.Linear(args.drrn_hidden_dim, args.drrn_hidden_dim))

        self.state_encoder = nn.Linear(
            3 * args.drrn_hidden_dim + (1 if self.augment_state_with_score else 0), args.drrn_hidden_dim)
        self.inverse_dynamics = nn.Sequential(nn.Linear(
            2 * args.drrn_hidden_dim, 2 * args.drrn_hidden_dim), nn.ReLU(), nn.Linear(2 * args.drrn_hidden_dim, args.drrn_hidden_dim))
        self.forward_dynamics = nn.Sequential(nn.Linear(
            2 * args.drrn_hidden_dim, 2 * args.drrn_hidden_dim), nn.ReLU(), nn.Linear(2 * args.drrn_hidden_dim, args.drrn_hidden_dim))

        self.act_decoder = nn.GRU(
            args.drrn_hidden_dim, args.drrn_embedding_dim)
        self.act_fc = nn.Linear(args.drrn_embedding_dim, vocab_size)

        self.obs_decoder = nn.GRU(
            args.drrn_hidden_dim, args.drrn_embedding_dim)
        self.obs_fc = nn.Linear(args.drrn_embedding_dim, vocab_size)

        self.fix_rep = args.fix_rep
        self.hash_rep = args.hash_rep
        self.act_obs = args.act_obs
        self.hash_cache = {}

        self.use_action_model = args.use_action_model
        if self.use_action_model:
            Ngram.init_model(self, action_models, args)

    def forward(self, state_batch: List[Union[State, StateWithActs]], act_batch):
        """
            Batched forward pass.
            obs_id_batch: iterable of unpadded sequence ids
            act_batch: iterable of lists of unpadded admissible command ids
            Returns a tuple of tensors containing q-values for each item in the batch
        """
        state_out = InvDyn.state_rep(self, state_batch)
        act_sizes, act_out = InvDyn.act_rep(self, act_batch)
        # Expand the state to match the batches of actions
        state_out = torch.cat([state_out[i].repeat(j, 1)
                              for i, j in enumerate(act_sizes)], dim=0)
        z = torch.cat((state_out, act_out), dim=1)  # Concat along hidden_dim
        z = F.relu(self.hidden(z))
        act_values = self.act_scorer(z).squeeze(-1)
        # Split up the q-values by batch
        return act_values.split(act_sizes)
