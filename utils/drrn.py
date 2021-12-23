# Built-in imports
from typing import Dict, Union, List
from urllib.parse import uses_relative
from numpy import isnan

# Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# Custom imports
from utils import util
import utils.ngram as Ngram
from utils.memory import State, StateWithActs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_env(self, envs):
    """
    Setup the environment.
    """
    obs, infos = envs.reset()
    if self.use_action_model:
        states = self.agent.build_states(
            obs, infos, ['reset'] * 8, [[]] * 8)
    else:
        states = self.agent.build_states(obs, infos)
    valid_ids = [self.agent.encode(info['valid']) for info in infos]
    transitions = [[] for info in infos]

    return obs, infos, states, valid_ids, transitions


def act(model,
        states: List[Union[State, StateWithActs]],
        valid_ids,
        valid_strs,
        log,
        graph_masks=None):
    """
    Returns an action-string, optionally sampling from the distribution
    of Q-Values.
    """
    act_sizes = [len(valid) for valid in valid_ids]

    if model.sample_uniform:
        q_values = tuple(torch.ones(len(valid_id)).to(device)
                         for valid_id in valid_ids)
    else:
        with torch.no_grad():
            q_values = model.forward(states, valid_ids)
            if torch.any(torch.isnan(q_values[0])):
                log(
                    f"Encountered nan!! State: {states[0]} Valid: {valid_ids[0]}")

    if model.use_action_model:
        act_values, betas = Ngram.action_model_forward(
            model, states, valid_strs, q_values, act_sizes)
    else:
        act_values = q_values

    if graph_masks is not None:
        log(f"Using graph mask: {graph_masks[0]}")
        log(f"Q values: {q_values[0]}")
        act_values = [qvals + mask for qvals,
                      mask in zip(q_values, graph_masks)]
        log(f"Act values: {act_values[0]}")

    if model.sample_argmax:
        act_idxs = [torch.argmax(vals, dim=0) for vals in act_values]
    else:
        probs = [F.softmax(vals/model.T[i], dim=0)
                 for i, vals in enumerate(act_values)]

        if model.use_action_model:
            for i, beta in enumerate(betas):
                if i == 0:
                    model.log('Q dist {}'.format(probs[i]))
                if not beta:
                    model.tb.logkv_mean(
                        'Q Entropy', Categorical(probs[i]).entropy())
                    model.tb.logkv_mean('Uniform entropy', Categorical(
                        torch.ones_like(probs[i])/len(probs[i])).entropy())
        else:
            model.log('Q dist {}'.format(probs[0]))
            model.tb.logkv_mean('Q Entropy', Categorical(probs[0]).entropy())
            model.tb.logkv_mean('Uniform entropy', Categorical(
                torch.ones_like(probs[0])/len(probs[0])).entropy())

        act_idxs = [
            torch.multinomial(dist, num_samples=1).item() for dist in probs
        ]

    return act_idxs, act_values


def init_model(model, args: Dict[str, Union[str, int, float]], vocab_size: int, tokenizer):
    model.use_drrn_inv_look = args.use_drrn_inv_look

    model.embedding = nn.Embedding(vocab_size, args.drrn_embedding_dim)
    model.drrn_hidden_dim = args.drrn_hidden_dim
    model.tokenizer = tokenizer

    model.obs_encoder = nn.GRU(
        args.drrn_embedding_dim, args.drrn_hidden_dim)
    model.look_encoder = nn.GRU(
        args.drrn_embedding_dim, args.drrn_hidden_dim)
    model.inv_encoder = nn.GRU(
        args.drrn_embedding_dim, args.drrn_hidden_dim)
    model.act_encoder = nn.GRU(
        args.drrn_embedding_dim, args.drrn_hidden_dim)

    if model.use_drrn_inv_look:
        model.hidden = nn.Linear(
            4 * args.drrn_hidden_dim, args.drrn_hidden_dim)
    else:
        model.hidden = nn.Linear(
            2 * args.drrn_hidden_dim, args.drrn_hidden_dim)

    model.act_scorer = nn.Linear(args.drrn_hidden_dim, 1)

    model.T = [args.T for _ in range(args.num_envs)]

    model.augment_state_with_score = args.augment_state_with_score
    model.hash_rep = args.hash_rep
    model.hash_cache = {}


def packed_hash(self, x):
    y = []
    for data in x:
        data = hash(tuple(data))
        if data in self.hash_cache:
            y.append(self.hash_cache[data])
        else:
            a = torch.zeros(self.drrn_hidden_dim).normal_(
                generator=torch.random.manual_seed(data))
            # torch.random.seed()
            y.append(a)
            self.hash_cache[data] = a
    y = torch.stack(y, dim=0).to(device)
    return y


def packed_rnn(model, x, rnn):
    """ Runs the provided rnn on the input x. Takes care of packing/unpacking.

        x: list of unpadded input sequences
        Returns a tensor of size: len(x) x hidden_dim
    """
    if model.hash_rep:
        return packed_hash(model, x)

    lengths = torch.tensor([len(n) for n in x],
                           dtype=torch.long,
                           device=device)

    # Sort this batch in descending order by seq length
    lengths, idx_sort = torch.sort(lengths, dim=0, descending=True)
    _, idx_unsort = torch.sort(idx_sort, dim=0)
    idx_sort = torch.autograd.Variable(idx_sort)
    idx_unsort = torch.autograd.Variable(idx_unsort)

    # Pads to longest action
    padded_x = util.pad_sequences(x)
    # print("padded x", padded_x)
    # print("padded x shape", padded_x.shape)
    x_tt = torch.from_numpy(padded_x).type(torch.long).to(device)
    x_tt = x_tt.index_select(0, idx_sort)

    # Run the embedding layer
    embed = model.embedding(x_tt).permute(1, 0, 2)  # Time x Batch x EncDim

    # Pack padded batch of sequences for RNN module
    packed = nn.utils.rnn.pack_padded_sequence(embed, lengths.cpu())

    # Run the RNN
    out, _ = rnn(packed)

    # Unpack
    out, _ = nn.utils.rnn.pad_packed_sequence(out)
    # print("out", out)
    # print("out shape", out.shape)

    # Get the last step of each sequence
    # print("lengths", lengths)
    # print("lengths view", (lengths-1).view(-1, 1))
    # print("out size 2", out.size(2))
    # print("lengths view expanded", (lengths-1).view(-1,1).expand(len(lengths), out.size(2)).unsqueeze(0))
    idx = (lengths - 1).view(-1, 1).expand(len(lengths),
                                           out.size(2)).unsqueeze(0)
    out = out.gather(0, idx).squeeze(0)

    # Unsort
    out = out.index_select(0, idx_unsort)
    return out
