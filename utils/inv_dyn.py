# Library imports
import itertools
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom imports
from utils import util
from utils.memory import StateWithActs, State

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init(model, args, vocab_size):
    model.embedding = nn.Embedding(vocab_size, args.drrn_embedding_dim)
    model.obs_encoder = nn.GRU(args.drrn_embedding_dim, args.drrn_hidden_dim)
    model.look_encoder = nn.GRU(args.drrn_embedding_dim, args.drrn_hidden_dim)
    model.inv_encoder = nn.GRU(args.drrn_embedding_dim, args.drrn_hidden_dim)
    model.act_encoder = nn.GRU(args.drrn_embedding_dim, args.drrn_hidden_dim)
    model.act_scorer = nn.Linear(args.drrn_hidden_dim, 1)

    model.hidden_dim = args.drrn_hidden_dim
    model.hidden = nn.Linear(2 * args.drrn_hidden_dim, args.drrn_hidden_dim)
    # model.hidden       = nn.Sequential(nn.Linear(2 * hidden_dim, 2 * hidden_dim), nn.Linear(2 * hidden_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim))

    model.state_encoder = nn.Linear(
        3 * args.drrn_hidden_dim, args.drrn_hidden_dim)
    model.inverse_dynamics = nn.Sequential(nn.Linear(
        2 * args.drrn_hidden_dim, 2 * args.drrn_hidden_dim), nn.ReLU(), nn.Linear(2 * args.drrn_hidden_dim, args.drrn_hidden_dim))
    model.forward_dynamics = nn.Sequential(nn.Linear(
        2 * args.drrn_hidden_dim, 2 * args.drrn_hidden_dim), nn.ReLU(), nn.Linear(2 * args.drrn_hidden_dim, args.drrn_hidden_dim))

    model.act_decoder = nn.GRU(args.drrn_hidden_dim, args.drrn_embedding_dim)
    model.act_fc = nn.Linear(args.drrn_embedding_dim, vocab_size)

    model.obs_decoder = nn.GRU(args.drrn_hidden_dim, args.drrn_embedding_dim)
    model.obs_fc = nn.Linear(args.drrn_embedding_dim, vocab_size)

    model.fix_rep = args.fix_rep
    model.hash_rep = args.hash_rep
    model.act_obs = args.act_obs
    model.hash_cache = {}


def packed_hash(self, x):
    import pdb
    pdb.set_trace()
    y = []
    for data in x:
        data = hash(tuple(data))
        if data in self.hash_cache:
            y.append(self.hash_cache[data])
        else:
            a = torch.zeros(self.hidden_dim).normal_(
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


def state_rep(model, state_batch: List[StateWithActs]):
    # Zip the state_batch into an easy access format
    class_name = util.get_class_name(model).lower()

    if 'drrn' in class_name and model.use_action_model:
        state = StateWithActs(*zip(*state_batch))
    elif 'drrn' in class_name and not model.use_action_model:
        state = State(*zip(*state_batch))

    # Encode the various aspects of the state
    with torch.set_grad_enabled(not model.fix_rep):
        obs_out = packed_rnn(model, state.obs, model.obs_encoder)
        if model.act_obs:
            return obs_out
        look_out = packed_rnn(model, state.description, model.look_encoder)
        inv_out = packed_rnn(model, state.inventory, model.inv_encoder)
        if model.augment_state_with_score:
            scores = torch.tensor(state.score).unsqueeze(1).to(device)
            state_out = model.state_encoder(
                torch.cat((obs_out, look_out, inv_out, scores), dim=1))
        else:
            state_out = model.state_encoder(
                torch.cat((obs_out, look_out, inv_out), dim=1))
    return state_out


def act_rep(model, act_batch):
    # This is number of admissible commands in each element of the batch
    act_sizes = [len(a) for a in act_batch]
    # Combine next actions into one long list
    act_batch = list(itertools.chain.from_iterable(act_batch))
    with torch.set_grad_enabled(not model.fix_rep):
        act_out = packed_rnn(model, act_batch, model.act_encoder)
    return act_sizes, act_out


def for_predict(model, state_batch, acts):
    _, act_out = act_rep(model, acts)
    state_out = state_rep(model, state_batch)
    next_state_out = state_out + \
        model.forward_dynamics(torch.cat((state_out, act_out), dim=1))
    return next_state_out


def inv_predict(model, state_batch, next_state_batch):
    state_out = state_rep(model, state_batch)
    next_state_out = state_rep(model, next_state_batch)
    act_out = model.inverse_dynamics(
        torch.cat((state_out, next_state_out - state_out), dim=1))
    return act_out


def inv_loss_l1(model, state_batch, next_state_batch, acts):
    _, act_out = act_rep(model, acts)
    act_out_hat = inv_predict(model, state_batch, next_state_batch)
    return F.l1_loss(act_out, act_out_hat)


def inv_loss_l2(model, state_batch, next_state_batch, acts):
    _, act_out = act_rep(model, acts)
    act_out_hat = inv_predict(model, state_batch, next_state_batch)
    return F.mse_loss(act_out, act_out_hat)


def inv_loss_ce(model, state_batch, next_state_batch, acts, valids, get_predict=False):
    act_sizes, valids_out = act_rep(model, valids)
    _, act_out = act_rep(model, acts)
    act_out_hat = inv_predict(model, state_batch, next_state_batch)
    now, loss, acc = 0, 0, 0
    if get_predict:
        predicts = []
    for i, j in enumerate(act_sizes):
        valid_out = valids_out[now: now + j]
        now += j
        values = valid_out.matmul(act_out_hat[i])
        label = valids[i].index(acts[i][0])
        loss += F.cross_entropy(values.unsqueeze(0),
                                torch.LongTensor([label]).to(device))
        predict = values.argmax().item()
        acc += predict == label
        if get_predict:
            predicts.append(predict)
    return (loss / len(act_sizes), acc / len(act_sizes), predicts) if get_predict else (loss / len(act_sizes), acc / len(act_sizes))


def inv_loss_decode(model, state_batch, next_state_batch, acts, hat=True, reduction='mean'):
    # hat: use rep(o), rep(o'); not hat: use rep(a)
    _, act_out = act_rep(model, acts)
    act_out_hat = inv_predict(model, state_batch, next_state_batch)

    acts_pad = util.pad_sequences([act[0] for act in acts])
    acts_tensor = torch.from_numpy(acts_pad).type(
        torch.long).to(device).transpose(0, 1)
    l, bs = acts_tensor.size()
    vocab = model.embedding.num_embeddings
    outputs = torch.zeros(l, bs, vocab).to(device)
    input, z = acts_tensor[0].unsqueeze(
        0), (act_out_hat if hat else act_out).unsqueeze(0)
    for t in range(1, l):
        input = model.embedding(input)
        output, z = model.act_decoder(input, z)
        output = model.act_fc(output)
        outputs[t] = output
        top = output.argmax(2)
        input = top
    outputs, acts_tensor = outputs[1:], acts_tensor[1:]
    loss = F.cross_entropy(outputs.reshape(-1, vocab),
                           acts_tensor.reshape(-1), ignore_index=0, reduction=reduction)
    if reduction == 'none':  # loss for each term in batch
        lens = [len(act[0]) - 1 for act in acts]
        loss = loss.reshape(-1, bs).sum(0).cpu() / torch.tensor(lens)
    nonzero = (acts_tensor > 0)
    same = (outputs.argmax(-1) == acts_tensor)
    acc_token = (same & nonzero).float().sum() / \
        (nonzero).float().sum()  # token accuracy
    acc_action = (same.int().sum(0) == nonzero.int().sum(
        0)).float().sum() / same.size(1)  # action accuracy
    return loss, acc_action


def for_loss_l2(model, state_batch, next_state_batch, acts):
    next_state_out = state_rep(model, next_state_batch)
    next_state_out_hat = for_predict(model, state_batch, acts)
    return F.mse_loss(next_state_out, next_state_out_hat)  # , reduction='sum')


def for_loss_ce_batch(model, state_batch, next_state_batch, acts):
    # consider duplicates in next_state_batch
    next_states, labels = [], []
    for next_state in next_state_batch:
        if next_state not in next_states:
            labels.append(len(next_states))
            next_states.append(next_state)
        else:
            labels.append(next_states.index(next_state))
    labels = torch.LongTensor(labels).to(device)
    next_state_out = state_rep(model, next_states)
    next_state_out_hat = for_predict(model, state_batch, acts)
    logits = next_state_out_hat.matmul(next_state_out.transpose(0, 1))
    loss = F.cross_entropy(logits, labels)
    acc = (logits.argmax(1) == labels).float().sum() / len(labels)
    return loss, acc


def for_loss_ce(model, state_batch, next_state_batch, acts, valids):
    # classify rep(o') from predict(o, a1), predict(o, a2), ...
    act_sizes, valids_out = act_rep(model, valids)
    _, act_out = act_rep(model, acts)
    next_state_out = state_rep(model, next_state_batch)
    now, loss, acc = 0, 0, 0
    for i, j in enumerate(act_sizes):
        valid_out = valids_out[now: now + j]
        now += j
        next_states_out_hat = for_predict(
            model, [state_batch[i]] * j, [[_] for _ in valids[i]])
        values = next_states_out_hat.matmul(next_state_out[i])
        label = valids[i].index(acts[i][0])
        loss += F.cross_entropy(values.unsqueeze(0),
                                torch.LongTensor([label]).to(device))
        predict = values.argmax().item()
        acc += predict == label
    return (loss / len(act_sizes), acc / len(act_sizes))


def for_loss_decode(model, state_batch, next_state_batch, acts, hat=True):
    # hat: use rep(o), rep(a); not hat: use rep(o')
    next_state_out = state_rep(model, next_state_batch)
    next_state_out_hat = for_predict(model, state_batch, acts)

    import pdb
    pdb.set_trace()
    next_state_pad = util.pad_sequences(next_state_batch)
    next_state_tensor = torch.from_numpy(next_state_batch).type(
        torch.long).to(device).transpose(0, 1)
    l, bs = next_state_tensor.size()
    vocab = model.embedding.num_embeddings
    outputs = torch.zeros(l, bs, vocab).to(device)
    input, z = next_state_tensor[0].unsqueeze(
        0), (next_state_out_hat if hat else next_state_out).unsqueeze(0)
    for t in range(1, l):
        input = model.embedding(input)
        output, z = model.obs_decoder(input, z)
        output = model.obs_fc(output)
        outputs[t] = output
        top = output.argmax(2)
        input = top
    outputs, next_state_tensor = outputs[1:].reshape(
        -1, vocab), next_state_tensor[1:].reshape(-1)
    loss = F.cross_entropy(outputs, next_state_tensor, ignore_index=0)
    nonzero = (next_state_tensor > 0)
    same = (outputs.argmax(1) == next_state_tensor)
    acc = (same & nonzero).float().sum() / \
        (nonzero).float().sum()  # token accuracy
    return loss, acc
