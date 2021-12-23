# Built-in Imports
import itertools
import os
import re
from typing import Any, List
from collections import deque

# Libraries
import numpy as np
import pickle
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

zork1_two_obj_acts = [
    'apply( \w+)+ to( \w+)+',
    'tie( \w+)+ to( \w+)+',
    'tie up( \w+)+ with( \w+)+',
    'hit( \w+)+ with( \w+)+',
    'break( \w+)+ with( \w+)+',
    'break down( \w+)+ with( \w+)+',
    'blow up( \w+)+ with( \w+)+',
    'wave( \w+)+ at( \w+)+',
    'clean( \w+)+ with( \w+)+',
    'burn( \w+)+ with( \w+)+',
    'burn down( \w+)+ with( \w+)+',
    'take( \w+)+ from( \w+)+',
    'take( \w+)+ off( \w+)+',
    'take( \w+)+ out( \w+)+',
    'throw( \w+)+( \w+)+',
    'throw( \w+)+ at( \w+)+',
    'throw( \w+)+ in( \w+)+',
    'throw( \w+)+ off( \w+)+',
    'throw( \w+)+ on( \w+)+',
    'throw( \w+)+ over( \w+)+',
    'throw( \w+)+ with( \w+)+',
    'cut( \w+)+ with( \w+)+',
    'dig( \w+)+ with( \w+)+',
    'dig in( \w+)+ with( \w+)+',
    'give( \w+)+( \w+)+',
    'give( \w+)+ to( \w+)+',
    'drop( \w+)+ down( \w+)+',
    'put( \w+)+ on( \w+)+',
    'put( \w+)+ in( \w+)+',
    'touch( \w+)+ with( \w+)+',
    'fill( \w+)+ with( \w+)+',
    'plug( \w+)+ with( \w+)+',
    'turn( \w+)+ for( \w+)+',
    'turn( \w+)+ to( \w+)+',
    'turn( \w+)+ with( \w+)+',
    'turn on( \w+)+ with( \w+)+',
    'untie( \w+)+ from( \w+)+',
    'look at( \w+)+ with( \w+)+',
    'oil( \w+)+ with( \w+)+',
    'put( \w+)+ behind( \w+)+',
    'put( \w+)+ under( \w+)+',
    'inflat( \w+)+ with( \w+)+',
    'is( \w+)+ in( \w+)+',
    'is( \w+)+ on( \w+)+',
    'light( \w+)+ with( \w+)+',
    'melt( \w+)+ with( \w+)+',
    'lock( \w+)+ with( \w+)+',
    'push( \w+)+ with( \w+)+',
    'open( \w+)+ with( \w+)+',
    'ring( \w+)+ with( \w+)+',
    'pick( \w+)+ with( \w+)+',
    'poke( \w+)+ with( \w+)+',
    'pour( \w+)+ from( \w+)+',
    'pour( \w+)+ in( \w+)+',
    'pour( \w+)+ on( \w+)+',
    'push( \w+)+( \w+)+',
    'push( \w+)+ to( \w+)+',
    'push( \w+)+ under( \w+)+',
    'pump up( \w+)+ with( \w+)+',
    'read( \w+)+( \w+)+',
    'read( \w+)+ with( \w+)+',
    'spray( \w+)+ on( \w+)+',
    'spray( \w+)+ with( \w+)+',
    'squeez( \w+)+ on( \w+)+',
    'strike( \w+)+ with( \w+)+',
    'swing( \w+)+ at( \w+)+',
    'unlock( \w+)+ with( \w+)+'
]

ZORK1_TWO_OBJ_REGEX = [re.compile(regexp) for regexp in zork1_two_obj_acts]


def obj_bfs(start, env):
    nodes = []
    visited = set()
    q = deque()
    visited.add(start.num)
    q.append(start)

    while len(q) > 0:
        node = q.popleft()
        nodes.append(node)
        if node.child:
            visited.add(node.child)
            q.append(env.get_object(node.child))
        if node.sibling:
            visited.add(node.sibling)
            q.append(env.get_object(node.sibling))

    return nodes


def extract_inventory(env):
    first_inv_num = env.get_player_object().child
    if not first_inv_num:
        return []

    first_inv_node = env.get_object(first_inv_num)

    return obj_bfs(first_inv_node, env)

# def extract_surrounding(env):
#     first_sur_num = env.get_player_location().child
#     if not first


def filter_two_obj_acts(acts, game: str):
    if game == 'zork1':
        return list(filter(lambda x: any(
            p.match(x) is not None for p in ZORK1_TWO_OBJ_REGEX), acts))
    else:
        raise NotImplementedError(f'Not implemented for game: {game}')


class RunningMeanStd(object):
    # https://github.com/jcwleo/random-network-distillation-pytorch/blob/e383fb95177c50bfdcd81b43e37c443c8cde1d94/utils.py#L44
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-9, shape=()):
        self.mean = torch.zeros(shape, dtype=torch.float32).to(device)
        self.var = torch.ones(shape, dtype=torch.float32).to(device)
        self.count = epsilon

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + \
            torch.square(delta) * self.count * batch_count / \
            (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


def get_class_name(obj: Any) -> str:
    """Get class name of object.

    Args:
        obj (Any): the object.

    Returns:
        [str]: the class name
    """
    return type(obj).__name__


def get_name_from_path(path: str):
    """
    Given a path string, extract the game name from it.
    """
    return path.split('/')[-1].split('.')[0]


def flatten_2d(l: list):
    return list(itertools.chain.from_iterable(l))


def load_object(path: str):
    """
    Load the Python object at the given path.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def save_object(obj, path: str):
    """
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def check_exists(file_path: str):
    """
    """
    return os.path.exists(file_path)


def setup_env(agent, envs):
    """
    TODO
    """
    _, infos = envs.reset()
    states, valid_ids = [], []

    for info in infos:
        states, valid_ids = states + [[]], valid_ids + [
            agent.encode(info['valid'])
        ]

    return infos, states, valid_ids


def convert_idxs_to_strs(act_idxs, tokenizer):
    """
    Given a list of action idxs, convert it to a list of action strings.
    """
    return [
        tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(act)).strip() for act in act_idxs
    ]


def process_action(action: str):
    """
    Transforms the action to lowercase and spells it
    out in case it was abbreviated.
    """
    abbr_map = {
        "w": "west",
        "n": "north",
        "e": "east",
        "s": "south",
        "se": "southeast",
        "sw": "southwest",
        "ne": "northeast",
        "nw": "northwest",
        "u": "up",
        "d": "down",
        "l": "look"
    }

    action = action.strip().lower()
    if action in abbr_map:
        return abbr_map[action]
    return action


def inv_process_action(action: str):
    abbr_map = {
        "west": "w",
        "north": "n",
        "east": "e",
        "south": "s",
        "southeast": "se",
        "southwest": "sw",
        "northeast": "ne",
        "northwest": "nw",
        "up": "u",
        "down": "d",
        "look": "l"
    }

    action = action.strip().lower()
    if action in abbr_map:
        return abbr_map[action]
    return action


def pad_sequences(sequences, maxlen=None, dtype='int32', value=0.):
    '''
    Partially borrowed from Keras
    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        value: float, value to pad the sequences to the desired value.
    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    '''
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break
    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        # pre truncating
        trunc = s[-maxlen:]
        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError(
                'Shape of sample %s of sequence at position %s is different from expected shape %s'
                % (trunc.shape[1:], idx, sample_shape))
        # post padding
        x[idx, :len(trunc)] = trunc
    return x


def add_special_tok_and_pad(trajectories: List[int], special_begin=101):
    """
    TODO
    """
    # Pad to max length of the batch & convert to tensor
    trajectories = pad_sequences(trajectories)
    trajectories = torch.tensor(trajectories, dtype=torch.long, device=device)

    # Add <sos> ([CLS]) token
    sos_tokens = special_begin * torch.ones(
        (len(trajectories), 1), dtype=torch.long, device=device)
    trajectories = torch.cat((sos_tokens, trajectories), dim=1)

    return trajectories


def create_trajectories(past_acts,
                        acts,
                        obs=None,
                        desc=None,
                        inv=None,
                        sep_id: int = None,
                        cls_id: int = None,
                        do_pad_and_special_tok: bool = True):
    """
    TODO
    """
    act_sizes = [len(a) for a in acts]

    # 2D list of unrolled valid actions in the batch
    act_batch = list(itertools.chain.from_iterable(acts))

    for act in act_batch:
        assert len(act) > 1, "empty action! {}".format(act)
        assert act[-1] == 50258, "not ending with sep!"

    if obs is not None:
        states = []
        for i in range(len(past_acts)):
            states.append(past_acts[i] + obs[i] + desc[i] + inv[i] + [sep_id])

        # Repeat state for each valid action in that state
        trajectories = [
            states[i] + acts[i][idx] for i, j in enumerate(act_sizes)
            for idx in range(j)
        ]
    else:
        states = past_acts
        # Repeat state for each valid action in that state
        trajectories = [
            states[i] + acts[i][idx] for i, j in enumerate(act_sizes)
            for idx in range(j)
        ]

    for i, size in enumerate(act_sizes):
        for idx in range(size):
            assert len(acts[i][idx]) > 1, "too short of an action! {}".format(
                acts[i][idx])

    # Note we subtract one here to not count [SEP] token
    mask = [[0] * len(states[i]) + [1] * (len(acts[i][idx]) - 1) + [0]
            for i, size in enumerate(act_sizes) for idx in range(size)]
    assert len(trajectories) == len(act_batch)

    # only pad and add CLS if asked for
    if do_pad_and_special_tok:
        if cls_id is None:
            trajectories = add_special_tok_and_pad(trajectories)
        else:
            trajectories = add_special_tok_and_pad(trajectories,
                                                   special_begin=cls_id)
        mask = add_special_tok_and_pad(mask, special_begin=0)

    # Make sure there is at least one element not masked out
    for el in mask:
        assert 1 in el, "mask consists of all zeros!"

    if hasattr(trajectories, 'shape'):
        assert trajectories.shape == mask.shape
    if hasattr(mask, 'cpu'):
        assert np.all(
            np.array(list(map(lambda x: len(x), act_batch))) ==
            np.sum(mask.cpu().numpy(), axis=1) + 1)

    return trajectories, act_sizes, mask
