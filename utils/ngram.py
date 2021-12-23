# Built-in imports
import time
from typing import Dict, Union, List
import heapq as pq
import statistics as stats

# Libraries
import numpy as np

from transformers import AdamW, TopPLogitsWarper

import torch

from torch.utils.data import DataLoader

from jericho.util import clean

from tqdm import tqdm

# Custom imports
import definitions.defs as defs

from utils.memory import StateWithActs
from utils.util import process_action, get_class_name, flatten_2d, pad_sequences
from utils.vec_env import VecEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_model(model, action_models, args):
    model.action_models = action_models
    model.sample_action_argmax = args.sample_action_argmax
    model.il_max_context = args.il_max_context
    model.il_vocab_size = args.il_vocab_size
    model.max_acts = args.max_acts
    model.action_model_type = args.action_model_type
    model.score_thresholds = [0.0 for _ in range(len(model.envs))]
    model.traj_lens = [0.0 for _ in range(len(model.envs))]
    model.cut_beta_at_threshold = args.cut_beta_at_threshold
    model.turn_action_model_off_after_falling = args.turn_action_model_off_after_falling
    model.traj_dropout_prob = args.traj_dropout_prob
    model.use_action_model = args.use_action_model
    model.num_bins = args.num_bins
    model.init_bin_prob = args.init_bin_prob
    if model.num_bins > 0:
        model.binning_probs = np.array(
            [model.init_bin_prob for _ in range(model.num_bins)])
        model.bins = np.array(
            [i/model.num_bins for i in range(model.num_bins + 1)])

    # IL stuff
    model.token_remap_cache = dict()
    model.il_temp = args.il_temp
    model.use_il = args.use_il
    model.old_betas = torch.tensor([0. for i in range(len(model.envs))])
    model.il_top_p = args.il_top_p
    model.il_use_dropout = args.il_use_dropout
    model.il_use_only_dropout = args.il_use_only_dropout

    model.top_p_warper = TopPLogitsWarper(
        top_p=model.il_top_p, filter_value=-1e9)


def remap_token_idxs(model, token_idxs):
    result = []
    for token_idx in token_idxs:
        if token_idx in model.token_remap_cache:
            result.append(model.token_remap_cache[token_idx])
        else:
            cur = max(model.token_remap_cache.values()) if len(
                model.token_remap_cache) > 0 else 0
            if cur + 1 >= model.il_vocab_size:
                model.log("Vocab size exceeded during remapping!")
            assert cur + 1 < model.il_vocab_size, "Vocab size exceeded during remapping!"
            result.append(cur + 1)
            model.token_remap_cache[token_idx] = cur + 1

    return result


def init_trainer(trainer, args):
    trainer.top_k_traj = []
    trainer.traj_to_train = []
    trainer.traj_k = args.traj_k
    trainer.last_ngram_update = 0
    trainer.cut_beta_at_threshold = args.cut_beta_at_threshold

    trainer.action_model_update_freq = args.action_model_update_freq
    trainer.action_model_scale_factor = args.action_model_update_freq / 100
    trainer.action_model_type = args.action_model_type
    trainer.use_multi_ngram = args.use_multi_ngram
    trainer.random_action_dropout = args.random_action_dropout
    trainer.max_acts = args.max_acts
    trainer.tf_num_epochs = args.tf_num_epochs

    trainer.init_bin_prob = args.init_bin_prob
    trainer.binning_prob_update_freq = args.binning_prob_update_freq
    trainer.num_bins = args.num_bins

    trainer.episode_ext_type = args.episode_ext_type

    # IL stuff
    trainer.il_batch_size = args.il_batch_size
    trainer.il_k = args.il_k
    trainer.il_lr = args.il_lr
    trainer.il_max_num_epochs = args.il_max_num_epochs
    trainer.il_num_eval_runs = args.il_num_eval_runs
    trainer.il_max_context = args.il_max_context
    trainer.il_eval_freq = args.il_eval_freq
    trainer.il_len_scale = args.il_len_scale


def build_traj_state(agent, prev_act: str, traj_acts: List[str]):
    """Update the trajectory state.
    Args:
        agent ([type]): the agent
        prev_act (str): the last act take by the agent
        traj_acts (List[str]): the past actions taken by agent
    """
    if agent.max_acts == 0 or prev_act == 'reset':
        acts = []
    elif len(traj_acts) == agent.max_acts:
        acts = traj_acts[1:] + [process_action(prev_act)]
    else:
        acts = traj_acts + [process_action(prev_act)]

    return acts


def push_to_traj_mem(trainer, next_info: Dict[str, Union[List[str], str, int]], traj: List[str]):
    """Push current trajectory to the trajectory heap.
    Args:
        trainer (Trainer): trainer to act on
        next_info (Dict[str, Union[str, int]]): contains valid actions etc.
        env ([type]): [description]
    """
    pq.heappush(
        trainer.top_k_traj,
        (next_info['score'], -1 * len(traj), traj.copy()))

    # pop if we have too many
    if len(trainer.top_k_traj) > trainer.traj_k:
        pq.heappop(trainer.top_k_traj)


def get_bin_prob(model):
    """
    """
    x = [min(env.steps/model.traj_len, 1.0) for env in model.envs]
    idxs = np.digitize(x, model.bins, right=True) - 1
    model.log('x {}'.format(x))
    model.log('bins {}'.format(model.bins))
    model.log('indices {}'.format(idxs))

    return [torch.bernoulli(torch.tensor(1. - model.binning_probs[idx])).item() for idx in idxs]


def get_beta_from_lm_vals(model, vals):
    """
    Compute beta parameter for each environment.
    Set beta = 0 if score threshold is reached, otherwise 1.
    """
    assert model.use_il, 'Agent needs to use IL when using LM vals.'

    current_steps = model.envs.get_current_steps()
    current_scores = model.envs.get_current_scores()
    # determine whether to use dropout
    if model.il_use_dropout:
        traj_dropout_probs = [compute_dropout_prob(
            model, traj_len) for traj_len in model.traj_lens]
        model.log("Adjusted dropout prob: {}, env length {}".format(
            traj_dropout_probs, model.traj_lens))
        dropout_results = [torch.bernoulli(torch.tensor(
            1. - traj_dropout_probs[i])).item() for i in range(len(vals))]
        model.log('Traj dropout results: {}'.format(dropout_results))

        betas = torch.tensor([
            1. if current_scores[i] < model.score_thresholds[i] and
            current_steps[i] < model.traj_lens[i] and
            dropout_results[i]
            else 0. for i, val in enumerate(vals)
        ], device=device)
    elif model.il_use_only_dropout:
        traj_dropout_probs = [
            model.traj_dropout_prob for traj_len in model.traj_lens]
        model.log("Hard dropout prob: {}, env length {}".format(
            traj_dropout_probs, model.traj_lens))
        dropout_results = [torch.bernoulli(torch.tensor(
            1. - traj_dropout_probs[i])).item() for i in range(len(vals))]
        model.log('Traj dropout results: {}'.format(dropout_results))

        betas = torch.tensor([
            1. if dropout_results[i]
            else 0. for i, val in enumerate(vals)
        ], device=device)
    else:
        betas = torch.tensor([
            1. if current_scores[i] < model.score_thresholds[i] and
            current_steps[i] < model.traj_lens[i]
            else 0. for i, val in enumerate(vals)
        ], device=device)

    model.old_betas = betas

    return betas


def compute_dropout_prob(model, traj_len):
    """
    """
    return model.traj_dropout_prob * 100 * (1/traj_len) if traj_len > 0 else 1


def action_model_forward(model, states, valid_strs, q_values, act_sizes, il_eval=False):
    """
    """
    # ** action model forward **
    class_name = get_class_name(model).lower()
    if 'drrn' in class_name:
        past_acts = StateWithActs(*zip(*states)).acts

    # TODO: finish transformer case
    if model.action_model_type == defs.TRANSFORMER:
        cls_id = model.tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
        sep_id = model.tokenizer.convert_tokens_to_ids(['[SEP]'])[0]

        input_ids = []
        act_masks = []
        att_masks = []

        for state, acts, valid in zip(states, past_acts, valid_strs):
            context_acts = acts[-model.max_acts:]
            act_history = [model.tokenizer.encode(
                act) for act in context_acts] if len(acts) > 0 else []
            context = [cls_id] + \
                flatten_2d([act[1:-1] + [sep_id]
                           for act in act_history]) + state.obs[1:-1] + state.description[1:-1] + state.inventory[1:-1] + [sep_id]
            for valid_act in valid:
                to_predict = model.tokenizer.encode(valid_act)[1:]
                ids = context + to_predict
                act_mask = np.zeros(len(ids))
                act_mask[len(context):] = 1
                att_mask = np.ones(len(ids))

                if len(ids) > model.il_max_context:
                    input_ids.append(ids[-model.il_max_context:])
                    act_masks.append(act_mask[-model.il_max_context:])
                    att_masks.append(att_mask[-model.il_max_context:])
                else:
                    input_ids.append(ids)
                    act_masks.append(act_mask)
                    att_masks.append(att_mask)

        # remap
        input_ids = [remap_token_idxs(model, seq) for seq in input_ids]

        pad_len = max([len(ids) for ids in input_ids])

        input_ids = torch.tensor(pad_sequences(
            input_ids, pad_len), dtype=torch.long).to(device)
        act_masks = torch.tensor(pad_sequences(
            act_masks, pad_len), dtype=torch.long)
        att_masks = torch.tensor(pad_sequences(
            att_masks, pad_len), dtype=torch.long).to(device)

        # Get lm values
        lm_values = []
        model.action_models.eval()
        with torch.no_grad():
            predictions = model.action_models(
                input_ids, attention_mask=att_masks)[0]
        for prediction, ids, act_mask in zip(predictions, input_ids, act_masks):
            prediction = prediction[np.argmax(act_mask)-1:-1]
            log_p = torch.nn.functional.log_softmax(prediction, dim=-1)
            score = 1./act_mask.sum() * log_p[range(act_mask.sum()),
                                              ids[act_mask == 1]].sum().item()
            lm_values.append(score)

        lm_values = (model.il_temp * torch.tensor(lm_values,
                     device=device)).split(act_sizes)
        lm_values = [model.top_p_warper(None, scores.unsqueeze(0))[
            0] for scores in lm_values]

    model.log("ngram values: {}".format(
        list(map(lambda x: x.item(), lm_values[0]))))

    # Assert shape match
    assert len(q_values) == len(lm_values)
    for q_val, lm_val in zip(q_values, lm_values):
        assert len(q_val) == len(lm_val)

    beta_vec = get_beta_from_lm_vals(model, lm_values)

    # Turn off ngram if it has fallen off
    if model.turn_action_model_off_after_falling:
        on_traj = []
        for i in range(len(model.envs)):
            if beta_vec[i] == 0:
                model.envs.turn_off_trajectory(i)

            on_traj.append(int(model.envs.get_trajectory_state(i)))

        old_beta_vec = beta_vec.clone()
        model.log('Old beta {}'.format(old_beta_vec))
        on_traj = torch.tensor(on_traj, device=device)
        model.log('On traj {}'.format(on_traj))

        beta_vec *= on_traj

        model.tb.logkv_sum('Turned off action model count',
                           (old_beta_vec - beta_vec).sum())

    # log beta & whether trajectory was fully recovered
    model.log('beta {}'.format(beta_vec))

    # update ngram hits
    if isinstance(model.envs, VecEnv):
        model.envs.update_ngram_hits(beta_vec.cpu().numpy())
    else:
        for i in range(len(model.envs)):
            model.envs[i].ngram_hits += beta_vec[i]

    act_values = [
        q_value * (1 - beta_vec[i]) + bert_value * beta_vec[i]
        for i, (q_value, bert_value) in enumerate(zip(q_values, lm_values))
    ]

    return act_values, beta_vec


def update_score_threshold(trainer, i: int):
    """Update score and length thresholds
    Args:
        trainer (Trainer): the trainer
        i (int): environment index
    """
    # NOTE: this assumes 1 trajectory for now
    if len(trainer.traj_to_train) > 0:
        traj = trainer.traj_to_train[0]
        score = traj[0]
        traj_len = -1 * traj[1]

        trainer.agent.network.score_thresholds[i] = score
        trainer.agent.network.traj_lens[i] = traj_len


def _build_traj_states_and_acts(trainer, trajs):
    # find first state
    first_state_id = trainer.agent.graphs[0].get_first_state_id()
    new_trajs = []
    for i in range(len(trajs)):
        new_trajs.append([(None, first_state_id)] + trajs[i])

    traj_states = []
    traj_acts = []
    for traj in new_trajs:
        traj_states.append([])
        traj_acts.append([])
        for t in traj:
            action, state_id = t
            if state_id is not None:
                node = trainer.agent.graphs[0][state_id]
                info = dict()
                info["look"] = node["loc"]
                info["inv"] = node["inv"]
                info["score"] = node["score"]
                state = trainer.agent.build_state(node["obs"], info)
                traj_states[-1].append(state)

            if action is not None:
                traj_acts[-1].append(trainer.agent.encode([action])[0])

    return traj_states, traj_acts


def _build_tf_input_elements(trainer, traj_states, traj_acts):
    input_ids = []
    act_masks = []
    att_masks = []
    for states, acts in zip(traj_states, traj_acts):
        for i in range(len(acts)):
            act_history = acts[max(i - trainer.max_acts, 0): i]
            context = [101] + \
                flatten_2d([act[1:-1] + [102] for act in act_history]) + \
                states[i].obs[1: -1] + states[i].description[1: -1] + \
                states[i].inventory[1: -1] + [102]
            to_predict = acts[i][1:]
            ids = context + to_predict
            act_mask = np.zeros(len(ids))
            act_mask[len(context):] = 1
            att_mask = np.ones(len(ids))

            if len(ids) > trainer.il_max_context:
                trainer.tb.logkv_sum('ExceededContext', 1)
                input_ids.append(ids[-trainer.il_max_context:])
                act_masks.append(act_mask[-trainer.il_max_context:])
                att_masks.append(att_mask[-trainer.il_max_context:])
            else:
                trainer.tb.logkv_sum('ExceededContext', 0)
                input_ids.append(ids)
                act_masks.append(act_mask)
                att_masks.append(att_mask)

    return input_ids, act_masks, att_masks

def my_collate(batch):
    input_ids = [el[0] for el in batch]
    act_masks = [el[1] for el in batch]
    att_masks = [el[2] for el in batch]

    pad_len = max([len(ids) for ids in input_ids])

    input_ids = torch.tensor(pad_sequences(
        input_ids, pad_len), dtype=torch.long)
    act_masks = torch.tensor(pad_sequences(
        act_masks, pad_len), dtype=torch.long)
    att_masks = torch.tensor(pad_sequences(
        att_masks, pad_len), dtype=torch.long)

    return (input_ids, act_masks, att_masks)


def update_il_threshold(trainer, score_threshold, len_threshold):
    trainer.agent.network.score_thresholds = [
        score_threshold for _ in range(len(trainer.envs))]
    trainer.agent.network.traj_lens = [
        len_threshold for _ in range(len(trainer.envs))
    ]

    for i in range(len(trainer.envs)):
        trainer.envs.set_env_limit(
            len_threshold + trainer.graph_num_explore_steps, i)

    trainer.log(f'Setting score threshold: {score_threshold}')
    trainer.log(f'Setting len threshold: {len_threshold}')
    trainer.log(
        f'Setting env limit: {len_threshold + trainer.graph_num_explore_steps}')

    trainer.tb.logkv('ScoreThreshold', score_threshold)
    trainer.tb.logkv('LenThreshold', len_threshold)


def end_step(trainer, step: int):
    """
    Update action model if necessary.
    """
    if (trainer.steps - trainer.last_ngram_update) == trainer.action_model_update_freq:
        trainer.last_ngram_update = trainer.steps
        if trainer.action_model_type == defs.TRANSFORMER:
            # get trajectories
            if trainer.use_il_graph_sampler:
                trainer.il_trajs = trainer.agent.graphs[0].get_graph_policy(
                    k=trainer.il_k)

                state_trajs, act_trajs = _build_traj_states_and_acts(
                    trainer, trainer.il_trajs)
            elif trainer.use_il_buffer_sampler:
                state_trajs, act_trajs = trainer.agent.il_buffer.sample_trajs(
                    trainer.il_k)

            max_score = max(
                [state_traj[-1].score for state_traj in state_trajs])
            max_traj_len = max([len(act_traj) for act_traj in act_trajs])

            # update IL threshold + env len limit
            update_il_threshold(trainer, max_score,
                                trainer.il_len_scale * max_traj_len)

            input_ids, act_masks, att_masks = _build_tf_input_elements(
                trainer, state_trajs, act_trajs)

            # remap
            input_ids = [remap_token_idxs(
                trainer.agent.network, seq) for seq in input_ids]

            X = [(ids, act_mask, att_mask) for ids, act_mask,
                 att_mask in zip(input_ids, act_masks, att_masks)]
            data = DataLoader(X, batch_size=trainer.il_batch_size,
                              shuffle=True, collate_fn=my_collate)

            lm = trainer.agent.action_models
            lm.train()

            # Train!
            optimizer = AdamW(
                lm.parameters(), lr=trainer.il_lr, eps=1e-8)
            start = time.time()
            for i in tqdm(range(trainer.il_max_num_epochs)):
                for batch in data:
                    b_input_ids, b_act_masks, b_att_masks = batch
                    b_input_ids = b_input_ids.to(device)
                    b_act_masks = b_act_masks.to(device)
                    b_att_masks = b_att_masks.to(device)

                    b_labels = b_input_ids.clone()
                    b_labels[b_act_masks == 0] = -100

                    outputs = lm(
                        b_input_ids, attention_mask=b_att_masks, labels=b_labels)

                    loss = outputs[0]
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0)
                    optimizer.step()
                    lm.zero_grad()

            end = time.time()

            trainer.tb.logkv("IL-Loss", loss)
            trainer.tb.logkv("IL-TimeToFit", end - start)

            trainer.action_model_update_freq = (
                max_traj_len + trainer.graph_num_explore_steps) * trainer.action_model_scale_factor

            trainer.tb.logkv('Action Model Update Freq',
                             trainer.action_model_update_freq)

        else:
            raise Exception("Unrecognized action model!")


def log_metrics(trainer):
    # Log min/max/median scores
    scores = []
    lengths = []
    for traj in trainer.traj_to_train:
        scores.append(traj[0])
        lengths.append(traj[1] * -1)

    scores.sort()
    lengths.sort()

    trainer.tb.logkv('Max Traj Score',
                     scores[-1] if len(scores) > 0 else 0)
    trainer.tb.logkv('Min Traj Score',
                     scores[0] if len(scores) > 0 else 0)
    trainer.tb.logkv('Median Traj Score', stats.median(
        scores) if len(scores) > 0 else 0)

    # Log min/max/median length of saved trajectory
    trainer.tb.logkv('Max Traj Length',
                     lengths[-1] if len(lengths) > 0 else 0)
    trainer.tb.logkv('Min Traj Length',
                     lengths[0] if len(lengths) > 0 else 0)
    trainer.tb.logkv('Median Traj Length', stats.median(
        lengths) if len(lengths) > 0 else 0)

    # Log the score threshold
    trainer.tb.logkv('Score Threshold',
                     max(trainer.agent.network.score_thresholds))
    # Log the current episode length
    if isinstance(trainer.envs, VecEnv):
        current_limit = trainer.envs.get_env_limit()
    else:
        current_limit = trainer.envs[0].step_limit
    trainer.tb.logkv('Episode Length', current_limit)


def log_recovery_metrics(trainer, i: int):
    # Log recovery metrics
    if isinstance(trainer.envs, VecEnv):
        ngram_hits = trainer.envs.get_ngram_hits(i)
    else:
        ngram_hits = trainer.envs[i].ngram_hits

    traj_lens = trainer.agent.network.traj_lens
    trainer.tb.logkv_mean(
        'Traj Fully Recovered', 1 if (ngram_hits == traj_lens[i] and traj_lens[i] > 0) else 0)
    trainer.tb.logkv_mean('Traj Part Recovered', ngram_hits /
                          traj_lens[i] if traj_lens[i] > 0 else 0)
