# Built-in Imports
import time
from typing import Dict, Union, Callable, List

# Libraries
import torch
import torch.nn.functional as F

import wandb

# Custom Imports
from agents import DrrnInvDynAgent

import utils.logger as logger
from utils.vec_env import VecEnv
from utils.util import process_action, convert_idxs_to_strs


class ILBuffer():
    def __init__(self, agent: DrrnInvDynAgent, args, log, tb):
        self.agent = agent
        self.traj_collection = []
        self.graph_score_temp = args.graph_score_temp
        self.graph_q_temp = args.graph_q_temp
        self.log = log
        self.tb = tb

    def add_traj(self, traj):
        self.traj_collection.append(traj)

    def bin_traj_by_scores(self):
        unknown_tok = self.agent.encode(['unknown'])[0]
        score_bins = dict()
        total = 0
        for traj in self.traj_collection:
            visited = set()
            for i in range(len(traj)):
                desc = traj[i].next_state.description
                inv = traj[i].next_state.inventory
                score = traj[i].next_state.score
                if desc == unknown_tok or inv == unknown_tok:
                    self.log('Encountered unknown token!')

                if score in score_bins and score not in visited and desc != unknown_tok and inv != unknown_tok:
                    score_bins[traj[i].next_state.score].append(
                        traj[:i + 1])
                    visited.add(score)
                elif score not in score_bins and desc != unknown_tok and inv != unknown_tok:
                    score_bins[traj[i].next_state.score] = [traj[:i + 1]]
                    visited.add(score)

            total += len(visited)

        assert total == sum(len(trajs) for trajs in score_bins.values())

        return score_bins

    def sample_trajs(self, k=5):
        if len(self.traj_collection) == 0:
            return [[]]

        start = time.time()

        score_bins = self.bin_traj_by_scores()

        max_score = max(score_bins.keys())

        if len(score_bins) == 0:
            return [[]]

        scores = torch.tensor(sorted(score_bins.keys())).type(torch.float32)
        m_score = scores.mean()
        std_score = torch.std(scores) if not torch.allclose(
            scores - scores[0], torch.zeros_like(scores)) else 1
        norm_scores = (scores - m_score)/std_score
        score_probs = F.softmax(self.graph_score_temp * norm_scores)
        score_idxs = torch.multinomial(
            score_probs, num_samples=k, replacement=True)
        sampled_scores = [int(scores[score_idx.item()].item())
                          for score_idx in score_idxs]

        traj_states = []
        traj_acts = []
        traj_lens = []
        traj_norm_lens = []
        for sampled_score in sampled_scores:
            all_lens = []
            for traj in score_bins[sampled_score]:
                all_lens.append(len(traj))

            all_lens = torch.tensor(all_lens).type(torch.float32)
            m_len = all_lens.mean()
            std_m_len = torch.std(all_lens) if not torch.allclose(
                all_lens - all_lens[0], torch.zeros_like(all_lens)) else 1
            norm_lens = (all_lens - m_len)/std_m_len

            probs = F.softmax(self.graph_q_temp * (-1) * norm_lens, dim=0)

            traj_idx = torch.multinomial(probs, num_samples=1).item()
            sampled_traj = score_bins[sampled_score][traj_idx]

            traj_states.append(
                [sampled_traj[0].state] + [transition.next_state for transition in sampled_traj])
            traj_acts.append([transition.act for transition in sampled_traj])
            traj_lens.append(all_lens[traj_idx])
            traj_norm_lens.append(norm_lens[traj_idx])

        max_len = max(traj_lens)

        for traj_state, traj_act, traj_len, traj_norm_len in zip(traj_states, traj_acts, traj_lens, traj_norm_lens):
            last_state = traj_state[-1]
            obs = convert_idxs_to_strs(
                [last_state.obs[1:-1]], self.agent.tokenizer)[0]
            self.log("Returning to:")
            self.log(
                f"Location: {convert_idxs_to_strs([last_state.description[1:-1]], self.agent.tokenizer)[0]}, \
                Inventory: {convert_idxs_to_strs([last_state.inventory[1:-1]], self.agent.tokenizer)[0]}, \
                Observation: {obs}, \
                Score: {last_state.score}, Len: {traj_len}, NormLen: {traj_norm_len} \
                Path: {convert_idxs_to_strs(list(map(lambda x: x[1:-1], traj_act)), self.agent.tokenizer)}"
            )

            # assert "you have died" not in obs

            self.tb.logkv_mean("ILTrainDataMean", last_state.score)
            self.tb.logkv_mean(
                "Hit@Max", last_state.score == max_score)
            self.tb.logkv_mean("AvgILTrainDataLen", traj_len)
            self.tb.logkv_mean("MaxILTrainDataLen", max_len)

        end = time.time()

        self.tb.logkv_mean("SampleTrajTime", end - start)
        self.tb.logkv("TotalNumTrajectories", len(self.traj_collection))

        return traj_states, traj_acts
