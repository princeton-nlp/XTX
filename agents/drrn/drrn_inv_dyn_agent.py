# Built-in Imports
import pickle
from os.path import join as pjoin
import logging
from typing import List
import traceback

# Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

# Custom Imports
from utils.memory import ABReplayMemory, Transition, StateWithActs, State
import utils.ngram as Ngram
import utils.inv_dyn as InvDyn

from agents import DrrnAgent

from models import DrrnInvDynQNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DrrnInvDynAgent(DrrnAgent):
    def __init__(self, args, action_models, tb, log, envs):
        super().__init__(tb, log, args, envs, action_models)

        self.network = DrrnInvDynQNetwork(
            len(self.tokenizer),
            args,
            envs,
            self.tokenizer,
            action_models,
            tb,
            log
        ).to(device)

        self.target_network = DrrnInvDynQNetwork(
            len(self.tokenizer),
            args,
            envs,
            self.tokenizer,
            action_models,
            tb,
            log
        ).to(device)
        self.target_network.eval()
        self.network.tokenizer = self.tokenizer

        self.memory = ABReplayMemory(args.memory_size, args.memory_alpha)

        self.optimizer = torch.optim.Adam(self.network.parameters(),
                                          lr=args.learning_rate)

        # Inverse Dynamics Stuff
        self.type_inv = args.type_inv
        self.type_for = args.type_for
        self.w_inv = args.w_inv
        self.w_for = args.w_for
        self.w_act = args.w_act
        self.perturb = args.perturb
        self.act_obs = args.act_obs

    def build_skip_state(self, ob, info, action_str: str, traj_act: List[str]):
        """ Returns a state representation built from various info sources. """
        if self.act_obs:
            acts = self.encode(info['valid'])
            obs_ids, look_ids, inv_ids = [], [], []
            for act in acts:
                obs_ids += act
            return State(obs_ids, look_ids, inv_ids)
        obs_ids = self.tokenizer.encode(ob)
        look_ids = self.tokenizer.encode(info['look'])
        inv_ids = self.tokenizer.encode(info['inv'])

        acts = Ngram.build_traj_state(self, action_str, traj_act)

        return StateWithActs(obs_ids, look_ids, inv_ids, acts, info['score'])

    def build_state(self, ob, info):
        """ Returns a state representation built from various info sources. """
        if self.act_obs:
            acts = self.encode(info['valid'])
            obs_ids, look_ids, inv_ids = [], [], []
            for act in acts:
                obs_ids += act
            return State(obs_ids, look_ids, inv_ids)
        obs_ids = self.tokenizer.encode(ob)
        look_ids = self.tokenizer.encode(info['look'])
        inv_ids = self.tokenizer.encode(info['inv'])

        return State(obs_ids, look_ids, inv_ids, info['score'])

    def q_loss(self, transitions, need_qvals=False):
        batch = Transition(*zip(*transitions))

        # Compute Q(s', a') for all a'
        # TODO: Use a target network???
        with torch.no_grad():
            next_qvals = self.target_network(batch.next_state, batch.next_acts)
        # Take the max over next q-values
        next_qvals = torch.tensor([vals.max()
                                  for vals in next_qvals], device=device)
        # Zero all the next_qvals that are done
        next_qvals = next_qvals * \
            (1-torch.tensor(batch.done, dtype=torch.float, device=device))
        targets = torch.tensor(
            batch.reward, dtype=torch.float, device=device) + self.gamma * next_qvals

        # Next compute Q(s, a)
        # Nest each action in a list - so that it becomes the only admissible cmd
        nested_acts = tuple([[a] for a in batch.act])
        qvals = self.network(batch.state, nested_acts)
        # Combine the qvals: Maybe just do a greedy max for generality
        qvals = torch.cat(qvals)
        loss = F.smooth_l1_loss(qvals, targets.detach())

        return (loss, qvals) if need_qvals else loss

    def update(self):
        if len(self.memory) < self.batch_size:
            return None

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        nested_acts = tuple([[a] for a in batch.act])
        terms, loss = {}, 0

        # Compute Q learning Huber loss
        terms['Loss_q'], qvals = self.q_loss(transitions, need_qvals=True)
        loss += terms['Loss_q']

        # Compute Inverse dynamics loss
        if self.w_inv > 0:
            if self.type_inv == 'decode':
                terms['Loss_id'], terms['Acc_id'] = InvDyn.inv_loss_decode(self.network,
                                                                           batch.state, batch.next_state, nested_acts, hat=True)
            elif self.type_inv == 'ce':
                terms['Loss_id'], terms['Acc_id'] = InvDyn.inv_loss_ce(self.network,
                                                                       batch.state, batch.next_state, nested_acts, batch.acts)
            else:
                raise NotImplementedError
            loss += self.w_inv * terms['Loss_id']

        # Compute Act reconstruction loss
        if self.w_act > 0:
            terms['Loss_act'], terms['Acc_act'] = InvDyn.inv_loss_decode(self.network,
                                                                         batch.state, batch.next_state, nested_acts, hat=False)
            loss += self.w_act * terms['Loss_act']

        # Compute Forward dynamics loss
        if self.w_for > 0:
            if self.type_for == 'l2':
                terms['Loss_fd'] = InvDyn.for_loss_l2(self.network,
                                                      batch.state, batch.next_state, nested_acts)
            elif self.type_for == 'ce':
                terms['Loss_fd'], terms['Acc_fd'] = InvDyn.for_loss_ce(self.network,
                                                                       batch.state, batch.next_state, nested_acts, batch.acts)
            elif self.type_for == 'decode':
                terms['Loss_fd'], terms['Acc_fd'] = InvDyn.for_loss_decode(self.network,
                                                                           batch.state, batch.next_state, nested_acts, hat=True)
            elif self.type_for == 'decode_obs':
                terms['Loss_fd'], terms['Acc_fd'] = InvDyn.for_loss_decode(self.network,
                                                                           batch.state, batch.next_state, nested_acts, hat=False)

            loss += self.w_for * terms['Loss_fd']

        # Backward
        terms.update({'Loss': loss, 'Q': qvals.mean()})
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.clip)
        self.optimizer.step()
        return {k: float(v) for k, v in terms.items()}
