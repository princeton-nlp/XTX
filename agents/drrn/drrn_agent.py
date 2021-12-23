# Built-in Imports
import logging
import traceback
import pickle
from os.path import join as pjoin
from typing import Dict, Union, List, Callable

# Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer
import wandb

from jericho.util import clean

# Custom imports
import utils.logger as logger
from utils.memory import PrioritizedReplayMemory, Transition, StateWithActs, State
from utils.env import JerichoEnv
import utils.ngram as Ngram

from models import DrrnQNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DrrnAgent:
    def __init__(
        self,
        tb: logger.Logger,
        log: Callable[..., None],
        args: Dict[str, Union[str, int, float]],
        envs: List[JerichoEnv],
        action_models
    ):
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.network = DrrnQNetwork(
            tb=tb,
            log=log,
            vocab_size=len(self.tokenizer),
            envs=envs,
            action_models=action_models,
            tokenizer=self.tokenizer,
            args=args
        ).to(device)
        self.target_network = DrrnQNetwork(
            tb=tb,
            log=log,
            vocab_size=len(self.tokenizer),
            envs=envs,
            action_models=action_models,
            tokenizer=self.tokenizer,
            args=args
        ).to(device)
        self.target_network.eval()
        # if args.wandb:
        #     wandb.watch(self.network, log='all')

        self.memory = PrioritizedReplayMemory(args.memory_size,
                                              args.priority_fraction)
        self.clip = args.clip
        self.tb = tb
        self.log = log

        self.optimizer = torch.optim.Adam(self.network.parameters(),
                                          lr=args.learning_rate)
        self.action_models = action_models
        self.max_acts = args.max_acts
        self.envs = envs

    def observe(self, transition, is_prior=False):
        """
        Push to replay memory.
        """
        self.memory.push(transition, is_prior=is_prior)

    def build_skip_state(self, ob: str, info: Dict[str, Union[List[str], float, str]], next_action_str: str, traj_acts: List[str]) -> StateWithActs:
        """Returns a state representation built from various info sources.

        Args:
            ob (str): the observation.
            info (Dict[str, Union[List[str], float, str]]): info dict.
            traj_acts (List[str]): past actions.
            next_action_str (str): current action.

        Returns:
            StateWithActs: state representation.
        """
        acts = Ngram.build_traj_state(self, next_action_str, traj_acts)

        obs_ids = self.tokenizer.encode(ob)
        look_ids = self.tokenizer.encode(info['look'])
        inv_ids = self.tokenizer.encode(info['inv'])
        return StateWithActs(obs_ids, look_ids, inv_ids, acts, info['score'])

    def build_state(self, ob: str, info: Dict[str, Union[List[str], float, str]]) -> State:
        """Returns a state representation built from various info sources. 

        Args:
            ob (str): the observation.
            info (Dict[str, Union[List[str], float, str]]): info dict.

        Returns:
            State: state representation.
        """
        obs_ids = self.tokenizer.encode(ob)
        look_ids = self.tokenizer.encode(info['look'])
        inv_ids = self.tokenizer.encode(info['inv'])

        return State(obs_ids, look_ids, inv_ids, info['score'])

    def build_states(
        self,
        obs: List[str],
        infos: List[Dict[str, Union[List[str], float, str]]],
        action_strs: List[str] = None,
        traj_acts: List[List[str]] = None
    ) -> Union[List[State], List[StateWithActs]]:
        """Build list of state representations.

        Args:
            obs (List[str]): list of observations per env.
            infos (List[Dict[str, Union[List[str], float, str]]]): list of info dicts per env.
            action_strs (List[str], optional): list of current action strings per env. Defaults to None.
            traj_acts (List[List[str]], optional): list of past action strings per env. Defaults to None.

        Returns:
            Union[List[State], List[StateWithActs]]: list of state representations.
        """
        if action_strs is None and traj_acts is None:
            return [self.build_state(ob, info) for ob, info in zip(obs, infos)]
        else:
            return [self.build_skip_state(ob, info, action_str, traj_act) for ob, info, action_str, traj_act in zip(obs, infos, action_strs, traj_acts)]

    def encode(self, obs: List[str]):
        """ 
        Encode a list of strings with [SEP] at the end.
        """
        return [self.tokenizer.encode(o) for o in obs]

    def transfer_weights(self):
        """
        TODO
        """
        self.target_network.load_state_dict(self.network.state_dict())

    def act(self, states, poss_acts, poss_act_strs, sample=True):
        """
        Parameters 
        ----------
        poss_acts: [
            [[<act_1_state_1>], [<act_2_state_1>], ...],
            ... (* number of env)
        ]

        Returns
        -------
        act_ids: the action IDs of the chosen action per env
            [
                [<IDx>, <IDy>, ...],
                ... (* number of env)
            ]
        idxs: index of the chosen action per env
            [
                <index_1>,
                ... (* number of env)
            ]
        qvals: tuple of qvals per valid action set (i.e. per env)
            (
                [<qval_1>, <qval_2>, ...],
                ... (* number of env)
            )
        """
        # Idxs: indices of the sampled (from the Q-vals) actions
        idxs, qvals = self.network.act(states, poss_acts, poss_act_strs)

        # Get the sampled action for each environment
        act_ids = [poss_acts[batch][idx] for batch, idx in enumerate(idxs)]
        return act_ids, idxs, qvals

    def act_topk(self, states, poss_acts):
        """
        """
        idxs = self.network.act_topk(states, poss_acts)

        return idxs

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Compute Q(s', a') for all a'
        with torch.no_grad():
            next_qvals = self.target_network(batch.next_state, batch.next_acts)

        # Take the max over next q-values
        next_qvals = torch.tensor([vals.max() for vals in next_qvals],
                                  device=device)

        # Zero all the next_qvals that are done
        next_qvals = next_qvals * (
            1 - torch.tensor(batch.done, dtype=torch.float, device=device))
        targets = torch.tensor(batch.reward, dtype=torch.float,
                               device=device) + self.gamma * next_qvals

        # Next compute Q(s, a)
        act_sizes = [1 for act in batch.act]

        nested_acts = tuple([[a] for a in batch.act])
        qvals = self.network(batch.state, nested_acts)
        qvals = torch.cat(qvals)

        # Compute Huber loss
        loss = F.smooth_l1_loss(qvals, targets.detach())

        self.tb.logkv_mean('Q', qvals.mean())

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.clip)
        self.optimizer.step()

        return loss.item()

    def load(self, run_id: str, weight_file: str, memory_file: str):
        try:
            api = wandb.Api()
            run = api.run(f"princeton-nlp/text-games/{run_id}")
            run.file(f"{weight_file}.pt").download(wandb.run.dir)
            run.file(f"{memory_file}.pkl").download(wandb.run.dir)

            self.memory = pickle.load(
                open(pjoin(wandb.run.dir, f"{memory_file}.pkl"), 'rb'))
            self.network.load_state_dict(
                torch.load(pjoin(wandb.run.dir, f"{weight_file}.pt")))
        except Exception as e:
            self.log(f"Error loading model {e}")
            logging.error(traceback.format_exc())
            raise Exception("Didn't properly load model!")

    def load_memory(self, run_id: str, memory_file: str):
        try:
            api = wandb.Api()
            run = api.run(f"princeton-nlp/text-games/{run_id}")
            run.file(f"{memory_file}.pkl").download(wandb.run.dir)

            self.memory = pickle.load(
                open(pjoin(wandb.run.dir, f"{memory_file}.pkl"), 'rb'))
        except Exception as e:
            self.log(f"Error loading replay memory {e}")
            logging.error(traceback.format_exc())
            raise Exception("Didn't properly load replay memory!")

    def save(self, step: int, traj: List = None):
        try:
            # save locally
            pickle.dump(
                self.memory,
                open(pjoin(wandb.run.dir, 'memory_{}.pkl'.format(step)), 'wb'))
            torch.save(self.network.state_dict(),
                       pjoin(wandb.run.dir, 'weights_{}.pt'.format(step)))

            if traj is not None:
                pickle.dump(
                    traj, open(
                        pjoin(wandb.run.dir, 'traj_{}.pkl'.format(step)), 'wb')
                )
                wandb.save(pjoin(wandb.run.dir, 'traj_{}.pkl'.format(step)))

            # upload to wandb
            wandb.save(pjoin(wandb.run.dir, 'weights_{}.pt'.format(step)))
            wandb.save(pjoin(wandb.run.dir, 'memory_{}.pkl'.format(step)))
        except Exception as e:
            print("Error saving model.")
            logging.error(traceback.format_exc())
