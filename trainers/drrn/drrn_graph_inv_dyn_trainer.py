# Built-in Imports
from typing import Dict, Union, Callable, List
import time
from os.path import join as pjoin

# Libraries
from jericho.util import clean
import wandb
import torch

# Custom Imports
from trainers import DrrnInvDynTrainer

from agents import DrrnInvDynAgent

from utils.vec_env import VecEnv
from utils.memory import Transition
import utils.logger as logger
from utils.env import JerichoEnv
import utils.drrn as Drrn
import utils.inv_dyn as InvDyn
import utils.ngram as Ngram
from utils.util import process_action


class DrrnGraphInvDynTrainer(DrrnInvDynTrainer):
    def __init__(
        self,
        tb: logger.Logger,
        log: Callable[..., None],
        agent: DrrnInvDynAgent,
        envs: VecEnv,
        eval_env: JerichoEnv,
        args: Dict[str, Union[str, int, float]]
    ):
        super().__init__(tb, log, agent, envs, eval_env, args)

        self.graph_num_explore_steps = args.graph_num_explore_steps
        self.graph_rescore_freq = args.graph_rescore_freq
        self.graph_merge_freq = args.graph_merge_freq
        self.log_top_blue_acts_freq = args.log_top_blue_acts_freq

        self.use_il_graph_sampler = args.use_il_graph_sampler
        self.use_il_buffer_sampler = args.use_il_buffer_sampler
        self.use_il = args.use_il

    def train(self):
        start = time.time()
        max_score = 0

        obs, infos, states, valid_ids, transitions = Drrn.setup_env(
            self, self.envs)

        for step in range(1, self.max_steps + 1):
            self.steps = step
            self.log("Step {}".format(step))
            action_ids, action_idxs, action_qvals = self.agent.act(states,
                                                                   valid_ids,
                                                                   [info['valid']
                                                                       for info in infos],
                                                                   sample=True)

            # Get the actual next action string for each env
            action_strs = [
                info['valid'][idx] for info, idx in zip(infos, action_idxs)
            ]

            # Log envs[0]
            s = ''
            for idx, (act, val) in enumerate(
                    sorted(zip(infos[0]['valid'], action_qvals[0]),
                           key=lambda x: x[1],
                           reverse=True), 1):
                s += "{}){:.2f} {} ".format(idx, val.item(), act)
            self.log('Q-Values: {}'.format(s))

            # Update all envs
            infos, next_states, next_valids, max_score, obs = self.update_envs(
                action_strs, action_ids, states, max_score, transitions, obs, infos, action_qvals)
            states, valid_ids = next_states, next_valids

            self.end_step(step, start, max_score, action_qvals)

    def update_envs(self, action_strs, action_ids, states, max_score: int,
                    transitions, obs, infos, qvals):
        """
        TODO
        """
        next_obs, next_rewards, next_dones, next_infos = self.envs.step(
            action_strs)

        if self.use_il_graph_sampler:
            next_node_ids = [graph.state_hash(next_info, next_ob) for graph, next_ob, next_info in zip(
                self.agent.graphs, next_obs, next_infos)]

            # Add to environment trajectory
            trajs = self.envs.add_traj(
                list(map(lambda x: (process_action(x[0]), x[1]),
                         zip(action_strs, next_node_ids))))

            # Update graph depending on state of environment
            self.log('Updating graph ...')
            for i, (graph, ob, info, qvals, next_ob, next_info, act) in enumerate(zip(self.agent.graphs, obs, infos, qvals, next_obs, next_infos, action_strs)):
                graph.maybe_update(ob, info, next_ob, next_info,
                                   qvals.cpu().detach().tolist(), i, process_action(act))
        if self.use_action_model:
            next_states = self.agent.build_states(
                next_obs, next_infos, action_strs, [state.acts for state in states])
        else:
            next_states = self.agent.build_states(next_obs, next_infos)

        # Update valid acts if next node is already in the tree
        next_valids = [self.agent.encode(next_info['valid'])
                       for next_info in next_infos]

        if self.r_for > 0:
            reward_curiosity, _ = InvDyn.inv_loss_decode(self.agent.network,
                                                         states, next_states, [[a] for a in action_ids], hat=True, reduction='none')
            next_rewards = next_rewards + reward_curiosity.detach().numpy() * self.r_for
            self.tb.logkv_mean('Curiosity', reward_curiosity.mean().item())

        for i, (next_ob, next_reward, next_done, next_info, state, next_state, next_action_str) in enumerate(zip(next_obs, next_rewards, next_dones, next_infos, states, next_states, action_strs)):
            # Log
            self.log('Action_{}: {}'.format(
                self.steps, next_action_str), condition=(i == 0))
            self.log("Reward{}: {}, Score {}, Done {}".format(
                self.steps, next_reward, next_info['score'], next_done), condition=(i == 0))
            self.log('Obs{}: {} Inv: {} Desc: {}'.format(
                self.steps, clean(next_ob), clean(next_info['inv']),
                clean(next_info['look'])), condition=(i == 0))

            transition = Transition(
                state, action_ids[i], next_reward, next_state, next_valids[i], next_done)
            transitions[i].append(transition)
            self.agent.observe(transition)

            if next_done:
                # Add trajectory to graph
                if self.use_il_buffer_sampler:
                    self.agent.il_buffer.add_traj(transitions[i])

                if next_info['score'] >= max_score:  # put in alpha queue
                    if next_info['score'] > max_score:
                        self.agent.memory.clear_alpha()
                        max_score = next_info['score']
                    for transition in transitions[i]:
                        self.agent.observe(transition, is_prior=True)
                transitions[i] = []

                if self.use_action_model:
                    Ngram.log_recovery_metrics(self, i)

                # Add last node to graph
                if self.use_il_graph_sampler:
                    if next_infos[i]['look'] != 'unknown' and next_infos[i]['inv'] != 'unknown':
                        with torch.no_grad():
                            _, qvals = self.agent.network.act(
                                next_states, next_valids, [next_info['valid'] for next_info in next_infos])
                        self.agent.graphs[i].maybe_update(
                            next_ob, next_info, None, None, qvals[i].cpu().tolist(), i, None)
                

                next_infos = list(next_infos)

                next_obs[i], next_infos[i] = self.envs.reset_one(i)

                if self.use_action_model:
                    next_states[i] = self.agent.build_skip_state(
                        next_obs[i], next_infos[i], 'reset', [])
                else:
                    next_states[i] = self.agent.build_state(
                        next_obs[i], next_infos[i])

                next_valids[i] = self.agent.encode(next_infos[i]['valid'])

        return next_infos, next_states, next_valids, max_score, next_obs

    def end_step(self, step: int, start, max_score: int, action_qvals):
        """
        TODO
        """
        if step % self.q_update_freq == 0:
            self.update_agent()

        if step % self.target_update_freq == 0:
            self.agent.transfer_weights()

        if self.use_action_model:
            Ngram.end_step(self, step)

        if step % self.log_freq == 0:
            # rank_metrics = self.evaluate_optimal()
            rank_metrics = dict()
            self.write_to_logs(step, start, self.envs, max_score, action_qvals,
                               rank_metrics)

        # Save model weights etc.
        if step % self.checkpoint_freq == 0:
            self.agent.save(int(step / self.checkpoint_freq))

            if self.use_il:
                # save locally
                torch.save(self.agent.action_models.state_dict(),
                           pjoin(wandb.run.dir, 'il_weights_{}.pt'.format(step)))

                # upload to wandb
                wandb.save(
                    pjoin(wandb.run.dir, 'il_weights_{}.pt'.format(step)))

    def write_to_logs(self, step, start, envs, max_score, qvals, rank_metrics,
                      *args):
        """
        Log any relevant metrics.
        """
        self.tb.logkv('Step', step)
        for key, val in rank_metrics.items():
            self.tb.logkv(key, val)
        self.tb.logkv("FPS", int(
            (step*self.envs.num_envs)/(time.time()-start)))
        self.tb.logkv("EpisodeScores100", self.envs.get_end_scores().mean())
        self.tb.logkv('MaxScore', max_score)
        if self.use_il_graph_sampler:
            self.tb.logkv('#BlueActs', sum(
                [len(node['blue_acts']) for node in self.agent.graphs[0].graph.values()]))
        self.tb.dumpkvs()
