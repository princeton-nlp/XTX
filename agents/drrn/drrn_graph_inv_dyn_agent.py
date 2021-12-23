# Libraries
import torch

# Custom Imports
from agents import DrrnInvDynAgent
from utils.il_buffer import ILBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DrrnGraphInvDynAgent(DrrnInvDynAgent):
    def __init__(self, args, tb, log, envs, action_models):
        super().__init__(args, action_models, tb, log, envs)

        if args.use_il_buffer_sampler:
            self.il_buffer = ILBuffer(self, args, log, tb)
        self.graph_policies = [None] * len(envs)
        self.fell_off_trajectory = [False for _ in range(len(envs))]
        self.use_il = args.use_il

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
        idxs, qvals = self.network.act(
            states, poss_acts, poss_act_strs)

        # Get the sampled action for each environment
        act_ids = [poss_acts[batch][idx] for batch, idx in enumerate(idxs)]
        return act_ids, idxs, qvals