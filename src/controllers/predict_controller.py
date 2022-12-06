from torch.optim import Adam
from modules.predict import PredictNet
import torch.nn.functional as F
import copy
import torch


class AgentPredict(object):
    """
    Independent predict network for each agent
    """
    def __init__(self, a_i, args, scheme):
        """
        Inputs:
            a_i (int): The i-th agent
        """
        self.args = args
        self.input_obs_shape = scheme["obs"]["vshape"]
        self.action_shape = scheme["avail_actions"]["vshape"]
        self.predict = PredictNet(a_i, self.input_obs_shape, self.action_shape[0], args)
        self.predict_optimizer = Adam(self.predict.parameters(), lr=args.predict_lr)

    def pre_step(self, obs, previous_other_acs, avail_other_actions):
        previous_other_acs = torch.stack(previous_other_acs).to(self.args.device)
        predict_actions = self.predict(obs, previous_other_acs)
        predict_actions[avail_other_actions == 0] = 0
        return predict_actions
