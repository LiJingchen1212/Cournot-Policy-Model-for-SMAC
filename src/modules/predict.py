import torch
import torch.nn as nn
import torch.nn.functional as F

class PredictNet(nn.Module):
    '''
    Network for predicting the actions of other agents
    '''
    def __init__(self, a_i, input_obs_dim, action_dim, args, norm_in=False):
        """
        Inputs:
            sa_sizes (list of (int, int)): Size of state and action spaces per
                                          agent
            hidden_dim (int): Number of hidden dimensions
            CVAE(Conditional Variational Auto-Encoder)
        """
        super(PredictNet, self).__init__()
        self.nagents = args.n_agents

        self.input_dim = input_obs_dim
        self.hidden_dim = args.predict_hidden_dim
        self.z_dim = args.predict_z_dim

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(self.input_dim, affine=False)
        else:
            self.in_fn = lambda x: x

        # encoder ： [b, input_dim] => [b, z_dim]
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3_mu = nn.Linear(self.hidden_dim, self.z_dim)  # mu
        self.fc3_log_std = nn.Linear(self.hidden_dim, self.z_dim)  # log_var

        # get epsilon, input: other agents' actions or strategy distributions
        self.predicts_eps = nn.ModuleList()
        for index in range(self.nagents):
            if index != a_i:
                predict_ep = nn.Sequential()
                predict_ep.add_module('eps_fc1_%d_%d' % (a_i, index), nn.Linear(action_dim, self.hidden_dim))
                predict_ep.add_module('relu1', nn.ReLU())
                predict_ep.add_module('eps_fc2_%d_%d' % (a_i, index), nn.Linear(self.hidden_dim, self.z_dim))
                self.predicts_eps.append(predict_ep)

        # decoder ： [b, z_dim] => [b, output_dim]
        # Output a prediction of the strategy distribution for all other agents
        self.predicts_outs = nn.ModuleList()
        for index in range(self.nagents):
            if index != a_i:
                predict_out = nn.Sequential()
                predict_out.add_module('fc4_%d_%d' % (a_i, index), nn.Linear(self.z_dim, self.hidden_dim))
                predict_out.add_module('relu1', nn.ReLU())
                predict_out.add_module('fc5_%d_%d' % (a_i, index), nn.Linear(self.hidden_dim, action_dim))
                self.predicts_outs.append(predict_out)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        mu = self.fc3_mu(h2)
        log_std = self.fc3_log_std(h2)
        return mu, log_std

    def get_epsilon(self, y):
        eps_h1 = [pre_eps(ac) for ac, pre_eps in zip(y, self.predicts_eps)]
        return eps_h1

    def reparametrize(self, mu, std, eps):
        # std = torch.exp(log_std)
        z = [mu + ep * std for ep in eps]
        return torch.stack(z)

    def decode(self, z):
        out = [pre(z_one) for z_one, pre in zip(z, self.predicts_outs)]
        return out

    def forward(self, obs, previous_other_acs, return_outs=False):
        """
        In:
            Current states
            others' actions or pi
        Out: \bar{a}_t
            Predicted actions of other agents

        :return:
        """
        inp = self.in_fn(obs)
        mu, log_std = self.encode(inp)
        eps = self.get_epsilon(previous_other_acs)
        z = self.reparametrize(mu, log_std, eps)
        outs = self.decode(z)
        pre_probs = [F.softmax(out, dim=-1) for out in outs]
        torch_pre_probs = torch.stack(pre_probs)
        if return_outs:
            return torch.stack(outs), torch_pre_probs
        return torch_pre_probs
