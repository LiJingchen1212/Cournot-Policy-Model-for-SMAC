import torch.nn as nn
import torch.nn.functional as F
import torch


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args
        pre_n_actions = args.n_actions * (args.n_agents - 1)

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim + pre_n_actions, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, pred_actions, hidden_state):
        x = F.relu(self.fc1(inputs))
        # pre_act_probs = torch.stack([pred_action.reshape(1, -1) for pred_action in pred_actions]).squeeze()
        x_pre_act = torch.cat((x, pred_actions), dim=1)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x_pre_act, h_in)
        q = self.fc2(h)
        return q, h
