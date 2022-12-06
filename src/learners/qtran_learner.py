import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.qtran import QTranBase
import torch as th
from torch.optim import RMSprop, Adam
import torch.nn.functional as F


class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.scheme = scheme

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer == "qtran_base":
            self.mixer = QTranBase(args)
        elif args.mixer == "qtran_alt":
            raise Exception("Not implemented here!")

        self.params += list(self.mixer.parameters())
        self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        mac_hidden_states = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            predict_actions = batch["predict_actions"][:, t]

            agent_outs = self.mac.forward(batch, predict_actions, t=t)
            mac_out.append(agent_outs)
            mac_hidden_states.append(self.mac.hidden_states)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        mac_hidden_states = th.stack(mac_hidden_states, dim=1)
        mac_hidden_states = mac_hidden_states.reshape(batch.batch_size, self.args.n_agents, batch.max_seq_length, -1).transpose(1,2) #btav

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        target_mac_hidden_states = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            predict_actions = batch["predict_actions"][:, t]

            target_agent_outs = self.target_mac.forward(batch, predict_actions, t=t)
            target_mac_out.append(target_agent_outs)
            target_mac_hidden_states.append(self.target_mac.hidden_states)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[:], dim=1)  # Concat across time
        target_mac_hidden_states = th.stack(target_mac_hidden_states, dim=1)
        target_mac_hidden_states = target_mac_hidden_states.reshape(batch.batch_size, self.args.n_agents, batch.max_seq_length, -1).transpose(1,2) #btav

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, :] == 0] = -9999999  # From OG deepmarl
        mac_out_maxs = mac_out.clone()
        mac_out_maxs[avail_actions == 0] = -9999999

        # Best joint action computed by target agents
        target_max_actions = target_mac_out.max(dim=3, keepdim=True)[1]
        # Best joint-action computed by regular agents
        max_actions_qvals, max_actions_current = mac_out_maxs[:, :].max(dim=3, keepdim=True)

        if self.args.mixer == "qtran_base":
            # -- TD Loss --
            # Joint-action Q-Value estimates
            joint_qs, vs = self.mixer(batch[:, :-1], mac_hidden_states[:,:-1])

            # Need to argmax across the target agents' actions to compute target joint-action Q-Values
            if self.args.double_q:
                max_actions_current_ = th.zeros(size=(batch.batch_size, batch.max_seq_length, self.args.n_agents, self.args.n_actions), device=batch.device)
                max_actions_current_onehot = max_actions_current_.scatter(3, max_actions_current[:, :], 1)
                max_actions_onehot = max_actions_current_onehot
            else:
                max_actions = th.zeros(size=(batch.batch_size, batch.max_seq_length, self.args.n_agents, self.args.n_actions), device=batch.device)
                max_actions_onehot = max_actions.scatter(3, target_max_actions[:, :], 1)
            target_joint_qs, target_vs = self.target_mixer(batch[:, 1:], hidden_states=target_mac_hidden_states[:,1:], actions=max_actions_onehot[:,1:])

            # Td loss targets
            td_targets = rewards.reshape(-1,1) + self.args.gamma * (1 - terminated.reshape(-1, 1)) * target_joint_qs
            td_error = (joint_qs - td_targets.detach())
            masked_td_error = td_error * mask.reshape(-1, 1)
            td_loss = (masked_td_error ** 2).sum() / mask.sum()
            # -- TD Loss --

            # -- Opt Loss --
            # Argmax across the current agents' actions
            if not self.args.double_q: # Already computed if we're doing double Q-Learning
                max_actions_current_ = th.zeros(size=(batch.batch_size, batch.max_seq_length, self.args.n_agents, self.args.n_actions), device=batch.device )
                max_actions_current_onehot = max_actions_current_.scatter(3, max_actions_current[:, :], 1)
            max_joint_qs, _ = self.mixer(batch[:, :-1], mac_hidden_states[:,:-1], actions=max_actions_current_onehot[:,:-1]) # Don't use the target network and target agent max actions as per author's email

            # max_actions_qvals = th.gather(mac_out[:, :-1], dim=3, index=max_actions_current[:,:-1])
            opt_error = max_actions_qvals[:,:-1].sum(dim=2).reshape(-1, 1) - max_joint_qs.detach() + vs
            masked_opt_error = opt_error * mask.reshape(-1, 1)
            opt_loss = (masked_opt_error ** 2).sum() / mask.sum()
            # -- Opt Loss --

            # -- Nopt Loss --
            # target_joint_qs, _ = self.target_mixer(batch[:, :-1])
            nopt_values = chosen_action_qvals.sum(dim=2).reshape(-1, 1) - joint_qs.detach() + vs # Don't use target networks here either
            nopt_error = nopt_values.clamp(max=0)
            masked_nopt_error = nopt_error * mask.reshape(-1, 1)
            nopt_loss = (masked_nopt_error ** 2).sum() / mask.sum()
            # -- Nopt loss --

        elif self.args.mixer == "qtran_alt":
            raise Exception("Not supported yet.")

        loss = td_loss + self.args.opt_loss * opt_loss + self.args.nopt_min_loss * nopt_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        pre_loss = self.update_predicts(batch)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("td_loss", td_loss.item(), t_env)
            self.logger.log_stat("opt_loss", opt_loss.item(), t_env)
            self.logger.log_stat("nopt_loss", nopt_loss.item(), t_env)
            self.logger.log_stat("pre_loss", pre_loss, t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            if self.args.mixer == "qtran_base":
                mask_elems = mask.sum().item()
                self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
                self.logger.log_stat("td_targets", ((masked_td_error).sum().item()/mask_elems), t_env)
                self.logger.log_stat("td_chosen_qs", (joint_qs.sum().item()/mask_elems), t_env)
                self.logger.log_stat("v_mean", (vs.sum().item()/mask_elems), t_env)
                self.logger.log_stat("agent_indiv_qs", ((chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents)), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def update_predicts(self, batch):
        predict_outputs = []
        real_actions = []
        avail_actions = batch["avail_actions"]
        agent_avail_actions = avail_actions.permute(2, 0, 1, 3)
        bigbatch_avail_actions = agent_avail_actions.reshape(self.mac.n_agents, batch.max_seq_length * batch.batch_size, -1)

        for t in range(batch.max_seq_length):
            if t == 0:
                previous_actions = [th.zeros((self.args.n_agents,
                                                 self.scheme["avail_actions"]["vshape"][0]))
                            for i in range(batch.batch_size)]
                previous_actions = th.stack(previous_actions).to(self.args.device)
            else:
                previous_actions = batch["actions_outputs"][:, t - 1]
            obs = batch["obs"][:, t]# (batch_size, n_agents, obs_shape)
            agent_previous_actions = previous_actions.permute(1, 0, 2)
            agent_obs = obs.permute(1, 0, 2)
            predict_actions = []
            predict_outs = []
            for a_i, pre in zip(range(self.mac.n_agents), self.mac.agent_predicts):
                other_previous_actions = [other_ac for other_ac in agent_previous_actions]
                del other_previous_actions[a_i]
                # (n_agents-1, batch_size, ac_shape)
                predict_out, predict_ac = pre(agent_obs[a_i], other_previous_actions, return_outs=True)
                predict_outs.append(predict_out)
                predict_actions.append(predict_ac)
            predict_outputs.append(th.stack(predict_outs))
            input_predict_actions = th.stack(predict_actions).permute(0, 2, 1, 3)
            agent_out = self.mac.forward(batch, input_predict_actions, t)
            real_action = [F.softmax(out, dim=-1) for out in agent_out]
            real_actions.append(th.stack(real_action, dim=1).detach())

        # Combining seq and batch into one big batch
        # (max_seq_length, n_agents, n_agents - 1, batch_size, action_shape)
        # → (n_agents, max_seq_length * batch_size, n_agents - 1 * action_shape)
        agent_predicts = th.stack(predict_outputs).permute(1, 0, 3, 2, 4)
        bigbatch_predicts = agent_predicts.reshape(self.mac.n_agents, self.mac.n_agents - 1, batch.max_seq_length * batch.batch_size, -1)
        agent_actions = th.stack(real_actions).permute(2, 1, 0, 3)
        bigbatch_real_actions = agent_actions.reshape(self.mac.n_agents, batch.max_seq_length * batch.batch_size, -1)
        bigbatch_real_actions[bigbatch_avail_actions == 0] = 0
        pre_loss = 0
        for a_i, pre_a in zip(range(self.mac.n_agents), bigbatch_predicts):
            real_actions = [ac for ac in bigbatch_real_actions]
            del real_actions[a_i]
            real_a = [th.argmax(act, dim=1) for act in real_actions]
            for pre, real in zip(pre_a, real_a):
                pre_loss += F.cross_entropy(pre, real.long())
        # pre_loss = pre_loss / batch.batch_size
        pre_loss.backward()
        for pre in self.mac.predicts:
            grad_norm = th.nn.utils.clip_grad_norm_(pre.predict.parameters(), 5)
            pre.predict_optimizer.step()
            pre.predict_optimizer.zero_grad()
        # self.scheduler.step()
        return pre_loss

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))

