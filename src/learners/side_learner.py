import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop
from modules.vi.vae import inner_state_autoencoder
from modules.vi.vgae import state_autoencoder, flatten_state_autoencoder
import numpy as np


class SIDELearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            self.mixer = QMixer(args, args.latent_dim * args.n_agents)
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        if self.args.vgae:
            self.state_vae = state_autoencoder(self.args).to(args.device)
        else:
            self.state_vae = flatten_state_autoencoder(self.args).to(args.device)
        self.state_prior_vae = inner_state_autoencoder(self.args).to(args.device)

        if self.args.prior:
            self.params += list(self.state_prior_vae.parameters())
        self.params += list(self.state_vae.parameters())

        self.rl_optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

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
        vae_mask = batch["filled"].float()
        prior_vae_mask = th.cat([th.zeros_like(vae_mask[:, 0]).unsqueeze(1), vae_mask[:, :-1]], dim=1)
        avail_actions = batch["avail_actions"]
        past_onehot_action = th.cat([th.zeros_like(batch["actions_onehot"][:, 0]).unsqueeze(1), batch["actions_onehot"][:, :-1]], dim=1)
        #past_reward = th.cat([th.zeros_like(batch["reward"][:, 0]).unsqueeze(1), batch["reward"][:, :-2]], dim=1)

        # Calculate estimated Q-Values
        mac_out = []
        mac_hidden_states = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            mac_hidden_states.append(self.mac.hidden_states)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        mac_hidden_states = th.stack(mac_hidden_states, dim=1)
        mac_hidden_states = mac_hidden_states.reshape(batch.batch_size, self.args.n_agents, batch.max_seq_length, -1).transpose(1, 2).detach() #btav

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        target_mac_hidden_states = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)
            target_mac_hidden_states.append(self.target_mac.hidden_states)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
        target_mac_hidden_states = th.stack(target_mac_hidden_states, dim=1)
        target_mac_hidden_states = target_mac_hidden_states.reshape(batch.batch_size, self.args.n_agents, batch.max_seq_length, -1).transpose(1, 2).detach() #btav

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # * vae for get state
        recon_node_features, state_mu, state_logvar, sample_state = self.state_vae.forward(target_mac_hidden_states, batch['alive_allies'])
        #sample_state = self.state_vae.sample_z(state_mu, state_logvar)
        past_sample_state = th.cat([th.zeros([sample_state.size(0), 1, sample_state.size(2)]).to(sample_state.device), sample_state[:, :-1]], dim=-2)
        recon_state, recon_action, prior_mu, prior_logvar = self.state_prior_vae.forward(past_sample_state, past_onehot_action)
        #inner_state = self.state_prior_vae.sample_z(prior_mu, prior_logvar)
        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, state_mu[:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, state_mu[:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        prior_kld_loss = 1 + prior_logvar - prior_mu.pow(2) - prior_logvar.exp()
        if self.args.prior:
            kld_loss = state_logvar - prior_logvar + 1 - (state_logvar.exp() + (state_mu - prior_mu).pow(2)) / prior_logvar.exp()
        else:
            kld_loss = 1 + state_logvar - state_mu.pow(2) - state_logvar.exp()

        feature_mse_loss = (recon_node_features - target_mac_hidden_states).pow(2)
        prior_state_mse_loss = (recon_state - past_sample_state).pow(2)
        prior_action_loss = - past_onehot_action * recon_action

        # Normal L2 loss, take mean over actual data
        q_loss = (masked_td_error ** 2).sum() / mask.sum()

        state_loss = (feature_mse_loss * vae_mask.unsqueeze(-1).expand_as(feature_mse_loss)).sum() / vae_mask.unsqueeze(-1).expand_as(feature_mse_loss).sum() \
            - 0.5 * (kld_loss * vae_mask.expand_as(kld_loss)).sum() / vae_mask.expand_as(kld_loss).sum()

        prior_loss = - 0.5 * (prior_kld_loss * prior_vae_mask.expand_as(prior_kld_loss)).sum() / prior_vae_mask.expand_as(prior_kld_loss).sum() \
            + (prior_state_mse_loss * prior_vae_mask.expand_as(prior_state_mse_loss)).sum() / prior_vae_mask.expand_as(prior_state_mse_loss).sum() \
            + (prior_action_loss * prior_vae_mask.unsqueeze(-1).expand_as(prior_action_loss)).sum() / prior_vae_mask.unsqueeze(-1).expand_as(prior_action_loss).sum()


        if not self.args.prior:
            prior_loss = 0
        loss = prior_loss + state_loss + q_loss

        # Optimise
        self.rl_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.rl_optimiser.step()


        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

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
        th.save(self.rl_optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.rl_optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
