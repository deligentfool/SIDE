import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class inner_state_autoencoder(nn.Module):
    def __init__(self, args):
        super(inner_state_autoencoder, self).__init__()
        self.state_input_dim = args.latent_dim * args.n_agents
        self.action_input_dim = args.n_actions 
        self.latent_dim = args.latent_dim * args.n_agents
        self.hidden_dim = args.hidden_dim
        self.action_embedding_dim = 8
        self.n_agents = args.n_agents

        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.hidden_dim),
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_input_dim, self.action_embedding_dim)
        )

        self.cat_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim + self.action_embedding_dim * args.n_agents, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.mu = nn.Sequential(
            nn.Linear(self.hidden_dim, self.state_input_dim)
        )

        self.logvar = nn.Sequential(
            nn.Linear(self.hidden_dim, self.state_input_dim)
        )

        self.state_decoder = nn.Sequential(
            nn.Linear(self.state_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.state_input_dim)
        )

        self.action_decoder = nn.ModuleList([nn.Sequential(
            nn.Linear(self.state_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_input_dim)
        ) for _ in range(args.n_agents)])

    def encode(self, s_t_1, a):
        state_encoder = self.state_encoder(s_t_1)
        action_encoder = self.action_encoder(a)
        encoder = torch.cat([state_encoder, action_encoder.view(action_encoder.size(0), action_encoder.size(1), -1)], dim=-1)
        encoder = self.cat_encoder(encoder)
        mu, logvar = self.mu(encoder), self.logvar(encoder)
        return mu, logvar

    def sample_z(self, mu, logvar):
        eps = torch.randn_like(logvar)
        return mu + eps * torch.exp(0.5 * logvar)

    def decode(self, z):
        state_decoder = self.state_decoder(z)
        action_decoders = []
        for n in range(self.n_agents):
            action_decoder = self.action_decoder[n](z)
            action_decoders.append(action_decoder)
        action_decoders = torch.stack(action_decoders, dim=-2)
        action_decoders = torch.log_softmax(action_decoders, dim=-1)
        return state_decoder, action_decoders

    def forward(self, s_t_1, a):
        mu, logvar = self.encode(s_t_1, a)
        z = self.sample_z(mu, logvar)
        state_decoder, action_decoders = self.decode(z)
        return state_decoder, action_decoders, mu, logvar
