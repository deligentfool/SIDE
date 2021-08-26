import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class state_autoencoder(nn.Module):
    def __init__(self, args):
        super(state_autoencoder, self).__init__()
        self.args = args
        #self.input_dim = int(np.prod(args.observation_shape)) + args.n_actions
        self.input_dim = args.rnn_hidden_dim
        self.latent_dim = args.latent_dim
        self.hidden_dim = args.hidden_dim
        self.n_agents = args.n_agents

        self.encoder_weight_1 = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        #self.gru_layer = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.weight_logvar = nn.Linear(self.hidden_dim, self.latent_dim, bias=False)
        self.weight_mu = nn.Linear(self.hidden_dim, self.latent_dim, bias=False)

        self.decoder_weight_1 = nn.Linear(self.latent_dim, self.hidden_dim, bias=False)
        self.decoder_weight_2 = nn.Linear(self.hidden_dim, self.input_dim, bias=False)

    def encode(self, node_features, adj_list):
        hidden = torch.zeros(1, node_features.size(0) * node_features.size(1), self.hidden_dim).to(node_features.device)

        a_tilde = adj_list + torch.eye(self.n_agents).unsqueeze(0).unsqueeze(0).expand_as(adj_list).to(adj_list.device)
        d_tilde_diag = a_tilde.sum(-1).view(-1, self.n_agents, 1) ** (-0.5)
        d_tilde = d_tilde_diag.bmm(torch.ones_like(d_tilde_diag).permute([0, 2, 1])) * torch.eye(self.n_agents).unsqueeze(0).to(a_tilde.device)
        encoder_factor = d_tilde.bmm(a_tilde.view(-1, self.n_agents, self.n_agents)).bmm(d_tilde)

        node_features = node_features.reshape([-1, self.n_agents, self.input_dim])

        encoder_1 = self.encoder_weight_1(encoder_factor.bmm(node_features))
        encoder_1 = F.relu(encoder_1)
        #encoder_1, _ = self.gru_layer(encoder_1, hidden)

        encoder_2 = encoder_factor.bmm(encoder_1)

        mu = self.weight_mu(encoder_2).view(adj_list.size(0), adj_list.size(1), self.n_agents, self.latent_dim)
        logvar = self.weight_logvar(encoder_2).view(adj_list.size(0), adj_list.size(1), self.n_agents, self.latent_dim)
        return mu, logvar

    def sample_z(self, mu, logvar):
        eps = torch.randn_like(logvar)
        return mu + eps * torch.exp(0.5 * logvar)

    def decode(self, z, adj_list):
        a_hat = -adj_list + 2 * torch.eye(self.n_agents).unsqueeze(0).unsqueeze(0).expand_as(adj_list).to(adj_list.device)
        d_hat_diag = (adj_list.sum(-1).view(-1, self.n_agents, 1) + 2) ** (-0.5)
        d_hat = d_hat_diag.bmm(torch.ones_like(d_hat_diag).permute([0, 2, 1])) * torch.eye(self.n_agents).unsqueeze(0).to(a_hat.device)
        decoder_factor = d_hat.bmm(a_hat.view(-1, self.n_agents, self.n_agents)).bmm(d_hat)

        z = z.view(-1, self.n_agents, self.latent_dim)

        decoder_1 = self.decoder_weight_1(decoder_factor.bmm(z))
        decoder_1 = F.relu(decoder_1)

        recon_node_features = self.decoder_weight_2(decoder_factor.bmm(decoder_1)).view(adj_list.size(0), adj_list.size(1), self.n_agents, self.input_dim)
        return recon_node_features

    def forward(self, node_features, adj_list):
        adj_list = adj_list * (1 - torch.eye(self.n_agents)).unsqueeze(0).unsqueeze(0).to(adj_list.device)
        mu, logvar = self.encode(node_features, adj_list)
        z = self.sample_z(mu, logvar)
        recon_node_features = self.decode(z, adj_list)
        return recon_node_features, mu.view(adj_list.size(0), adj_list.size(1), -1), logvar.view(adj_list.size(0), adj_list.size(1), -1), z.view(adj_list.size(0), adj_list.size(1), -1)


class flatten_state_autoencoder(nn.Module):
    def __init__(self, args):
        super(flatten_state_autoencoder, self).__init__()
        self.args = args
        self.input_dim = args.rnn_hidden_dim
        self.latent_dim = args.latent_dim
        self.hidden_dim = args.hidden_dim
        self.n_agents = args.n_agents

        self.encoder_weight_1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.weight_logvar = nn.Linear(self.hidden_dim, self.latent_dim)
        self.weight_mu = nn.Linear(self.hidden_dim, self.latent_dim)

        self.decoder_weight_1 = nn.Linear(self.latent_dim, self.hidden_dim)
        self.decoder_weight_2 = nn.Linear(self.hidden_dim, self.input_dim)

    def encode(self, node_features, adj_list):
        bs = adj_list.size(0)
        sl = adj_list.size(1)
        encoder_1 = F.relu(self.encoder_weight_1(node_features))
        mu = self.weight_mu(encoder_1).view(bs, sl, self.n_agents, self.latent_dim)
        logvar = self.weight_logvar(encoder_1).view(bs, sl, self.n_agents, self.latent_dim)
        return mu, logvar

    def sample_z(self, mu, logvar):
        eps = torch.randn_like(logvar)
        return mu + eps * torch.exp(0.5 * logvar)

    def decode(self, z, adj_list):
        decoder_1 = F.relu(self.decoder_weight_1(z))
        recon_node_features = self.decoder_weight_2(decoder_1).view(adj_list.size(0), adj_list.size(1), self.n_agents, self.input_dim)
        return recon_node_features

    def forward(self, node_features, adj_list):
        adj_list = adj_list * (1 - torch.eye(self.n_agents)).unsqueeze(0).unsqueeze(0).to(adj_list.device)
        mu, logvar = self.encode(node_features, adj_list)
        z = self.sample_z(mu, logvar)
        recon_node_features = self.decode(z, adj_list)
        return recon_node_features, mu.view(adj_list.size(0), adj_list.size(1), -1), logvar.view(adj_list.size(0), adj_list.size(1), -1), z.view(adj_list.size(0), adj_list.size(1), -1)
