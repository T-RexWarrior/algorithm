import torch
import torch.nn as nn

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)

class TCN_Transformer_CrossAttention(nn.Module):
    def __init__(self, num_forcing_features, num_state_features, seq_len,
                 num_static=2, time_feature_dim=4, num_lc_classes=None,
                 lc_embed_dim=8, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.1):
        super(TCN_Transformer_CrossAttention, self).__init__()

        self.tcn = TemporalConvNet(num_inputs=num_forcing_features,
                                   num_channels=[d_model] * 6,
                                   kernel_size=3, dropout=dropout)

        self.lc_embedding = (
            nn.Embedding(num_lc_classes, lc_embed_dim)
            if num_lc_classes is not None else None
        )
        combined_state_dim = num_state_features + num_static
        if self.lc_embedding is not None:
            combined_state_dim += lc_embed_dim
        self.state_linear = nn.Linear(combined_state_dim, d_model)
        self.time_projector = nn.Linear(time_feature_dim, d_model)

        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)

        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4), nn.ReLU(),
            nn.Linear(d_model // 4, 1)
        )

    def forward(self, x_forcing, x_state, time_x, x_static, x_lc=None):
        x_tcn_in = x_forcing.transpose(1, 2)
        f_tcn = self.tcn(x_tcn_in)
        f_met_memory = f_tcn.transpose(1, 2)

        state_parts = [x_state, x_static]
        if self.lc_embedding is not None:
            if x_lc is None:
                raise ValueError("x_lc is required when land-cover embedding is enabled")
            state_parts.append(self.lc_embedding(x_lc))
        combined_state = torch.cat(state_parts, dim=-1)
        x_s_emb = self.state_linear(combined_state)
        time_emb = self.time_projector(time_x)
        x_state_combined = x_s_emb + time_emb
        f_state_global = self.transformer_encoder(x_state_combined)

        fused_features, _ = self.cross_attention(
            query=f_state_global,
            key=f_met_memory,
            value=f_met_memory
        )

        last_step_features = fused_features[:, -1, :]
        out = self.regressor(last_step_features)
        return out.squeeze(-1)


TCNTransformerCrossAttention = TCN_Transformer_CrossAttention
