import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import load_config

class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

class AlphaZeroNet(nn.Module):
    def __init__(self):
        super().__init__()
        config = load_config()
        net_params = config["network_params"]
        self.conv = nn.Conv2d(net_params["input_channels"], net_params["num_filters"], kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(net_params["num_filters"])
        self.res_blocks = nn.ModuleList([ResidualBlock(net_params["num_filters"]) for _ in range(net_params["num_blocks"])])
        # Policy Head
        self.conv_policy = nn.Conv2d(net_params["num_filters"], 2, kernel_size=1)
        self.bn_policy = nn.BatchNorm2d(2)
        self.dropout_policy = nn.Dropout(net_params["dropout_rate"])
        self.fc_policy = nn.Linear(2 * net_params["board_size"] * net_params["board_size"], net_params["num_actions"])
        # Value Head
        self.conv_value = nn.Conv2d(net_params["num_filters"], 1, kernel_size=1)
        self.bn_value = nn.BatchNorm2d(1)
        self.dropout_value = nn.Dropout(net_params["dropout_rate"])
        self.fc_value1 = nn.Linear(net_params["board_size"] * net_params["board_size"], 256)
        self.fc_value2 = nn.Linear(256, 1)

    def forward(self, x, legal_moves=None):
        x = F.relu(self.bn(self.conv(x)))
        for block in self.res_blocks:
            x = block(x)
        # Policy Head
        policy = F.relu(self.bn_policy(self.conv_policy(x)))
        policy = self.dropout_policy(policy)
        policy = policy.view(policy.size(0), -1)
        policy = self.fc_policy(policy)
        if legal_moves is not None:
            mask = torch.zeros_like(policy)
            for from_idx, to_idx in legal_moves:
                idx = from_idx * 32 + to_idx
                mask[:, idx] = 1.0
            policy = policy * mask
            policy_sum = torch.sum(policy, dim=1, keepdim=True)
            policy = policy / (policy_sum + 1e-8)
        else:
            policy = F.softmax(policy, dim=1)
        # Value Head
        value = F.relu(self.bn_value(self.conv_value(x)))
        value = self.dropout_value(value)
        value = value.view(value.size(0), -1)
        value = F.relu(self.fc_value1(value))
        value = torch.tanh(self.fc_value2(value))
        return policy, value