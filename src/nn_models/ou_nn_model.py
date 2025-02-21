import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Tuple, Any, Optional
from . ou_abstract_model import OUModel

class LSTMModel(OUModel):
    """Enhanced LSTM model for Ornstein-Uhlenbeck parameter estimation with attention and deep architecture.
    
        :param input_size: Number of input features per time step
        :param hidden_size: Number of LSTM hidden units per layer
        :param num_layers: Number of stacked LSTM layers
        :param dropout: Dropout probability between layers
        :param use_layer_norm: Whether to use LayerNorm instead of BatchNorm
        :param bidirectional: Use bidirectional LSTM
        :param num_directions: 2 if bidirectional else 1 (automatically set)
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 256,
        num_layers: int = 5,
        dropout: float = 0.2,
        use_layer_norm: bool = True,
        bidirectional: bool = True,
        learning_rate: float = 1e-4
    ):
        """Initialize enhanced LSTM model.

            :param input_size: Input feature dimension (default 1 for univariate series)
            :param hidden_size: Number of hidden units in LSTM layers
            :param num_layers: Number of stacked LSTM layers
            :param dropout: Dropout probability between layers
            :param use_layer_norm: Use LayerNorm if True, else BatchNorm
            :param bidirectional: Use bidirectional LSTM
            :param learning_rate: Initial learning rate for optimizer
        """
        super().__init__(
            learning_rate=learning_rate
        )
        
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM Stack
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        if use_layer_norm:
            self.ln = nn.LayerNorm(hidden_size * self.num_directions)
        else:
            self.bn = nn.BatchNorm1d(hidden_size * self.num_directions)

        self.attention = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, 512),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
            :param x: Input tensor of shape (batch_size, seq_len, input_size)
            :return: Tensor of shape (batch_size, 3) containing parameter estimates:
            [theta, mu, sigma] with constraints applied
        """
        lstm_out, _ = self.lstm(x)  
        
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        if hasattr(self, 'ln'):
            context_vector = self.ln(context_vector)
        else:
            context_vector = self.bn(context_vector)
        
        out = self.fc(context_vector)
        
        theta = F.softplus(out[:, 0]) + 1e-6  
        mu = out[:, 1]  
        sigma = F.softplus(out[:, 2]) + 1e-6 
        
        return torch.stack([theta, mu, sigma], dim=1)