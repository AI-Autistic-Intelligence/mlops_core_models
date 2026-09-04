from abc import ABC, abstractmethod
import torch.nn as nn

class BaseForecaster(nn.Module, ABC):
    """
    Abstract Base Class for all Time-Series Forecasting Models.
    Ensures a unified API across different architectures (LSTM, Transformer, etc.)
    """
    
    @abstractmethod
    def forward(self, x):
        """
        Forward pass for the model.
        Args:
            x: Input tensor of shape (batch, sequence_length, features)
        Returns:
            Output tensor.
        """
        pass
    
    def get_parameter_count(self):
        """Returns the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
