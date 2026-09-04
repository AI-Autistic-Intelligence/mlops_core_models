# MLOps Core Models

A centralized repository for Deep Learning architectures used across the MLOps ecosystem.

## Usage
All models inherit from `BaseForecaster` to ensure a consistent API.

```python
from mlops_core_models.lstm import LSTMMarketPredictor

model = LSTMMarketPredictor(input_dim=2)
print(f"Parameters: {model.get_parameter_count()}")
```
