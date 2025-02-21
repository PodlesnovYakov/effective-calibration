import torch
import pytorch_lightning as pl
import numpy as np

from typing import Optional, Tuple, Dict
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


class OUDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for generating and managing Ornstein-Uhlenbeck process data.
    
        :param num_samples: Number of trajectories to generate
        :param traj_length: Length of each trajectory in time steps
        :param batch_size: Batch size for DataLoaders
        :param num_workers: Number of workers for DataLoaders
        :param dt: Time step size for Euler-Maruyama discretization
        :param param_ranges: Dictionary with parameter ranges for theta, mu, sigma
        :param data_generated: Flag indicating if data has been generated
    """
    
    def __init__(self,
                 num_samples: int = 50000,
                 traj_length: int = 200,
                 batch_size: int = 64,
                 num_workers: int = 16,
                 dt: float = 0.01,
                 theta_range: Tuple[float, float] = (0.1, 10),
                 mu_range: Tuple[float, float] = (-5, 5),
                 sigma_range: Tuple[float, float] = (0.1, 5)):
        """Initialize the OUDataModule with configurable parameters.
        
            :param num_samples: Number of OU process trajectories to generate
            :param traj_length: Length of each trajectory in time steps
            :param batch_size: Batch size for training/validation/test loaders
            :param num_workers: Number of workers for data loading
            :param dt: Time step size for Euler-Maruyama discretization
            :param theta_range: Tuple (min, max) for mean reversion rate parameter
            :param mu_range: Tuple (min, max) for long-term mean parameter
            :param sigma_range: Tuple (min, max) for volatility parameter
        """
        super().__init__()
        self.save_hyperparameters(ignore=["theta_range", "mu_range", "sigma_range"])
        self.num_workers = num_workers
        self.num_samples = num_samples
        self.traj_length = traj_length
        self.batch_size = batch_size
        self.dt = dt
        self.param_ranges = {
            "theta": theta_range,
            "mu": mu_range,
            "sigma": sigma_range
        }
        self.data_generated = False
        self.X_train: Optional[torch.Tensor] = None
        self.y_train: Optional[torch.Tensor] = None
        self.X_val: Optional[torch.Tensor] = None
        self.y_val: Optional[torch.Tensor] = None
        self.X_test: Optional[torch.Tensor] = None
        self.y_test: Optional[torch.Tensor] = None

    def generate_ou_data(self) -> None:
        """Generate OU process trajectories using Euler-Maruyama method."""
        np.random.seed(42)
        trajectories = []
        parameters = []
        
        for i in range(self.num_samples):
            if i % 10000 == 0:
                print(f"Generating {i}/{self.num_samples} trajectories...")
                
            theta = np.random.uniform(*self.param_ranges["theta"])
            mu = np.random.uniform(*self.param_ranges["mu"])
            sigma = np.random.uniform(*self.param_ranges["sigma"])
            
            X = np.zeros(self.traj_length)
            X[0] = mu
            for t in range(1, self.traj_length):
                dW = np.random.normal(0, np.sqrt(self.dt))
                dX = theta * (mu - X[t-1]) * self.dt + sigma * dW
                X[t] = X[t-1] + dX
                
            trajectories.append(X)
            parameters.append([theta, mu, sigma])
            
        X = np.array(trajectories)[..., np.newaxis]  
        y = np.array(parameters)
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42)
            
        self.X_train = torch.FloatTensor(X_train)
        self.y_train = torch.FloatTensor(y_train)
        self.X_val = torch.FloatTensor(X_val)
        self.y_val = torch.FloatTensor(y_val)
        self.X_test = torch.FloatTensor(X_test)
        self.y_test = torch.FloatTensor(y_test)

    def setup(self, stage: Optional[str] = None) -> None:
        """Prepare datasets for current stage (fit/test)."""
        if not self.data_generated:
            self.generate_ou_data()
            self.data_generated = True
            
        if stage in ("fit", None):
            self.train_dataset = TensorDataset(self.X_train, self.y_train)
            self.val_dataset = TensorDataset(self.X_val, self.y_val)
            
        if stage in ("test", None):
            self.test_dataset = TensorDataset(self.X_test, self.y_test)

    def _create_dataloader(self, 
                          dataset: TensorDataset, 
                          shuffle: bool = False) -> DataLoader:
        """Create DataLoader with consistent configuration.
        
            :param dataset: TensorDataset to load
            :param shuffle: Whether to shuffle the data
            :return: Configured DataLoader instance
        """
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0
        )

    def train_dataloader(self) -> DataLoader:
        """Get training DataLoader."""
        return self._create_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Get validation DataLoader."""
        return self._create_dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        """Get test DataLoader."""
        return self._create_dataloader(self.test_dataset)
