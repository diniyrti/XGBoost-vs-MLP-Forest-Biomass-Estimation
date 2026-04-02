#import library
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class BiomassDataModule(pl.LightningDataModule):
    def __init__(self, X_data, y_data, batch_size=64, test_size=0.2, val_size=0.1):
        super().__init__()
        # Input
        self.X_raw = X_data.values if hasattr(X_data, 'values') else X_data
        self.y_raw = y_data.values if hasattr(y_data, 'values') else y_data
        
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

        self.train_set = None
        self.val_set = None
        self.test_set = None

    def setup(self, stage=None):
        # Split data (Random Split for Regression)
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.X_raw, self.y_raw, test_size=self.test_size, random_state=42
        )
        
        # Calculate the proportion of val to the remaining temp data
        val_relative_size = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_relative_size, random_state=42
        )

        # Fit & Transform Fitur (X)
        X_train_s = self.scaler_x.fit_transform(X_train)
        X_val_s = self.scaler_x.transform(X_val)
        X_test_s = self.scaler_x.transform(X_test)

        # Fit & Transform Target (y)
        # y is already 2D because you use df[['gedi']]
        y_train_s = self.scaler_y.fit_transform(y_train)
        y_val_s = self.scaler_y.transform(y_val)
        y_test_s = self.scaler_y.transform(y_test)

        # Create TensorDatasets
        self.train_set = TensorDataset(
            torch.tensor(X_train_s, dtype=torch.float32),
            torch.tensor(y_train_s, dtype=torch.float32)
        )
        self.val_set = TensorDataset(
            torch.tensor(X_val_s, dtype=torch.float32),
            torch.tensor(y_val_s, dtype=torch.float32)
        )
        self.test_set = TensorDataset(
            torch.tensor(X_test_s, dtype=torch.float32),
            torch.tensor(y_test_s, dtype=torch.float32)
        )

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)