import os
import pickle
import numpy as np
import pandas as pd
import uproot
import awkward as ak

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
from torch.utils.data import DataLoader, Dataset
import lightning as L

class RootProcess:
    def __init__(self, kinematics, weights):
        self.kinematics = kinematics
        self.weights = weights

    def split(self, train_size=1, val_size=1, test_size=None):

        if test_size is not None:
            total_size = train_size + val_size + test_size

            X_train, X_tmp, w_train, w_tmp = train_test_split(
                self.kinematics,
                self.weights,
                train_size=train_size / total_size,
                test_size=(val_size + test_size) / total_size,
                # shuffle=False,
                shuffle=True,
                random_state=42,

            )

            X_val, X_test, w_val, w_test = train_test_split(
                X_tmp,
                w_tmp,
                train_size=val_size / (val_size + test_size),
                test_size=test_size / (val_size + test_size),
                # shuffle=False,
                shuffle=True,
                random_state=42,
            )

            total_wsum = self.weights.sum()
            w_train = w_train * total_wsum / w_train.sum()
            w_val = w_val * total_wsum / w_val.sum()
            w_test = w_test * total_wsum / w_test.sum()

            return (
                RootProcess(X_train.reset_index(drop=True), w_train.reset_index(drop=True)),
                RootProcess(X_val.reset_index(drop=True), w_val.reset_index(drop=True)),
                RootProcess(X_test.reset_index(drop=True), w_test.reset_index(drop=True)),
            )

        total_size = train_size + val_size
        X_train, X_val, w_train, w_val = train_test_split(
            self.kinematics,
            self.weights,
            train_size=train_size / total_size,
            test_size=val_size / total_size,
            # shuffle=False,
            shuffle=True,
            random_state=42,
        )

        total_wsum = self.weights.sum()
        w_train = w_train * total_wsum / w_train.sum()
        w_val = w_val * total_wsum / w_val.sum()

        return (
            RootProcess(X_train.reset_index(drop=True), w_train.reset_index(drop=True)),
            RootProcess(X_val.reset_index(drop=True), w_val.reset_index(drop=True)),
        )

class BalancedDataModule(L.LightningDataModule):


    def __init__(
        self,
        numerator_events: str = '',
        denominator_events: str = '',
        numerator_reweight: tuple = None,
        denominator_reweight: tuple = None,
        features=None,
        batch_size: int = 32,
        random_state=None,
        data_dir: str = './',
    ):
        super().__init__()

        if features is None:
            features = [
                'Z1_PT', 'Z1_Eta', 'Z1_Phi', 'Z1_Mass',
                'Z2_PT', 'Z2_Eta', 'Z2_Phi', 'Z2_Mass'
            ]

        self.features = features
        self.numerator_file = numerator_events
        self.denominator_file = denominator_events
        self.numerator_rwt = numerator_reweight
        self.denominator_rwt = denominator_reweight
        self.batch_size = batch_size
        self.random_state = random_state
        self.data_dir = data_dir
        self.scaler = StandardScaler()

    # =====================================================
    # use the reconstructed Z1 and Z2 features
    # =====================================================

    def load_root_features(self, file_path, tree_name="LHEF", max_weight=1e6):
        tree = uproot.open(file_path)[tree_name]

        z1_pt = tree["Z1_PT"].array(library="np")
        z1_eta = tree["Z1_Eta"].array(library="np")
        z1_phi = tree["Z1_Phi"].array(library="np")
        z1_mass = tree["Z1_Mass"].array(library="np")

        z2_pt = tree["Z2_PT"].array(library="np")
        z2_eta = tree["Z2_Eta"].array(library="np")
        z2_phi = tree["Z2_Phi"].array(library="np")
        z2_mass = tree["Z2_Mass"].array(library="np")

        w = tree["LIV_Weight"].array(library="np")

        X = np.stack(
            [
                z1_pt, z1_eta, z1_phi, z1_mass,
                z2_pt, z2_eta, z2_phi, z2_mass
            ],
            axis=1,
        )
        # X = np.stack(
        #     [z1_pt, z1_eta,
        #      z2_pt, z2_eta],
        #     axis=1,
        # )

        valid = (
            ~np.isnan(X).any(axis=1)
            & ~np.isinf(X).any(axis=1)
            & ~np.isnan(w)
            & ~np.isinf(w)
        )

        X = X[valid]
        w = np.clip(w[valid], 0.0, max_weight)

        return X, w

    def prepare_data(self):
        os.makedirs(self.data_dir, exist_ok=True)
        X, W = self.load_root_features(self.numerator_file)

        events_numerator = RootProcess(
            kinematics=pd.DataFrame(X, columns=self.features),
            weights=pd.Series(W),
        )
        events_denominator = RootProcess(
            kinematics=pd.DataFrame(X, columns=self.features),
            weights=pd.Series(np.ones(len(W))),
        )

        train_size, val_size, test_size = 6, 2, 2
        events_numerator_train, events_numerator_val, events_numerator_test = events_numerator.split(
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
        )
        events_denominator_train, events_denominator_val, events_denominator_test = events_denominator.split(
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
        )

        self.training_data = BalancedDataset(
            events_numerator_train,
            events_denominator_train,
            self.features,
            scaler=None,
            random_state=self.random_state,
        )
        self.scaler.fit(self.training_data.X)

        with open(os.path.join(self.data_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(os.path.join(self.data_dir, 'events_numerator_train.pkl'), 'wb') as f:
            pickle.dump(events_numerator_train, f)
        with open(os.path.join(self.data_dir, 'events_denominator_train.pkl'), 'wb') as f:
            pickle.dump(events_denominator_train, f)
        with open(os.path.join(self.data_dir, 'events_numerator_val.pkl'), 'wb') as f:
            pickle.dump(events_numerator_val, f)
        with open(os.path.join(self.data_dir, 'events_denominator_val.pkl'), 'wb') as f:
            pickle.dump(events_denominator_val, f)
        with open(os.path.join(self.data_dir, 'events_numerator_test.pkl'), 'wb') as f:
            pickle.dump(events_numerator_test, f)
        with open(os.path.join(self.data_dir, 'events_denominator_test.pkl'), 'wb') as f:
            pickle.dump(events_denominator_test, f)

    def setup(self, stage: str):

        if stage == 'fit':

            with open(os.path.join(self.data_dir, 'scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)
            with open(os.path.join(self.data_dir, 'events_numerator_train.pkl'), 'rb') as f:
                events_numerator_train = pickle.load(f)
            with open(os.path.join(self.data_dir, 'events_denominator_train.pkl'), 'rb') as f:
                events_denominator_train = pickle.load(f)
            with open(os.path.join(self.data_dir, 'events_numerator_val.pkl'), 'rb') as f:
                events_numerator_val = pickle.load(f)
            with open(os.path.join(self.data_dir, 'events_denominator_val.pkl'), 'rb') as f:
                events_denominator_val = pickle.load(f)

            self.training_data = BalancedDataset(events_numerator_train, events_denominator_train, self.features, scaler = self.scaler, random_state=self.random_state)
            self.validation_data = BalancedDataset(events_numerator_val, events_denominator_val, self.features, scaler = self.scaler, random_state=self.random_state)

        elif stage == 'test':
            with open(os.path.join(self.data_dir, 'events_numerator_test.pkl'), 'rb') as f:
                events_numerator_test = pickle.load(f)
            with open(os.path.join(self.data_dir, 'events_denominator_test.pkl'), 'rb') as f:
                events_denominator_test = pickle.load(f)

            self.testing_data = BalancedDataset(events_numerator_test, events_denominator_test, self.features, scaler = self.scaler, random_state=self.random_state)

    # def train_dataloader(self):
    #     return DataLoader(self.training_data, batch_size=self.batch_size, num_workers=8)

    # def val_dataloader(self):
    #     return DataLoader(self.validation_data, batch_size=self.batch_size, num_workers=8)

    # def test_dataloader(self):
    #     return DataLoader(self.testing_data, batch_size=self.batch_size, num_workers=8)

    def train_dataloader(self):
        return DataLoader(
            self.training_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testing_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
        )


class BalancedDataset(Dataset):
    def __init__(self, events_numerator = None, events_denominator = None, features = None, scaler = None, random_state = None):
        super().__init__()

        # get features
        X_numerator = events_numerator.kinematics[features].to_numpy()
        X_denominator = events_denominator.kinematics[features].to_numpy()
        self.X = np.concatenate([X_numerator, X_denominator])

        # balanced weights
        w_numerator = events_numerator.weights.to_numpy()
        w_denominator = events_denominator.weights.to_numpy()
        w_numerator /= w_numerator.sum()
        w_denominator /= w_denominator.sum()
        self.w = np.concatenate([w_numerator, w_denominator])

        # numerator = signal = 1, denominator = background = 0
        self.s = np.concatenate([np.ones_like(w_numerator), np.zeros_like(w_denominator)])

        if scaler is not None:
            self.X = scaler.transform(self.X)
        
        # self.X, self.s, self.w = shuffle(self.X, self.s, self.w, random_state=random_state)
        self.X, self.s, self.w = shuffle(self.X, self.s, self.w, random_state=42)
    
    def __len__(self):
        return len(self.s)

    def __getitem__(self, index):
        return torch.tensor(self.X[index], dtype=torch.float32), torch.tensor(self.s[index], dtype=torch.float32), torch.tensor(self.w[index], dtype=torch.float32)
