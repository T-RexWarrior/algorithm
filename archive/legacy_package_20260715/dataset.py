"""Archived pre-unified dataset; not part of the public package."""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class TimeAwareMultiStationDataset(Dataset):
    # 移除了原有的 split_type='train' 参数
    def __init__(self, filepaths, seq_len=24, target_col='GPP_DT_VUT_REF', time_col='date',
                 forcing_cols=None, state_cols=None,
                 static_cols=['Lat', 'Long'],
                 feat_min_f=None, feat_max_f=None, feat_min_s=None, feat_max_s=None,
                 static_min=None, static_max=None, split_type='train'):

        self.seq_len = seq_len
        self.split_type = split_type
        self.samples = []

        self.station_forcing = []
        self.station_state = []
        self.station_time_features = []
        self.station_targets = []
        self.station_dates = []
        self.station_static = []

        for filepath in filepaths:
            data = pd.read_csv(filepath)
            if time_col not in data.columns:
                raise ValueError(f"在文件 {filepath} 中找不到时间列 '{time_col}'")

            data[time_col] = pd.to_datetime(data[time_col])
            data = data.sort_values(by=time_col).reset_index(drop=True)

            # ====== 删除了原有的 train_end, val_end 和切片逻辑 ======
            # 现在默认使用传入的整个 .csv 文件数据
            if len(data) < seq_len:
                print(f"⚠️ 警告: 文件 {filepath} 数据太短，已跳过。")
                continue

            dates = data[time_col]
            self.station_dates.append(dates.values)
            # ====== 后面的提取时间和归一化逻辑保持完全不变 ======
            hours = dates.dt.hour.values
            days = dates.dt.dayofyear.values
            time_feats = np.column_stack([
                np.sin(2 * np.pi * hours / 24.0), np.cos(2 * np.pi * hours / 24.0),
                np.sin(2 * np.pi * days / 365.25), np.cos(2 * np.pi * days / 365.25)
            ])

            forcing_data = data[forcing_cols].values
            state_data = data[state_cols].values
            static_data = data[static_cols].values

            if target_col in data.columns:
                targets = data[target_col].values
            else:
                targets = data.iloc[:, -1].values

            self.station_forcing.append(forcing_data)
            self.station_state.append(state_data)
            self.station_time_features.append(time_feats)
            self.station_targets.append(targets)
            self.station_static.append(static_data)

        if not self.station_forcing:
            raise ValueError(f"加载 {split_type} 数据失败，可能数据太短或文件列表为空。")

        all_forcing_concat = np.vstack(self.station_forcing)
        all_state_concat = np.vstack(self.station_state)
        all_static_concat = np.vstack(self.station_static)

        self.feat_min_f = np.min(all_forcing_concat, axis=0) if feat_min_f is None else feat_min_f
        self.feat_max_f = np.max(all_forcing_concat, axis=0) if feat_max_f is None else feat_max_f
        self.feat_min_s = np.min(all_state_concat, axis=0) if feat_min_s is None else feat_min_s
        self.feat_max_s = np.max(all_state_concat, axis=0) if feat_max_s is None else feat_max_s
        self.static_min = np.min(all_static_concat, axis=0) if static_min is None else static_min
        self.static_max = np.max(all_static_concat, axis=0) if static_max is None else static_max

        for i in range(len(self.station_forcing)):
            self.station_forcing[i] = (self.station_forcing[i] - self.feat_min_f) / (self.feat_max_f - self.feat_min_f + 1e-8)
            self.station_state[i] = (self.station_state[i] - self.feat_min_s) / (self.feat_max_s - self.feat_min_s + 1e-8)
            self.station_static[i] = (self.station_static[i] - self.static_min) / (self.static_max - self.static_min + 1e-8)

            num_samples = len(self.station_forcing[i]) - seq_len + 1
            if num_samples > 0:
                for j in range(num_samples):
                    self.samples.append((i, j))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        station_idx, start_idx = self.samples[idx]

        x_forcing = self.station_forcing[station_idx][start_idx : start_idx + self.seq_len]
        x_state = self.station_state[station_idx][start_idx : start_idx + self.seq_len]
        time_x = self.station_time_features[station_idx][start_idx : start_idx + self.seq_len]
        y = self.station_targets[station_idx][start_idx + self.seq_len - 1]
        target_date = self.station_dates[station_idx][start_idx + self.seq_len - 1]
        x_static = self.station_static[station_idx][start_idx : start_idx + self.seq_len]

        return (
            torch.tensor(x_forcing, dtype=torch.float32),
            torch.tensor(x_state, dtype=torch.float32),
            torch.tensor(time_x, dtype=torch.float32),
            torch.tensor(x_static, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            str(target_date)
        )
