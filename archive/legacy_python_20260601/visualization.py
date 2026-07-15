import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from torch.utils.data import DataLoader

def plot_trend_ma(all_times, all_targets, all_preds, station_name, window_size, output_img_folder):
    all_targets_smooth = pd.Series(all_targets).rolling(window=window_size, min_periods=1).mean().values
    all_preds_smooth = pd.Series(all_preds).rolling(window=window_size, min_periods=1).mean().values

    plt.figure(figsize=(15, 6))
    plt.plot(all_times, all_targets, color='royalblue', linewidth=0.5, alpha=0.2, label='Actual (Raw)')
    plt.plot(all_times, all_preds, color='crimson', linewidth=0.5, alpha=0.2, label='Predicted (Raw)')
    plt.plot(all_times, all_targets_smooth, label=f'Actual GPP (MA-{window_size})', color='royalblue', linewidth=1.5, alpha=0.9)
    plt.plot(all_times, all_preds_smooth, label=f'Predicted GPP (MA-{window_size})', color='crimson', linewidth=1.5, linestyle='--', alpha=0.9)

    plt.title(f'[{station_name}] Test Set GPP Prediction (Moving Average Window={window_size})', fontsize=14, fontname='Arial')
    plt.xlabel('Time', fontsize=12, fontname='Arial')
    plt.ylabel('GPP Value', fontsize=12, fontname='Arial')
    plt.legend(prop={'family': 'Arial'})

    ax = plt.gca()
    ax.tick_params(direction='in')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_folder, f"{station_name}_trend_ma.png"), dpi=300)
    plt.close()

def plot_zoom_30days(all_times, all_targets, all_preds, station_name, output_img_folder):
    window_size = 12
    all_targets_smooth = pd.Series(all_targets).rolling(window=window_size, min_periods=1).mean().values
    all_preds_smooth = pd.Series(all_preds).rolling(window=window_size, min_periods=1).mean().values

    zoom_days, steps_per_day = 30, 24
    zoom_steps = min(zoom_days * steps_per_day, len(all_times))

    if zoom_steps > 0:
        peak_idx = np.argmax(all_targets_smooth)
        start_idx = max(0, peak_idx - zoom_steps // 2)
        end_idx = min(len(all_times), start_idx + zoom_steps)

        if end_idx - start_idx < zoom_steps:
            start_idx = max(0, end_idx - zoom_steps)

        plt.figure(figsize=(15, 5))
        plt.plot(all_times[start_idx:end_idx], all_targets_smooth[start_idx:end_idx],
                 label='Actual GPP', color='royalblue', linewidth=2)
        plt.plot(all_times[start_idx:end_idx], all_preds_smooth[start_idx:end_idx],
                 label='Predicted GPP', color='crimson', linewidth=2, linestyle='--')

        peak_date_str = all_times[peak_idx].strftime('%Y-%m-%d %H:%M')
        plt.title(f'[{station_name}] 30-Day Zoomed Prediction (Peak around {peak_date_str})', fontsize=14, fontname='Arial')
        plt.xlabel('Time', fontsize=12, fontname='Arial')
        plt.ylabel('GPP Value', fontsize=12, fontname='Arial')
        plt.legend(prop={'family': 'Arial'})

        ax = plt.gca()
        ax.tick_params(direction='in')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(os.path.join(output_img_folder, f"{station_name}_zoom_30days.png"), dpi=300)
        plt.close()

def plot_scatter(all_targets, all_preds, station_name, output_img_folder):
    plt.figure(figsize=(6, 6))
    plt.scatter(all_targets, all_preds, alpha=0.6, color='teal', s=15, edgecolors='k', linewidth=0.2)

    min_val = min(all_targets.min(), all_preds.min())
    max_val = max(all_targets.max(), all_preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 Line')

    plt.title(f'[{station_name}] Actual vs Predicted Scatter', fontname='Arial')
    plt.xlabel('Actual GPP', fontname='Arial')
    plt.ylabel('Predicted GPP', fontname='Arial')
    plt.legend(prop={'family': 'Arial'})

    ax = plt.gca()
    ax.tick_params(direction='in')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_folder, f"{station_name}_scatter.png"), dpi=300)
    plt.close()

def perform_shap_analysis(model, dataloader, device, output_img_folder, forcing_cols, state_cols):
    print("\n🔍 开始执行 SHAP 变量重要性分析 (启用随机采样与时序展平)...")
    model.eval()

    shap_loader = DataLoader(dataloader.dataset, batch_size=120, shuffle=True)
    batch = next(iter(shap_loader))

    batch_forcing, batch_state, batch_time = batch[0].to(device), batch[1].to(device), batch[2].to(device)
    batch_static = batch[3].to(device)

    bg_size, test_size = 50, 50
    if batch_forcing.size(0) < (bg_size + test_size):
        print("⚠️ Batch size 太小，跳过 SHAP 分析。")
        return

    bg_f, bg_s = batch_forcing[:bg_size], batch_state[:bg_size]
    test_f, test_s = batch_forcing[bg_size:bg_size+test_size], batch_state[bg_size:bg_size+test_size]

    class SHAPWrapper(nn.Module):
        def __init__(self, base_model, t_ref, st_ref):
            super().__init__()
            self.base_model = base_model
            self.t_ref = t_ref[:1]
            self.st_ref = st_ref[:1]

        def forward(self, x_f, x_s):
            b_size = x_f.size(0)
            t_x = self.t_ref.expand(b_size, -1, -1)
            x_st = self.st_ref.expand(b_size, -1, -1)
            out = self.base_model(x_f, x_s, t_x, x_st)
            return out.unsqueeze(-1)

    wrapper_model = SHAPWrapper(model, batch_time, batch_static).to(device)
    wrapper_model.eval()

    explainer = shap.GradientExplainer(wrapper_model, [bg_f, bg_s])
    shap_values = explainer.shap_values([test_f, test_s])

    if isinstance(shap_values, list) and len(shap_values) == 1 and isinstance(shap_values[0], list):
        shap_values = shap_values[0]

    shap_forcing = np.array(shap_values[0])
    shap_state = np.array(shap_values[1])

    shap_forcing_2d = shap_forcing.reshape(-1, len(forcing_cols))
    shap_state_2d = shap_state.reshape(-1, len(state_cols))

    test_f_2d = test_f.cpu().numpy().reshape(-1, len(forcing_cols))
    test_s_2d = test_s.cpu().numpy().reshape(-1, len(state_cols))

    shap_combined = np.concatenate([shap_forcing_2d, shap_state_2d], axis=1)
    features_combined = np.concatenate([test_f_2d, test_s_2d], axis=1)
    feature_names = forcing_cols + state_cols

    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_combined, features_combined, feature_names=feature_names, show=False)
    plt.title("SHAP Summary: Global Feature Impact on GPP (Time-Flattened)", fontname='Arial', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_folder, "SHAP_Summary_Plot.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_combined, features_combined, feature_names=feature_names, plot_type="bar", show=False)
    plt.title("SHAP Global Feature Importance (Magnitude)", fontname='Arial', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_folder, "SHAP_Bar_Plot.png"), dpi=300)
    plt.close()

    print(f"✅ SHAP 彻底修复！生成了 {shap_combined.shape[0]} 个时序数据点的蜂群图，已保存至: {output_img_folder}")