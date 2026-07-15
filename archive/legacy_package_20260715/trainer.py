"""Archived pre-unified trainer; not part of the public package."""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from .dataset import TimeAwareMultiStationDataset
from . import visualization


def train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, patience, output_img_folder):
    checkpoint_latest_path = os.path.join(output_img_folder, "checkpoint_latest.pth")
    checkpoint_best_path = os.path.join(output_img_folder, "checkpoint_best.pth")

    start_epoch = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0

    if os.path.exists(checkpoint_latest_path):
        print(f"\n🔄 检测到本地存在中断的训练记录 ({checkpoint_latest_path})，正在无缝恢复...")
        checkpoint = torch.load(checkpoint_latest_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        epochs_no_improve = checkpoint['epochs_no_improve']
        print(f"✅ 成功从第 {start_epoch} 个 Epoch 恢复训练！历史最佳验证集 MSE 为: {best_val_loss:.4f}")
    else:
        print(f"\n🆕 未检测到历史进度，将从头开始全新的训练。")

    if start_epoch < num_epochs and epochs_no_improve < patience:
        print(f"🚀 开始训练...\n" + "-" * 40)
        for epoch in range(start_epoch, num_epochs):
            model.train()
            train_loss = 0

            for batch_forcing, batch_state, batch_time, batch_static, batch_y, _ in train_loader:
                batch_forcing, batch_state, batch_time = batch_forcing.to(device), batch_state.to(
                    device), batch_time.to(device)
                batch_static, batch_y = batch_static.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_forcing, batch_state, batch_time, batch_static)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_forcing, batch_state, batch_time, batch_static, batch_y, _ in val_loader:
                    batch_forcing, batch_state, batch_time = batch_forcing.to(device), batch_state.to(
                        device), batch_time.to(device)
                    batch_static, batch_y = batch_static.to(device), batch_y.to(device)

                    outputs = model(batch_forcing, batch_state, batch_time, batch_static)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            print(
                f"Epoch [{epoch + 1:03d}/{num_epochs}] | Train MSE: {avg_train_loss:.4f} | Val MSE: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                print(f"   🌟 发现新的最佳模型！验证集 MSE 从 {best_val_loss:.4f} 降至 {avg_val_loss:.4f}。")
                print(f"   💾 最佳参数已硬拷贝至本地: {checkpoint_best_path}")
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), checkpoint_best_path)
            else:
                epochs_no_improve += 1
                print(f"   ⏳ 验证集 MSE 未改善 (早停计数: {epochs_no_improve}/{patience})")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'epochs_no_improve': epochs_no_improve
            }, checkpoint_latest_path)

            if epochs_no_improve >= patience:
                print(f"\n🛑 触发早停机制，在连续 {patience} 个 Epoch 中验证集表现未提升，训练提前结束。")
                break
            print("-" * 40)

    return model


def evaluate_and_test(model, test_files, seq_len, batch_size, device, output_img_folder,
                      forcing_cols, state_cols, static_cols, scalers, time_column_name):
    checkpoint_best_path = os.path.join(output_img_folder, "checkpoint_best.pth")
    if os.path.exists(checkpoint_best_path):
        print(f"\n🎯 训练结束，正在加载全局表现最好的模型参数进行测试集评估...")
        model.load_state_dict(torch.load(checkpoint_best_path, map_location=device))

    model.eval()
    global_all_preds, global_all_targets = [], []

    for test_file in test_files:
        station_name = os.path.basename(test_file).replace('.csv', '')

        single_test_dataset = TimeAwareMultiStationDataset(
            [test_file], seq_len, time_col=time_column_name,
            forcing_cols=forcing_cols, state_cols=state_cols, static_cols=static_cols,
            feat_min_f=scalers['f_min'], feat_max_f=scalers['f_max'],
            feat_min_s=scalers['s_min'], feat_max_s=scalers['s_max'],
            static_min=scalers['st_min'], static_max=scalers['st_max']
        )

        if len(single_test_dataset) == 0: continue

        single_test_loader = DataLoader(single_test_dataset, batch_size=batch_size, shuffle=False)
        all_preds, all_targets, all_times = [], [], []

        with torch.no_grad():
            for batch_forcing, batch_state, batch_time, batch_static, batch_y, batch_dt in single_test_loader:
                batch_forcing, batch_state, batch_time = batch_forcing.to(device), batch_state.to(
                    device), batch_time.to(device)
                batch_static = batch_static.to(device)

                outputs = model(batch_forcing, batch_state, batch_time, batch_static)
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(batch_y.numpy())
                all_times.extend(batch_dt)

        all_preds, all_targets = np.array(all_preds), np.array(all_targets)
        all_times = pd.to_datetime(all_times)

        global_all_preds.extend(all_preds)
        global_all_targets.extend(all_targets)

        # ------------------ 新增的 MSE 与格式化输出 ------------------
        station_mse = np.mean((all_targets - all_preds) ** 2)
        station_r2 = r2_score(all_targets, all_preds)
        print(f"📢 站点: {station_name:<12} | 测试集 MSE: {station_mse:.4f} | 测试集 R²: {station_r2:.4f}")
        # -------------------------------------------------------------

        # 调用抽离出来的可视化函数
        visualization.plot_trend_ma(all_times, all_targets, all_preds, station_name, 12, output_img_folder)
        visualization.plot_zoom_30days(all_times, all_targets, all_preds, station_name, output_img_folder)
        visualization.plot_scatter(all_targets, all_preds, station_name, output_img_folder)

    if len(global_all_targets) > 0:
        global_all_preds, global_all_targets = np.array(global_all_preds), np.array(global_all_targets)
        global_mse = np.mean((global_all_preds - global_all_targets) ** 2)
        global_r2 = r2_score(global_all_targets, global_all_preds)

        print("\n" + "=" * 50)
        print("🌎 所有站点测试集全局评估结果")
        print("=" * 50)
        print(f"总测试样本数: {len(global_all_targets)}")
        print(f"Global Test MSE: {global_mse:.4f}")
        print(f"Global Test R² Score: {global_r2:.4f}")
        print("=" * 50)
