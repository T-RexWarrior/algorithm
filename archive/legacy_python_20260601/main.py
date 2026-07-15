import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import TimeAwareMultiStationDataset
from models import TCN_Transformer_CrossAttention
from trainer import train_model, evaluate_and_test
from visualization import perform_shap_analysis


def main():
    # 1. 超参数配置
    seq_len = 96
    batch_size = 64
    num_epochs = 100
    learning_rate = 0.001
    patience = 10
    time_column_name = 'date'

    # 自动识别环境设备 (RTX 5060 Ti)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 路径与特征配置
    data_folder = r"C:\Users\Admin\Desktop\实验三\全波段全变量DT"
    output_img_folder = os.path.join(data_folder, "Results_TCN_Transformer_NoIGBP")
    os.makedirs(output_img_folder, exist_ok=True)

    forcing_cols = ['SW_IN_F', 'SW_IN_POT', 'CO2_F_MDS', 'P_F', 'VPD_F', 'TA_F', 'TS_F_MDS_1', 'SWC_F_MDS_1', 'WS_F']
    state_cols = ['EPIC_Available_Mask', 'Band317nm_Ref', 'Band325nm_Ref', 'Band340nm_Ref',
                  'Band388nm_Ref', 'Band443nm_Ref', 'Band551nm_Ref', 'Band680nm_Ref',
                  'Band688nm_Ref', 'Band764nm_Ref', 'Band780nm_Ref', 'Mean_SZA', 'Mean_VZA', 'Mean_RAA']
    static_cols = ['Lat', 'Long']

    all_files = glob.glob(os.path.join(data_folder, "*.csv"))
    if not all_files:
        print("❌ 错误：未在指定目录找到CSV文件。")
        return

    print(f"📁 共找到 {len(all_files)} 个站点文件。")
    print(f"📁 结果图表与模型将保存至: {output_img_folder}")

    # 3. 加载训练集并获取归一化参数
    train_dataset = TimeAwareMultiStationDataset(
        all_files, seq_len, time_col=time_column_name,
        forcing_cols=forcing_cols, state_cols=state_cols,
        static_cols=static_cols, split_type='train'
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 打包归一化参数，保证验证集和测试集标准一致
    scalers = {
        'f_min': train_dataset.feat_min_f, 'f_max': train_dataset.feat_max_f,
        's_min': train_dataset.feat_min_s, 's_max': train_dataset.feat_max_s,
        'st_min': train_dataset.static_min, 'st_max': train_dataset.static_max
    }

    # 4. 加载验证集
    val_dataset = TimeAwareMultiStationDataset(
        all_files, seq_len, time_col=time_column_name,
        forcing_cols=forcing_cols, state_cols=state_cols, static_cols=static_cols,
        feat_min_f=scalers['f_min'], feat_max_f=scalers['f_max'],
        feat_min_s=scalers['s_min'], feat_max_s=scalers['s_max'],
        static_min=scalers['st_min'], static_max=scalers['st_max'], split_type='val'
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 5. 初始化模型与优化器
    model = TCN_Transformer_CrossAttention(
        num_forcing_features=len(forcing_cols),
        num_state_features=len(state_cols),
        seq_len=seq_len,
        num_static=len(static_cols)
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 6. 训练模型
    model = train_model(model, train_loader, val_loader, optimizer, criterion,
                        device, num_epochs, patience, output_img_folder)

    # 7. 测试模型与绘图
    evaluate_and_test(model, all_files, seq_len, batch_size, device, output_img_folder,
                      forcing_cols, state_cols, static_cols, scalers, time_column_name)

    # 8. SHAP 可解释性分析
    try:
        perform_shap_analysis(model, val_loader, device, output_img_folder, forcing_cols, state_cols)
    except Exception as e:
        print(f"⚠️ SHAP 分析过程中出现异常: {e}")


if __name__ == "__main__":
    main()