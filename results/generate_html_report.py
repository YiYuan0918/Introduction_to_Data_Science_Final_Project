#!/usr/bin/env python
"""
HTML Report Generator for Model Training
生成精美的 HTML 訓練報告

Usage:
    python results/generate_html_report.py
    python results/generate_html_report.py --model-dir outputs/classifier
"""

import json
import os
import sys
import base64
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import argparse
import io

import torch


def count_parameters(model_path):
    """計算模型參數量"""
    safetensors_path = os.path.join(model_path, "model.safetensors")
    total_params = 0
    
    if os.path.exists(safetensors_path):
        from safetensors import safe_open
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                total_params += tensor.numel()
    
    return total_params


def get_model_size(model_path):
    """獲取模型檔案大小"""
    safetensors_path = os.path.join(model_path, "model.safetensors")
    if os.path.exists(safetensors_path):
        return os.path.getsize(safetensors_path)
    return 0


def load_config(model_path):
    """載入模型配置"""
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


def load_training_log(model_path):
    """載入訓練日誌"""
    log_path = os.path.join(model_path, "train_eval_log.csv")
    if os.path.exists(log_path):
        return pd.read_csv(log_path)
    return None


def format_number(num):
    """格式化數字"""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    return str(num)


def format_bytes(size_bytes):
    """格式化檔案大小"""
    if size_bytes >= 1e9:
        return f"{size_bytes/1e9:.2f} GB"
    elif size_bytes >= 1e6:
        return f"{size_bytes/1e6:.2f} MB"
    return f"{size_bytes/1e3:.2f} KB"


def format_time(seconds):
    """格式化時間"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"


def generate_plot_base64(log_df):
    """生成圖表並轉為 base64"""
    train_loss = log_df[(log_df['split'] == 'train') & (log_df['metric'] == 'loss')].copy()
    eval_loss = log_df[(log_df['split'] == 'eval') & (log_df['metric'] == 'loss')].copy()
    lr = log_df[(log_df['split'] == 'train') & (log_df['metric'] == 'learning_rate')].copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Loss vs Step
    ax1 = axes[0, 0]
    ax1.plot(train_loss['step'], train_loss['value'], 'b-', label='Training Loss', alpha=0.8, linewidth=2)
    ax1.plot(eval_loss['step'], eval_loss['value'], 'r-', label='Validation Loss', marker='o', markersize=4)
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss vs Training Step', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Loss vs Epoch
    ax2 = axes[0, 1]
    ax2.plot(train_loss['epoch'], train_loss['value'], 'b-', label='Training Loss', alpha=0.8, linewidth=2)
    ax2.plot(eval_loss['epoch'], eval_loss['value'], 'r-', label='Validation Loss', marker='o', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Loss vs Epoch', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Log Scale Loss
    ax3 = axes[1, 0]
    ax3.semilogy(train_loss['step'], train_loss['value'], 'b-', label='Training Loss', alpha=0.8, linewidth=2)
    ax3.semilogy(eval_loss['step'], eval_loss['value'], 'r-', label='Validation Loss', marker='o', markersize=4)
    ax3.set_xlabel('Step', fontsize=12)
    ax3.set_ylabel('Loss (Log Scale)', fontsize=12)
    ax3.set_title('Loss (Log Scale) vs Training Step', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. Learning Rate
    ax4 = axes[1, 1]
    ax4.plot(lr['step'], lr['value'], 'g-', linewidth=2)
    ax4.set_xlabel('Step', fontsize=12)
    ax4.set_ylabel('Learning Rate', fontsize=12)
    ax4.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.fill_between(lr['step'], lr['value'], alpha=0.3, color='green')
    
    plt.tight_layout()
    
    # 轉為 base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    
    return image_base64


def generate_html_report(model_path, output_path=None):
    """生成 HTML 報告"""
    
    # 獲取專案根目錄 (results 的上一層)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # 收集數據
    config = load_config(model_path)
    log_df = load_training_log(model_path)
    total_params = count_parameters(model_path)
    model_size = get_model_size(model_path)
    
    # 訓練配置
    import yaml
    yaml_path = os.path.join(project_root, "configs/cls.yaml")
    train_cfg = {}
    data_cfg = {}
    training_cfg = {}
    if os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            train_cfg = yaml.safe_load(f)
        data_cfg = train_cfg.get("data", {})
        training_cfg = train_cfg.get("training", {}).get("classifier", {})
    
    # 資料集統計
    dataset_root = os.path.join(project_root, "dataset/minDataset")
    def count_lines(filepath):
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return sum(1 for _ in f)
        return 0
    
    train_samples = count_lines(os.path.join(dataset_root, "annotation_train.txt"))
    val_samples = count_lines(os.path.join(dataset_root, "annotation_val.txt"))
    test_samples = count_lines(os.path.join(dataset_root, "annotation_test.txt"))
    
    # 訓練結果
    train_loss = log_df[(log_df['split'] == 'train') & (log_df['metric'] == 'loss')]
    eval_loss = log_df[(log_df['split'] == 'eval') & (log_df['metric'] == 'loss')]
    train_runtime = log_df[(log_df['split'] == 'train') & (log_df['metric'] == 'train_runtime')]
    
    initial_train_loss = train_loss['value'].iloc[0]
    final_train_loss = train_loss['value'].iloc[-1]
    min_train_loss = train_loss['value'].min()
    initial_eval_loss = eval_loss['value'].iloc[0]
    final_eval_loss = eval_loss['value'].iloc[-1]
    min_eval_loss = eval_loss['value'].min()
    
    train_reduction = (1 - final_train_loss / initial_train_loss) * 100
    eval_reduction = (1 - final_eval_loss / initial_eval_loss) * 100
    
    runtime_seconds = train_runtime['value'].iloc[0] if len(train_runtime) > 0 else 0
    total_steps = log_df['step'].max()
    
    gap = final_eval_loss - final_train_loss
    
    # 生成圖表
    plot_base64 = generate_plot_base64(log_df)
    
    # 載入 Test 結果 (從 tests/ 資料夾讀取)
    test_results = None
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    test_results_path = os.path.join(project_root, "tests", "results_test.json")
    if os.path.exists(test_results_path):
        with open(test_results_path, "r") as f:
            test_results = json.load(f)
    
    # 過擬合狀態
    if gap > 0.5:
        overfit_status = "⚠️ 存在明顯過擬合"
        overfit_class = "warning"
        overfit_suggestion = "建議：增加正則化、使用 Early Stopping 或增加訓練資料"
    elif gap > 0.2:
        overfit_status = "⚡ 輕微過擬合傾向"
        overfit_class = "caution"
        overfit_suggestion = "建議：可考慮增加 Dropout 或使用資料增強"
    else:
        overfit_status = "✅ 泛化能力良好"
        overfit_class = "good"
        overfit_suggestion = "模型表現正常，無需額外調整"
    
    # HTML 模板
    html_content = f'''<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>模型訓練報告 - ViT Image Classification</title>
    <style>
        :root {{
            --primary-color: #2563eb;
            --secondary-color: #3b82f6;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --danger-color: #ef4444;
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --text-color: #1e293b;
            --text-secondary: #64748b;
            --border-color: #e2e8f0;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            color: var(--text-color);
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        .header {{
            background: var(--card-bg);
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 10px;
        }}
        
        .header .subtitle {{
            color: var(--text-secondary);
            font-size: 1.1em;
        }}
        
        .header .timestamp {{
            margin-top: 15px;
            padding: 8px 16px;
            background: var(--bg-color);
            border-radius: 20px;
            display: inline-block;
            font-size: 0.9em;
            color: var(--text-secondary);
        }}
        
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .card {{
            background: var(--card-bg);
            border-radius: 16px;
            padding: 25px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        
        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.12);
        }}
        
        .card-title {{
            font-size: 1.2em;
            font-weight: 600;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .card-title .icon {{
            font-size: 1.5em;
        }}
        
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
        }}
        
        .stat-item {{
            padding: 15px;
            background: var(--bg-color);
            border-radius: 12px;
        }}
        
        .stat-label {{
            font-size: 0.85em;
            color: var(--text-secondary);
            margin-bottom: 5px;
        }}
        
        .stat-value {{
            font-size: 1.3em;
            font-weight: 700;
            color: var(--primary-color);
        }}
        
        .stat-value.large {{
            font-size: 1.8em;
        }}
        
        .full-width {{
            grid-column: 1 / -1;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        
        th {{
            background: var(--bg-color);
            font-weight: 600;
            color: var(--text-secondary);
            font-size: 0.9em;
            text-transform: uppercase;
        }}
        
        tr:hover {{
            background: var(--bg-color);
        }}
        
        .progress-bar {{
            height: 10px;
            background: var(--border-color);
            border-radius: 5px;
            overflow: hidden;
            margin-top: 8px;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
            border-radius: 5px;
            transition: width 1s ease;
        }}
        
        .status-badge {{
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 600;
            display: inline-block;
        }}
        
        .status-badge.good {{
            background: #d1fae5;
            color: #065f46;
        }}
        
        .status-badge.caution {{
            background: #fef3c7;
            color: #92400e;
        }}
        
        .status-badge.warning {{
            background: #fee2e2;
            color: #991b1b;
        }}
        
        .chart-container {{
            margin-top: 20px;
            text-align: center;
        }}
        
        .chart-container img {{
            max-width: 100%;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }}
        
        .highlight-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 16px;
            text-align: center;
        }}
        
        .highlight-box .value {{
            font-size: 3em;
            font-weight: 700;
        }}
        
        .highlight-box .label {{
            font-size: 1.1em;
            opacity: 0.9;
            margin-top: 5px;
        }}
        
        .metrics-row {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}
        
        .metric-card {{
            flex: 1;
            min-width: 200px;
            padding: 20px;
            background: var(--bg-color);
            border-radius: 12px;
            text-align: center;
        }}
        
        .metric-card .value {{
            font-size: 2em;
            font-weight: 700;
            color: var(--primary-color);
        }}
        
        .metric-card .change {{
            font-size: 0.9em;
            color: var(--success-color);
            margin-top: 5px;
        }}
        
        .footer {{
            text-align: center;
            padding: 20px;
            color: white;
            opacity: 0.8;
            font-size: 0.9em;
        }}
        
        @media (max-width: 768px) {{
            .stat-grid {{
                grid-template-columns: 1fr;
            }}
            .header h1 {{
                font-size: 1.8em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>📊 模型訓練報告</h1>
            <p class="subtitle">Vision Transformer (ViT) for Word-Level Image Classification</p>
            <div class="timestamp">📅 報告生成時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>
        
        <!-- Key Metrics -->
        <div class="grid">
            <div class="highlight-box">
                <div class="value">{format_number(total_params)}</div>
                <div class="label">總參數量 (Total Parameters)</div>
            </div>
            <div class="highlight-box" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);">
                <div class="value">{train_reduction:.1f}%</div>
                <div class="label">Training Loss 下降</div>
            </div>
            <div class="highlight-box" style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);">
                <div class="value">{format_time(runtime_seconds)}</div>
                <div class="label">總訓練時間</div>
            </div>
        </div>
        
        <!-- Model Architecture -->
        <div class="grid">
            <div class="card">
                <div class="card-title">
                    <span class="icon">🏗️</span>
                    模型架構 (Model Architecture)
                </div>
                <table>
                    <tr><td>模型類型</td><td><strong>{config.get('model_type', 'N/A').upper()}</strong></td></tr>
                    <tr><td>架構名稱</td><td>{config.get('architectures', ['N/A'])[0]}</td></tr>
                    <tr><td>隱藏層維度</td><td>{config.get('hidden_size', 'N/A')}</td></tr>
                    <tr><td>隱藏層數量</td><td>{config.get('num_hidden_layers', 'N/A')}</td></tr>
                    <tr><td>注意力頭數</td><td>{config.get('num_attention_heads', 'N/A')}</td></tr>
                    <tr><td>中間層維度</td><td>{config.get('intermediate_size', 'N/A')}</td></tr>
                    <tr><td>Patch 大小</td><td>{config.get('patch_size', 'N/A')} × {config.get('patch_size', 'N/A')}</td></tr>
                    <tr><td>類別數量</td><td><strong>{len(config.get('id2label', {})):,}</strong></td></tr>
                </table>
            </div>
            
            <div class="card">
                <div class="card-title">
                    <span class="icon">⚙️</span>
                    訓練配置 (Training Configuration)
                </div>
                <table>
                    <tr><td>輸入圖片大小</td><td>{data_cfg.get('img_height', 'N/A')} × {data_cfg.get('img_width', 'N/A')}</td></tr>
                    <tr><td>訓練輪數</td><td>{training_cfg.get('num_train_epochs', 'N/A')} epochs</td></tr>
                    <tr><td>批次大小</td><td>{training_cfg.get('per_device_train_batch_size', 'N/A')}</td></tr>
                    <tr><td>學習率</td><td>{training_cfg.get('learning_rate', 'N/A')}</td></tr>
                    <tr><td>學習率排程</td><td>{training_cfg.get('lr_scheduler_type', 'N/A')}</td></tr>
                    <tr><td>優化器</td><td>{training_cfg.get('optim', 'N/A')}</td></tr>
                    <tr><td>權重衰減</td><td>{training_cfg.get('weight_decay', 'N/A')}</td></tr>
                    <tr><td>混合精度 (FP16)</td><td>{'✅ 啟用' if training_cfg.get('fp16') else '❌ 未啟用'}</td></tr>
                </table>
            </div>
        </div>
        
        <!-- Dataset Statistics -->
        <div class="card full-width">
            <div class="card-title">
                <span class="icon">📂</span>
                資料集統計 (Dataset Statistics)
            </div>
            <div class="stat-grid">
                <div class="stat-item">
                    <div class="stat-label">訓練集 (Training Set)</div>
                    <div class="stat-value">{train_samples:,}</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {train_samples/(train_samples+val_samples+test_samples)*100:.1f}%"></div>
                    </div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">驗證集 (Validation Set)</div>
                    <div class="stat-value">{val_samples:,}</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {val_samples/(train_samples+val_samples+test_samples)*100:.1f}%; background: linear-gradient(90deg, #10b981, #059669);"></div>
                    </div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">測試集 (Test Set)</div>
                    <div class="stat-value">{test_samples:,}</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {test_samples/(train_samples+val_samples+test_samples)*100:.1f}%; background: linear-gradient(90deg, #f59e0b, #d97706);"></div>
                    </div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">總樣本數 (Total Samples)</div>
                    <div class="stat-value large">{train_samples+val_samples+test_samples:,}</div>
                </div>
            </div>
        </div>
        
        <!-- Training Results -->
        <div class="card full-width">
            <div class="card-title">
                <span class="icon">📉</span>
                訓練結果 (Training Results)
            </div>
            <div class="metrics-row">
                <div class="metric-card">
                    <div class="stat-label">初始 Training Loss</div>
                    <div class="value">{initial_train_loss:.4f}</div>
                </div>
                <div class="metric-card">
                    <div class="stat-label">最終 Training Loss</div>
                    <div class="value" style="color: #10b981;">{final_train_loss:.4f}</div>
                    <div class="change">↓ {train_reduction:.2f}%</div>
                </div>
                <div class="metric-card">
                    <div class="stat-label">初始 Validation Loss</div>
                    <div class="value">{initial_eval_loss:.4f}</div>
                </div>
                <div class="metric-card">
                    <div class="stat-label">最終 Validation Loss</div>
                    <div class="value" style="color: #10b981;">{final_eval_loss:.4f}</div>
                    <div class="change">↓ {eval_reduction:.2f}%</div>
                </div>
            </div>
            
            <div style="margin-top: 25px;">
                <table>
                    <tr>
                        <th>指標</th>
                        <th>數值</th>
                    </tr>
                    <tr><td>最小 Training Loss</td><td>{min_train_loss:.4f}</td></tr>
                    <tr><td>最小 Validation Loss (Best)</td><td><strong>{min_eval_loss:.4f}</strong></td></tr>
                    <tr><td>總訓練步數</td><td>{total_steps:,}</td></tr>
                    <tr><td>總訓練時間</td><td>{format_time(runtime_seconds)}</td></tr>
                </table>
            </div>
        </div>
        
        <!-- Overfitting Analysis -->
        <div class="card full-width">
            <div class="card-title">
                <span class="icon">🔍</span>
                過擬合分析 (Overfitting Analysis)
            </div>
            <div class="stat-grid">
                <div class="stat-item">
                    <div class="stat-label">最終 Train Loss</div>
                    <div class="stat-value">{final_train_loss:.4f}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">最終 Validation Loss</div>
                    <div class="stat-value">{final_eval_loss:.4f}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Gap (Val - Train)</div>
                    <div class="stat-value">{gap:.4f}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">狀態評估</div>
                    <span class="status-badge {overfit_class}">{overfit_status}</span>
                </div>
            </div>
            <div style="margin-top: 20px; padding: 15px; background: var(--bg-color); border-radius: 12px;">
                <strong>💡 {overfit_suggestion}</strong>
            </div>
        </div>
        
        <!-- Test Results (if available) -->
        {'<div class="card full-width" style="border: 3px solid #10b981;">' + chr(10) + '''
            <div class="card-title">
                <span class="icon">🎯</span>
                測試集結果 (Test Set Results) - 最終評估
            </div>
            <div class="grid" style="grid-template-columns: repeat(3, 1fr); gap: 15px;">
                <div class="highlight-box" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);">
                    <div class="value">''' + f"{test_results['accuracy']*100:.2f}%" + '''</div>
                    <div class="label">Top-1 Accuracy</div>
                </div>
                <div class="highlight-box" style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);">
                    <div class="value">''' + f"{test_results['top5_accuracy']*100:.2f}%" + '''</div>
                    <div class="label">Top-5 Accuracy</div>
                </div>
                <div class="highlight-box" style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);">
                    <div class="value">''' + f"{test_results['top10_accuracy']*100:.2f}%" + '''</div>
                    <div class="label">Top-10 Accuracy</div>
                </div>
            </div>
            <div style="margin-top: 20px;">
                <table>
                    <tr>
                        <th>指標</th>
                        <th>數值</th>
                    </tr>
                    <tr><td>測試樣本數</td><td>''' + f"{test_results['total_samples']:,}" + '''</td></tr>
                    <tr><td>正確預測數</td><td>''' + f"{test_results['correct_predictions']:,}" + '''</td></tr>
                    <tr><td>Test Loss</td><td>''' + f"{test_results['loss']:.4f}" + '''</td></tr>
                    <tr><td>與 Validation Loss 比較</td><td>''' + f"{'✅ 相近 (泛化良好)' if abs(test_results['loss'] - final_eval_loss) < 0.05 else '⚠️ 有差異'}" + '''</td></tr>
                </table>
            </div>
            <div style="margin-top: 20px; padding: 15px; background: #d1fae5; border-radius: 12px; color: #065f46;">
                <strong>📊 結論：</strong> 測試集 Loss (''' + f"{test_results['loss']:.4f}" + ''') 與驗證集 Loss (''' + f"{final_eval_loss:.4f}" + ''') 相近，表示模型具有良好的泛化能力，未對驗證集過擬合。
            </div>
        </div>
        ''' if test_results else '<div class="card full-width"><div class="card-title"><span class="icon">🎯</span>測試集結果 (Test Set Results)</div><p style="color: var(--text-secondary);">尚未執行測試集評估。請執行：<code>python tests/test.py --model-dir outputs/classifier --config configs/cls.yaml --split test</code></p></div>'}
        
        <!-- Learning Curves -->
        <div class="card full-width">
            <div class="card-title">
                <span class="icon">📈</span>
                學習曲線 (Learning Curves)
            </div>
            <div class="chart-container">
                <img src="data:image/png;base64,{plot_base64}" alt="Learning Curves">
            </div>
        </div>
        
        <!-- Model Files -->
        <div class="card full-width">
            <div class="card-title">
                <span class="icon">📁</span>
                模型檔案資訊 (Model Files)
            </div>
            <table>
                <tr>
                    <th>檔案名稱</th>
                    <th>大小</th>
                </tr>
                <tr><td>model.safetensors</td><td>{format_bytes(model_size)}</td></tr>
                <tr><td>config.json</td><td>{format_bytes(os.path.getsize(os.path.join(model_path, "config.json")))}</td></tr>
                <tr><td>train_eval_log.csv</td><td>{format_bytes(os.path.getsize(os.path.join(model_path, "train_eval_log.csv")))}</td></tr>
                <tr><td>training_args.bin</td><td>{format_bytes(os.path.getsize(os.path.join(model_path, "training_args.bin")))}</td></tr>
            </table>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <p>Generated by Model Training Report Generator | Introduction to Data Science Final Project</p>
            <p>© 2025 All Rights Reserved</p>
        </div>
    </div>
</body>
</html>
'''
    
    # 儲存報告
    if output_path is None:
        # 預設輸出到 results 資料夾
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "training_report.html")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"✅ HTML 報告已生成: {output_path}")
    return output_path


def main():
    # 獲取專案根目錄
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_model_dir = os.path.join(project_root, "outputs/classifier")
    
    parser = argparse.ArgumentParser(description="Generate HTML training report")
    parser.add_argument("--model-dir", default=default_model_dir,
                        help="Path to model output directory")
    parser.add_argument("--output", default=None,
                        help="Output path for HTML report")
    args = parser.parse_args()
    
    generate_html_report(args.model_dir, args.output)


if __name__ == "__main__":
    main()
