import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import CGConv, global_mean_pool
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import gc

# ================= НАСТРОЙКИ ПЕРЕБОРА =================
DATA_DIR = './dataset'
EPOCHS = 20           # Для тюнинга можно взять поменьше эпох, чтобы сэкономить время
PATCHES_PER_FILE = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Сетка гиперпараметров
HIDDEN_DIMS = [32, 64]
LEARNING_RATES = [0.005, 0.001]
BATCH_SIZES = [16, 32]
# ======================================================

class Normalizer:
    def __init__(self):
        self.min_val = None
        self.max_val = None

    def fit(self, dataset):
        y_all = torch.cat([data.y for data in dataset], dim=0)
        self.min_val = y_all.min(dim=0)[0].to(DEVICE)
        self.max_val = y_all.max(dim=0)[0].to(DEVICE)

    def transform(self, y):
        return (y - self.min_val) / (self.max_val - self.min_val + 1e-6)

    def inverse_transform(self, y_norm):
        return y_norm * (self.max_val - self.min_val + 1e-6) + self.min_val

class PhysicsGNN(nn.Module):
    def __init__(self, hidden_dim):
        super(PhysicsGNN, self).__init__()
        self.node_emb = nn.Linear(1, hidden_dim)
        self.conv1 = CGConv(hidden_dim, dim=4, batch_norm=True)
        self.conv2 = CGConv(hidden_dim, dim=4, batch_norm=True)
        self.conv3 = CGConv(hidden_dim, dim=4, batch_norm=True)
        self.fc1 = nn.Linear(hidden_dim + 1, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x, edge_index, edge_attr, batch_idx, rho):
        x = F.relu(self.node_emb(x))
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x_graph = global_mean_pool(x, batch_idx)
        rho = rho.view(-1, 1)
        x_combined = torch.cat([x_graph, rho], dim=1)
        out = F.relu(self.fc1(x_combined))
        return self.fc2(out)

def train_and_evaluate(hidden_dim, lr, batch_size, train_data, val_data, test_data, normalizer, exp_id):
    print(f"\n[{exp_id}] СТАРТ ЭКСПЕРИМЕНТА | HDim: {hidden_dim} | LR: {lr} | Batch: {batch_size}")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    model = PhysicsGNN(hidden_dim=hidden_dim).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Для построения графиков
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_model_state = None

    # === ЦИКЛ ОБУЧЕНИЯ ===
    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            y_norm = normalizer.transform(batch.y)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.rho)
            loss = criterion(out, y_norm)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs
        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)

        # Val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                y_norm = normalizer.transform(batch.y)
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.rho)
                loss = criterion(out, y_norm)
                val_loss += loss.item() * batch.num_graphs
        val_loss /= len(val_loader.dataset)
        history['val_loss'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

        print(f"  Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # === ТЕСТИРОВАНИЕ ЛУЧШЕЙ МОДЕЛИ ===
    model.load_state_dict(best_model_state)
    model.eval()

    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            out_norm = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.rho)
            out_phys = normalizer.inverse_transform(out_norm)
            y_true_list.append(batch.y.cpu().numpy())
            y_pred_list.append(out_phys.cpu().numpy())

    y_true = np.vstack(y_true_list)
    y_pred = np.vstack(y_pred_list)

    # Усреднение по симуляциям (как обсуждали!)
    num_sims = len(y_true) // PATCHES_PER_FILE
    y_true_sim = y_true[::PATCHES_PER_FILE]
    y_pred_sim = y_pred.reshape(num_sims, PATCHES_PER_FILE, 2).mean(axis=1)

    h_true, p_true = y_true_sim[:, 0], y_true_sim[:, 1]
    h_pred, p_pred = y_pred_sim[:, 0], y_pred_sim[:, 1]

    metrics = {
        'Exp_ID': exp_id,
        'Hidden_Dim': hidden_dim,
        'LR': lr,
        'Batch_Size': batch_size,
        'Val_Loss_Norm': best_val_loss,
        'R2_h': r2_score(h_true, h_pred),
        'MAE_h': mean_absolute_error(h_true, h_pred),
        'R2_p': r2_score(p_true, p_pred),
        'MAE_p': mean_absolute_error(p_true, p_pred)
    }

    return metrics, history

def main():
    print("Загрузка данных (в память один раз для всех экспериментов)...")
    train_data = torch.load(os.path.join(DATA_DIR, 'train.pt'), weights_only=False)
    val_data = torch.load(os.path.join(DATA_DIR, 'val.pt'), weights_only=False)
    test_data = torch.load(os.path.join(DATA_DIR, 'test.pt'), weights_only=False)

    normalizer = Normalizer()
    normalizer.fit(train_data)

    results = []
    histories = {}
    exp_counter = 1

    # Grid Search
    for h_dim in HIDDEN_DIMS:
        for lr in LEARNING_RATES:
            for bs in BATCH_SIZES:
                exp_id = f"Exp_{exp_counter}"

                # Запускаем обучение и сбор метрик
                metrics, history = train_and_evaluate(h_dim, lr, bs, train_data, val_data, test_data, normalizer, exp_id)

                results.append(metrics)
                histories[exp_id] = history
                exp_counter += 1

                # Очистка видеопамяти после каждого эксперимента
                torch.cuda.empty_cache()
                gc.collect()

    # ================= 1. СОХРАНЕНИЕ ТАБЛИЦЫ =================
    df = pd.DataFrame(results)
    # Сортируем таблицу по лучшей метрике R2 для дипольного момента
    df = df.sort_values(by='R2_p', ascending=False).round(4)

    df.to_csv('hyperparameters_results.csv', index=False)
    print("\n" + "="*50)
    print("ТАБЛИЦА РЕЗУЛЬТАТОВ (отсортирована по лучшему R2_p):")
    print(df.to_string(index=False))
    print("="*50)
    print("Таблица сохранена в 'hyperparameters_results.csv'")

    # ================= 2. ПОСТРОЕНИЕ КРИВЫХ ОБУЧЕНИЯ (Loss Curves) =================
    num_exps = len(results)
    cols = 4
    rows = (num_exps + cols - 1) // cols

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for idx, row in df.iterrows():
        exp_id = row['Exp_ID']
        ax = axes[idx]
        hist = histories[exp_id]

        epochs = range(1, len(hist['train_loss']) + 1)
        ax.plot(epochs, hist['train_loss'], label='Train Loss', color='blue', lw=2)
        ax.plot(epochs, hist['val_loss'], label='Val Loss', color='orange', lw=2)

        ax.set_title(f"{exp_id}: HD={row['Hidden_Dim']}, LR={row['LR']}, BS={row['Batch_Size']}\n$R^2_p={row['R2_p']:.3f}$", fontsize=10)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss (Norm)')
        ax.legend()
        ax.grid(True)

    # Удаляем пустые графики, если сетка не кратна cols
    for i in range(num_exps, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig('loss_curves_grid.png', dpi=300)
    print("Графики кривых обучения сохранены в 'loss_curves_grid.png'")

if __name__ == "__main__":
    main()
