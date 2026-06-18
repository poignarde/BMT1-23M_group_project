import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import CGConv, global_mean_pool
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ================= НАСТРОЙКИ =================
DATA_DIR = './dataset'
# MODEL_PATH = 'best_model.pth'
MODEL_PATH = 'best_model_tuned.pth'
BATCH_SIZE = 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# =============================================

# Копируем классы Normalizer и PhysicsGNN, чтобы скрипт был независимым
class Normalizer:
    def __init__(self):
        self.min_val = None
        self.max_val = None

    def fit(self, dataset):
        y_all = torch.cat([data.y for data in dataset], dim=0)
        self.min_val = y_all.min(dim=0)[0].to(DEVICE)
        self.max_val = y_all.max(dim=0)[0].to(DEVICE)

    def inverse_transform(self, y_norm):
        return y_norm * (self.max_val - self.min_val + 1e-6) + self.min_val

class PhysicsGNN(nn.Module):
    def __init__(self, hidden_dim=32):
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

def main():
    print("Загрузка данных...")
    # Нам нужен Train датасет только для того, чтобы восстановить min/max для нормализатора
    train_dataset = torch.load(os.path.join(DATA_DIR, 'train.pt'), weights_only=False)
    test_dataset = torch.load(os.path.join(DATA_DIR, 'test.pt'), weights_only=False)

    normalizer = Normalizer()
    normalizer.fit(train_dataset)
    del train_dataset # Удаляем из ОЗУ, чтобы не занимать память

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Загрузка модели...")
    model = PhysicsGNN(hidden_dim=32).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Списки для сбора результатов
    y_true_list = []
    y_pred_list = []

    print("Тестирование модели (Inference)...")
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            # Сеть выдает нормализованные предсказания [0, 1]
            out_norm = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.rho)

            # Возвращаем в реальные физические единицы
            out_phys = normalizer.inverse_transform(out_norm)

            y_true_list.append(batch.y.cpu().numpy())
            y_pred_list.append(out_phys.cpu().numpy())

    # Объединяем батчи в единые массивы numpy
    y_true = np.vstack(y_true_list)
    y_pred = np.vstack(y_pred_list)

    # ================= РАСЧЕТ МЕТРИК =================
    # Усреднение предсказаний (Ensemble averaging) по целым симуляциям
    PATCHES_PER_FILE = 4 # Укажите то число, которое было в prepare_dataset.py

    num_simulations = len(y_true) // PATCHES_PER_FILE

    # Берем истинные значения (они одинаковые для всех 4 патчей одной симуляции, берем первое)
    y_true_sim = y_true[::PATCHES_PER_FILE]

    # Усредняем предсказания сети по 4 патчам для каждой симуляции
    # Меняем форму массива: [N_patches, 2] -> [N_simulations, 4, 2] и берем среднее по оси 1
    y_pred_sim = y_pred.reshape(num_simulations, PATCHES_PER_FILE, 2).mean(axis=1)

    h_true, p_true = y_true_sim[:, 0], y_true_sim[:, 1]
    h_pred, p_pred = y_pred_sim[:, 0], y_pred_sim[:, 1]

    metrics = {
        'h': {
            'R2': r2_score(h_true, h_pred),
            'MAE': mean_absolute_error(h_true, h_pred),
            'RMSE': np.sqrt(mean_squared_error(h_true, h_pred))
        },
        'p': {
            'R2': r2_score(p_true, p_pred),
            'MAE': mean_absolute_error(p_true, p_pred),
            'RMSE': np.sqrt(mean_squared_error(p_true, p_pred))
        }
    }

    print("\nИтоговые метрики на TEST выборке (с усреднением по целым симуляциям):")
    print(f"Высота (h): R^2 = {metrics['h']['R2']:.4f} | MAE = {metrics['h']['MAE']:.4f} | RMSE = {metrics['h']['RMSE']:.4f}")
    print(f"Диполь (p): R^2 = {metrics['p']['R2']:.4f} | MAE = {metrics['p']['MAE']:.4f} | RMSE = {metrics['p']['RMSE']:.4f}")

    # ================= ПОСТРОЕНИЕ ГРАФИКОВ =================
    plt.style.use('seaborn-v0_8-whitegrid') # Красивый стиль графиков
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    def plot_parity(ax, true, pred, name, symbol, metric_dict):
        # Рисуем точки (alpha=0.3 делает скопления точек более заметными)
        ax.scatter(true, pred, alpha=0.3, edgecolors='k', s=20)

        # Рисуем идеальную диагональ y = x
        min_val, max_val = np.min(true), np.max(true)
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Идеал (y=x)')

        # Добавляем текст с метриками
        textstr = '\n'.join((
            rf'$R^2=%.3f$' % (metric_dict['R2'], ),
            rf'$MAE=%.3f$' % (metric_dict['MAE'], ),
            rf'$RMSE=%.3f$' % (metric_dict['RMSE'], )))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)

        ax.set_title(f'Предсказание: {name}', fontsize=14, fontweight='bold')
        ax.set_xlabel(f'Истинное значение ${symbol}$ (Ground Truth)', fontsize=12)
        ax.set_ylabel(f'Предсказание GNN ${symbol}$', fontsize=12)
        ax.legend(loc='lower right')
        ax.grid(True)

    plot_parity(axes[0], h_true, h_pred, 'Высота конфайнмента', 'h', metrics['h'])
    plot_parity(axes[1], p_true, p_pred, 'Дипольный момент', 'p', metrics['p'])

    plt.tight_layout()
    plt.savefig('results_parity_plot.png', dpi=300)
    print("\nГрафик сохранен как 'results_parity_plot.png'")
    plt.show()

if __name__ == "__main__":
    main()
