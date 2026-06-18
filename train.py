import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import CGConv, global_mean_pool
from tqdm import tqdm
import os

# ================= НАСТРОЙКИ =================
DATA_DIR = './dataset'
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.005
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# =============================================

class Normalizer:
    """Класс для нормализации таргетов (h, p) в диапазон [0, 1] и обратно"""
    def __init__(self):
        self.min_val = None
        self.max_val = None

    def fit(self, dataset):
        y_all = torch.cat([data.y for data in dataset], dim=0)
        self.min_val = y_all.min(dim=0)[0].to(DEVICE)
        self.max_val = y_all.max(dim=0)[0].to(DEVICE)
        print(f"Target H bounds: [{self.min_val[0]:.3f}, {self.max_val[0]:.3f}]")
        print(f"Target P bounds: [{self.min_val[1]:.3f}, {self.max_val[1]:.3f}]")

    def transform(self, y):
        # Добавляем 1e-6 для защиты от деления на ноль
        return (y - self.min_val) / (self.max_val - self.min_val + 1e-6)

    def inverse_transform(self, y_norm):
        return y_norm * (self.max_val - self.min_val + 1e-6) + self.min_val


class PhysicsGNN(nn.Module):
    def __init__(self, hidden_dim=32):
        super(PhysicsGNN, self).__init__()

        # 1. Встраивание узлов (сейчас все узлы = 1.0, размерность 1 -> hidden_dim)
        self.node_emb = nn.Linear(1, hidden_dim)

        # 2. Графовые свертки (CGConv отлично работает с признаками ребер)
        # edge_dim=4, так как у нас [dx, dy, dz, r]
        self.conv1 = CGConv(hidden_dim, dim=4, batch_norm=True)
        self.conv2 = CGConv(hidden_dim, dim=4, batch_norm=True)
        self.conv3 = CGConv(hidden_dim, dim=4, batch_norm=True)

        # 3. Финальные полносвязные слои для предсказания
        # Вход: hidden_dim (от графа) + 1 (значение плотности) = hidden_dim + 1
        self.fc1 = nn.Linear(hidden_dim + 1, 32)
        self.fc2 = nn.Linear(32, 2) # Выход: [h, p]

    def forward(self, x, edge_index, edge_attr, batch_idx, rho):
        # Шаг 1: Эмбеддинг узлов
        x = self.node_emb(x)
        x = F.relu(x)

        # Шаг 2: Графовые свертки (передача сообщений)
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.relu(self.conv3(x, edge_index, edge_attr))

        # Шаг 3: Глобальное усреднение (превращаем узлы в один вектор графа)
        x_graph = global_mean_pool(x, batch_idx)

        # Шаг 4: Конкатенация плотности! (Убеждаемся, что rho имеет форму [batch_size, 1])
        rho = rho.view(-1, 1)
        x_combined = torch.cat([x_graph, rho], dim=1)

        # Шаг 5: Регрессия
        out = F.relu(self.fc1(x_combined))
        out = self.fc2(out)

        return out

def main():
    print(f"Используем устройство: {DEVICE}")

    # 1. Загрузка данных (Следите за оперативной памятью!)
    print("Загрузка Train датасета (это может занять пару минут и много ОЗУ)...")
    train_dataset = torch.load(os.path.join(DATA_DIR, 'train.pt'), weights_only=False)

    print("Загрузка Val датасета...")
    val_dataset = torch.load(os.path.join(DATA_DIR, 'val.pt'), weights_only=False)

    # 2. Нормализация таргетов (h, p)
    normalizer = Normalizer()
    normalizer.fit(train_dataset)

    # 3. Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Инициализация модели, функции потерь и оптимизатора
    model = PhysicsGNN(hidden_dim=32).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Снижение Learning Rate если сеть перестает учиться
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    print("\nСтарт обучения!")
    best_val_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        # ================= TRAIN =================
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [TRAIN]"):
            batch = batch.to(DEVICE)

            # Нормализуем таргеты
            y_norm = normalizer.transform(batch.y)

            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.rho)

            loss = criterion(out, y_norm)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch.num_graphs

        train_loss /= len(train_loader.dataset)

        # ================= VAL =================
        model.eval()
        val_loss = 0.0
        # Метрики абсолютной ошибки в реальных физических единицах (не нормализованных)
        mae_h = 0.0
        mae_p = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [VAL]"):
                batch = batch.to(DEVICE)
                y_norm = normalizer.transform(batch.y)

                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.rho)
                loss = criterion(out, y_norm)
                val_loss += loss.item() * batch.num_graphs

                # Переводим предсказания обратно в физические единицы для оценки метрик
                out_phys = normalizer.inverse_transform(out)
                y_phys = batch.y

                mae_h += torch.abs(out_phys[:, 0] - y_phys[:, 0]).sum().item()
                mae_p += torch.abs(out_phys[:, 1] - y_phys[:, 1]).sum().item()

        val_loss /= len(val_loader.dataset)
        mae_h /= len(val_loader.dataset)
        mae_p /= len(val_loader.dataset)

        scheduler.step(val_loss)

        print(f"Loss(MSE): Train={train_loss:.4f} | Val={val_loss:.4f} || MAE (физ. ед.): h_err={mae_h:.4f}, p_err={mae_p:.4f}")

        # Сохраняем лучшую модель
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(">>> Модель сохранена!")

if __name__ == "__main__":
    main()
