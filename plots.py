import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.loader import DataLoader
from evaluate import PhysicsGNN, Normalizer, DATA_DIR, MODEL_PATH, DEVICE

def generate_advanced_plots():
    # Загружаем данные и модель (как в evaluate.py)
    train_dataset = torch.load(os.path.join(DATA_DIR, 'train.pt'), weights_only=False)
    test_dataset = torch.load(os.path.join(DATA_DIR, 'test.pt'), weights_only=False)

    normalizer = Normalizer()
    normalizer.fit(train_dataset)
    del train_dataset

    model = PhysicsGNN(hidden_dim=32).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    y_true_list, y_pred_list, rho_list = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            out_norm = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.rho)
            out_phys = normalizer.inverse_transform(out_norm)

            y_true_list.append(batch.y.cpu().numpy())
            y_pred_list.append(out_phys.cpu().numpy())
            rho_list.append(batch.rho.cpu().numpy())

    y_true = np.vstack(y_true_list)
    y_pred = np.vstack(y_pred_list)
    rhos = np.concatenate(rho_list).flatten()

    h_true, p_true = y_true[:, 0], y_true[:, 1]
    h_pred, p_pred = y_pred[:, 0], y_pred[:, 1]

    # Усреднение предсказаний (Ensemble averaging) по целым симуляциям
    PATCHES_PER_FILE = 4 # Укажите то число, которое было в prepare_dataset.py

    num_simulations = len(y_true) // PATCHES_PER_FILE

    # Берем истинные значения (они одинаковые для всех 4 патчей одной симуляции, берем первое)
    y_true_sim = y_true[::PATCHES_PER_FILE]
    rhos = rhos[::PATCHES_PER_FILE]

    # Усредняем предсказания сети по 4 патчам для каждой симуляции
    # Меняем форму массива: [N_patches, 2] -> [N_simulations, 4, 2] и берем среднее по оси 1
    y_pred_sim = y_pred.reshape(num_simulations, PATCHES_PER_FILE, 2).mean(axis=1)

    h_true, p_true = y_true_sim[:, 0], y_true_sim[:, 1]
    h_pred, p_pred = y_pred_sim[:, 0], y_pred_sim[:, 1]

    # Абсолютные ошибки
    error_h = np.abs(h_true - h_pred)
    error_p = np.abs(p_true - p_pred)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(16, 6))

    # ================= ГРАФИК 1: ТЕПЛОВАЯ КАРТА ОШИБОК В ПРОСТРАНСТВЕ (h, p) =================
    ax1 = plt.subplot(1, 2, 1)
    # Используем scatter, где цвет (c) - это размер ошибки P
    sc = ax1.scatter(h_true, p_true, c=error_p, cmap='inferno_r', s=30, alpha=0.8, edgecolor='none')
    plt.colorbar(sc, ax=ax1, label='Абсолютная ошибка $\\Delta p$')
    ax1.set_title('Пространственное распределение ошибки (Фазовая карта)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Высота конфайнмента ($h$)', fontsize=12)
    ax1.set_ylabel('Дипольный момент ($p$)', fontsize=12)
    # ax1.text(0.05, 0.95, 'Чем светлее, тем выше ошибка.\nВидно, в каких фазах сеть "сомневается"',
    #          transform=ax1.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

    # ================= ГРАФИК 2: ЗАВИСИМОСТЬ ОШИБКИ ОТ ПЛОТНОСТИ (Violin / Boxplot) =================
    ax2 = plt.subplot(1, 2, 2)

    # Заворачиваем данные в DataFrame. Это железобетонно заставит Seaborn
    # сгруппировать данные по уникальным значениям плотности.
    df = pd.DataFrame({
        'Density': np.round(rhos, 2),
        'Error_p': error_p
    })
    df = df.sort_values(by='Density')
    df['Density'] = df['Density'].astype(str)

    # Строим график, передавая DataFrame
    sns.boxplot(data=df, x='Density', y='Error_p', ax=ax2, palette='Blues')

    ax2.set_title('Зависимость ошибки $\\Delta p$ от плотности системы $\\rho$', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Плотность ($\\rho$)', fontsize=12)
    ax2.set_ylabel('Абсолютная ошибка дипольного момента', fontsize=12)

    plt.tight_layout()
    plt.savefig('advanced_results.png', dpi=300)
    print("Дополнительные графики сохранены как 'advanced_results.png'")

if __name__ == "__main__":
    generate_advanced_plots()
