import os
import glob
import numpy as np
import torch
from torch_geometric.data import Data
from scipy.spatial import cKDTree
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ==================== НАСТРОЙКИ ====================
DUMPS_DIR = "/mnt/Snakes/Data"             # Папка, где лежат ваши 9000 .lammpstrj файлов
OUTPUT_DIR = "./dataset"          # Папка для сохранения готовых .pt файлов
CUTOFF_RADIUS = 2.5               # Радиус отсечки для графа (в сигмах)
PATCHES_PER_FILE = 4              # Сколько случайных патчей вырезать из одной симуляции (аугментация)
RANDOM_SEED = 42
# ===================================================

def parse_filename(filepath):
    """
    Извлекает параметры из имени файла.
    Пример: dump_100_100_0.70000000_1.10000000_1.00000000_8.80000000_0_.lammpstrj
    """
    base = os.path.basename(filepath)
    parts = base.split('_')
    try:
        density = float(f"{parts[3]}")  # 0.7
        h = float(f"{parts[4]}")        # 1.1
        p = float(f"{parts[6]}")       # 8.8
        return density, h, p
    except Exception as e:
        # Игнорируем файлы, которые не подходят под паттерн названия
        return None, None, None

def read_last_frame(filepath):
    """
    Быстро читает размеры ящика и координаты последнего кадра из .lammpstrj файла.
    Использует экономичное построчное чтение памяти и защищен от 'битых' файлов.
    """
    last_frame = []

    try:
        # Читаем файл построчно, не загружая его целиком в ОЗУ
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith("ITEM: TIMESTEP"):
                    last_frame = [line]  # Наткнулись на новый кадр -> сбрасываем буфер
                elif last_frame:
                    last_frame.append(line)

    except (OSError, IOError) as e:
        # Ловим ту самую ошибку Input/output error [Errno 5]
        print(f"\n[ПРЕДУПРЕЖДЕНИЕ] Ошибка чтения диска (файл поврежден): {filepath}")
        return None, None, None
    except UnicodeDecodeError:
        print(f"\n[ПРЕДУПРЕЖДЕНИЕ] Ошибка кодировки (файл поврежден): {filepath}")
        return None, None, None

    if not last_frame:
        return None, None, None

    try:
        # Парсим границы ящика
        xlo, xhi = map(float, last_frame[5].split())
        ylo, yhi = map(float, last_frame[6].split())
        zlo, zhi = map(float, last_frame[7].split())

        box = np.array([xhi - xlo, yhi - ylo, zhi - zlo])
        box_origin = np.array([xlo, ylo, zlo])

        # Парсим координаты атомов
        coords = []
        for line in last_frame[9:]:
            parts = line.split()
            if len(parts) >= 4:
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

        return np.array(coords), box, box_origin

    except Exception as e:
        print(f"\n[ПРЕДУПРЕЖДЕНИЕ] Ошибка структуры внутри файла: {filepath} | Ошибка: {e}")
        return None, None, None

def extract_random_patch(coords, box, box_origin):
    """
    Вырезает случайный квадрант по XY (50% по X и 50% по Y, Z сохраняется целиком).
    Это гарантирует ровно ~2500 частиц при N=10000 для любой плотности.
    """
    Lx, Ly, Lz = box

    # Переносим координаты в систему отсчета [0, L] для удобства резки
    shifted_coords = coords - box_origin

    # Случайный выбор начала окна резки по X и Y
    x_start = np.random.uniform(0, 0.5 * Lx)
    y_start = np.random.uniform(0, 0.5 * Ly)

    x_end = x_start + 0.5 * Lx
    y_end = y_start + 0.5 * Ly

    # Маска для выбора частиц внутри квадранта (Z оставляем весь)
    mask = (shifted_coords[:, 0] >= x_start) & (shifted_coords[:, 0] < x_end) & \
           (shifted_coords[:, 1] >= y_start) & (shifted_coords[:, 1] < y_end)

    patch_coords = shifted_coords[mask]
    return patch_coords

def build_patch_graph(patch_coords, density, h, p, cutoff):
    """
    Строит PyG граф по вырезанным координатам патча.
    Так как это вырезанный кусок, PBC (периодические границы) НЕ применяются.
    """
    num_nodes = patch_coords.shape[0]

    # Строим дерево соседей (без периодических границ для вырезанного патча)
    tree = cKDTree(patch_coords)
    pairs = list(tree.query_pairs(r=cutoff))
    pairs = np.array(pairs)

    if len(pairs) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 4), dtype=torch.float)
    else:
        # Делаем граф неориентированным
        src = np.concatenate([pairs[:, 0], pairs[:, 1]])
        dst = np.concatenate([pairs[:, 1], pairs[:, 0]])

        # Считаем относительные расстояния
        delta = patch_coords[src] - patch_coords[dst]
        distances = np.linalg.norm(delta, axis=1, keepdims=True)

        edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
        # Атрибуты ребер: [dx, dy, dz, r]
        edge_attr = torch.tensor(np.hstack([delta, distances]), dtype=torch.float)

    # Признаки узлов (все единицы)
    x = torch.ones((num_nodes, 1), dtype=torch.float)

    # Таргеты (h, p)
    y = torch.tensor([[h, p]], dtype=torch.float)

    # Создаем объект PyTorch Geometric Data
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    # Добавляем плотность как глобальный параметр (для конкатенации в конце GNN)
    data.rho = torch.tensor([density], dtype=torch.float)

    return data

def process_file_list(files, desc):
    """
    Проходит по списку файлов, вырезает из каждого патчи и собирает список графов.
    """
    graphs_list = []
    for filepath in tqdm(files, desc=desc):
        density, h, p = parse_filename(filepath)
        if density is None:
            continue

        coords, box, box_origin = read_last_frame(filepath)
        if coords is None:
            continue

        # Генерируем несколько случайных патчей из одной симуляции
        for _ in range(PATCHES_PER_FILE):
            patch_coords = extract_random_patch(coords, box, box_origin)
            if len(patch_coords) < 100: # Защита от пустых патчей
                continue
            graph = build_patch_graph(patch_coords, density, h, p, CUTOFF_RADIUS)
            graphs_list.append(graph)

    return graphs_list

def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_files = glob.glob(os.path.join(DUMPS_DIR, "*.lammpstrj"))

    if not all_files:
        print(f"Ошибка: В папке {DUMPS_DIR} не найдено .lammpstrj файлов.")
        return

    print(f"Всего найдено файлов: {len(all_files)}")

    # Сначала парсим плотности всех файлов, чтобы сделать стратифицированный сплит
    valid_files = []
    densities_str = [] # Будем хранить как строки для корректной работы stratify в sklearn

    print("Парсинг параметров файлов...")
    for f in all_files:
        density, _, _ = parse_filename(f)
        if density is not None:
            valid_files.append(f)
            densities_str.append(f"{density:.2f}") # Переводим в строковую категорию (например '0.70')

    print(f"Корректных файлов для работы: {len(valid_files)}")

    # ================= СТРАТИФИЦИРОВАННЫЙ СПЛИТ ИСХОДНЫХ ФАЙЛОВ =================
    # Разбиваем в пропорции: 70% Train, 15% Val, 15% Test
    train_files, temp_files, _, temp_dens = train_test_split(
        valid_files, densities_str,
        test_size=0.3,
        stratify=densities_str,
        random_state=RANDOM_SEED
    )

    val_files, test_files = train_test_split(
        temp_files,
        test_size=0.5,
        stratify=temp_dens,
        random_state=RANDOM_SEED
    )

    print(f"Файлы распределены (без утечки данных):")
    print(f"  - Train симуляций: {len(train_files)}")
    print(f"  - Val симуляций:   {len(val_files)}")
    print(f"  - Test симуляций:  {len(test_files)}")

    # ================= НАРЕЗКА ПАТЧЕЙ И СБОРКА ГРАФОВ =================
    print("\nГенерация патчей...")
    train_graphs = process_file_list(train_files, desc="Нарезка TRAIN")
    val_graphs = process_file_list(val_files, desc="Нарезка VAL")
    test_graphs = process_file_list(test_files, desc="Нарезка TEST")

    print(f"\nИтог по патчам (графам):")
    print(f"  - Train графов: {len(train_graphs)}")
    print(f"  - Val графов:   {len(val_graphs)}")
    print(f"  - Test графов:  {len(test_graphs)}")

    # ================= СОХРАНЕНИЕ =================
    print("\nСохранение датасета в .pt файлы...")
    torch.save(train_graphs, os.path.join(OUTPUT_DIR, 'train.pt'))
    torch.save(val_graphs, os.path.join(OUTPUT_DIR, 'val.pt'))
    torch.save(test_graphs, os.path.join(OUTPUT_DIR, 'test.pt'))

    print(f"Готово! Датасет сохранен в папку: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
