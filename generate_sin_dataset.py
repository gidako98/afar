import numpy as np
import pandas as pd

def generate_fault_dataset(n_samples=500, seed=42):
    np.random.seed(seed)
    N = 16  # количество элементов решётки
    
    # Нормальные параметры
    AMP_NORM = 52.96
    AMP_FAULT = 35.32  # падение при отказе (~33%)
    PHASE_NORM = -np.pi  # -180° → sin=-1.0, cos≈0.0
    PHASE_FAULT = 0.0    # 0° → sin≈0.0, cos=1.0
    
    rows = []
    
    for _ in range(n_samples):
        # 1. Генерируем случайную маску отказов (1–4 отказавших элемента)
        n_faults = np.random.choice([0, 1, 2, 3, 4], p=[0.4, 0.3, 0.15, 0.1, 0.05])
        fault_mask = np.zeros(N, dtype=int)
        
        if n_faults > 0:
            # Отказы чаще каскадные (соседние элементы)
            start = np.random.randint(0, N - n_faults + 1)
            fault_mask[start:start + n_faults] = 1
        
        # 2. Генерируем признаки
        amp = np.full(N, AMP_NORM)
        phase = np.full(N, PHASE_NORM)
        
        # Применяем отказы
        fault_idx = np.where(fault_mask == 1)[0]
        if len(fault_idx) > 0:
            amp[fault_idx] = AMP_FAULT + np.random.normal(0, 0.5, len(fault_idx))  # шум
            phase[fault_idx] = PHASE_FAULT + np.random.normal(0, 0.1, len(fault_idx))
        
        # 3. Добавляем шум измерений для всех элементов
        amp += np.random.normal(0, 0.3, N)
        phase += np.random.normal(0, 0.05, N)
        
        # 4. Преобразуем фазу в sin/cos
        sinp = np.sin(phase)
        cosp = np.cos(phase)
        
        # 5. Собираем строку: 48 признаков + 16 меток
        row = np.concatenate([
            amp,    # 16
            sinp,   # 16
            cosp,   # 16
            fault_mask  # 16 меток
        ])
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv("dataset_correct.csv", header=False, index=False)
    print(f"✅ Сгенерирован корректный датасет: {n_samples} строк")
    print(f"   Пример отказа (строка 100): элементы {[i+1 for i, m in enumerate(rows[100][48:]) if m == 1]}")
    return df

# Генерация датасета
df = generate_fault_dataset(500)