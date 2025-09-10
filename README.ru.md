# Глава 153: DeepONet для финансов

## Обзор

Deep Operator Networks (DeepONet) представляют собой смену парадигмы в том, как нейронные сети обучаются отображениям в функциональных пространствах. В отличие от традиционных нейронных сетей, которые обучают отображения между конечномерными векторами, DeepONet обучает **операторы** -- отображения из одного функционального пространства в другое. В финансах это означает обучение целым семействам ценовых функций, кривых доходности и отображений рисков одновременно, а не подбор отдельных точечных оценок.

Предложенная Lu et al. (2021), DeepONet основана на **универсальной теореме аппроксимации для операторов**, которая гарантирует, что сеть с branch-сетью и trunk-сетью может аппроксимировать любой непрерывный нелинейный оператор с произвольной точностью.

## Почему DeepONet для финансов?

### Парадигма обучения операторов

Традиционные нейронные сети в финансах решают задачи вида:
- По состоянию рынка **x** предсказать цену **y** (аппроксимация функции)
- По временному ряду **X** предсказать следующее значение **y** (моделирование последовательностей)

DeepONet решает принципиально другую задачу:
- По **входной функции** (например, поверхности волатильности) обучить **оператор**, который отображает её в **выходную функцию** (например, цены опционов по всем страйкам и срокам)

```
Традиционная НС:    x ∈ R^n  →  y ∈ R^m        (вектор в вектор)
DeepONet:           u(·)     →  G(u)(y)          (функция в функцию)

где:
  u(·)  = входная функция (напр., поверхность подразумеваемой волатильности)
  y     = точка запроса (напр., страйк K, срок T)
  G(u)  = выходной оператор (напр., цена опциона при (K,T))
```

### Ключевые преимущества

| Характеристика | Стандартная НС | DeepONet |
|---------------|----------------|----------|
| Тип входа | Векторы фиксированного размера | Функции (переменная дискретизация) |
| Тип выхода | Векторы фиксированного размера | Функции, вычисляемые в любой точке |
| Обобщение | Интерполяция в пространстве данных | Обобщение в функциональном пространстве |
| Обучение | Одна модель на сценарий | Одна модель для всех сценариев |
| Перенос | Ограниченный | Естественный перенос между активами |
| Физические ограничения | Сложно включить | PI-DeepONet добавляет невязку PDE |

## Архитектура DeepONet

### Базовая структура

DeepONet состоит из двух подсетей:

```
                    Входная функция u(x)
                    измеренная в {x_1, ..., x_m}
                           │
                    ┌──────▼──────┐
                    │  Branch Net  │     Кодирует входную функцию
                    │  (MLP/CNN/RNN)│     в латентное представление
                    └──────┬──────┘
                           │
                    [b_1, b_2, ..., b_p]   Выход branch (p нейронов)
                           │
                           ●─── скалярное произв. ───●
                           │                          │
                    [t_1, t_2, ..., t_p]   Выход trunk (p нейронов)
                           │
                    ┌──────▲──────┐
                    │  Trunk Net   │     Кодирует точку запроса
                    │    (MLP)     │     (где вычислить выход)
                    └──────┬──────┘
                           │
                    Точка запроса y
                    (напр., страйк K, срок T)

    Выход: G(u)(y) = Σ_{k=1}^{p} b_k · t_k + bias
```

### Математическая формулировка

Аппроксимация DeepONet:

```
G(u)(y) ≈ Σ_{k=1}^{p} br_k(u(x_1), u(x_2), ..., u(x_m)) · tr_k(y) + b_0
```

где:
- `br_k` -- k-й выход branch-сети
- `tr_k` -- k-й выход trunk-сети
- `p` -- размерность латентного пространства (число базисных функций)
- `b_0` -- обучаемое смещение

### Варианты Branch-сети

Branch-сеть кодирует входную функцию. Различные архитектуры подходят для разных типов входных данных:

#### MLP Branch (для табулированных функций)

```python
class MLPBranch(nn.Module):
    """Branch-сеть на основе многослойного перцептрона.

    Лучше всего для: дискретизированных значений функции
    в фиксированных сенсорных точках.
    Пример: Поверхность волатильности на фиксированной сетке (K, T).
    """
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.GELU(),
                nn.LayerNorm(h_dim),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # [batch, p]
```

#### CNN Branch (для данных на сетке)

```python
class CNNBranch(nn.Module):
    """Branch-сеть на основе 1D-CNN.

    Лучше всего для: временных рядов с локальными паттернами.
    Пример: Исторические ценовые ряды, снимки стакана заявок.
    """
    def __init__(self, input_channels, seq_len, output_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        h = self.conv(x).squeeze(-1)
        return self.fc(h)  # [batch, p]
```

#### RNN Branch (для последовательных входов)

```python
class RNNBranch(nn.Module):
    """Branch-сеть на основе LSTM/GRU.

    Лучше всего для: временных рядов переменной длины.
    Пример: Тиковые торговые данные с нерегулярными метками времени.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])  # [batch, p]
```

### Trunk-сеть

Trunk-сеть кодирует точки запроса, в которых вычисляется выходная функция:

```python
class TrunkNet(nn.Module):
    """Trunk-сеть для кодирования точек запроса.

    Для ценообразования опционов: y = (S, t, K, T, r)
    Для кривых доходности: y = (срок погашения,)
    Для отображения рисков: y = (id_актива, горизонт)
    """
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.GELU(),
                nn.LayerNorm(h_dim),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, y):
        return self.net(y)  # [batch, p]
```

## Универсальная теорема аппроксимации для операторов

### Формулировка теоремы

**Теорема (Chen & Chen, 1995; Lu et al., 2021):** Пусть G -- непрерывный оператор, отображающий банахово пространство V в банахово пространство U. Тогда для любого компактного множества K в V и любого epsilon > 0 существует DeepONet с branch-сетью `br` и trunk-сетью `tr` такой, что:

```
|G(u)(y) - Σ_{k=1}^{p} br_k(u(x_1), ..., u(x_m)) · tr_k(y)| < ε
```

для всех u из K и y из области определения.

### Значение для финансов

1. **Ценообразование опционов**: Одна DeepONet может обучить оператор Блэка-Шоулза, оператор Хестона или любой оператор ценообразования с произвольной точностью
2. **Кривые доходности**: Одна модель отображает экономические условия во всю кривую доходности
3. **Поверхности рисков**: Одна модель отображает состав портфеля в риски по всем горизонтам

## Финансовые приложения

### Приложение 1: Оператор ценообразования опционов

Обучение отображения из поверхностей волатильности в поверхности цен опционов:

```
Входная функция:   σ(K, T)          -- поверхность подразумеваемой волатильности
Точка запроса:     y = (S, K, T, r)  -- спот, страйк, срок, ставка
Выход:             C(S, K, T)        -- цена опциона

G: σ(·,·) → C(S, ·, ·)
```

### Приложение 2: Оператор кривой доходности

Обучение отображения из макроэкономических индикаторов в кривые доходности:

```
Входная функция:   macro(t)          -- экономические индикаторы во времени
Точка запроса:     y = (срок,)       -- срок погашения облигации
Выход:             r(срок)           -- доходность при данном сроке

G: macro(·) → r(·)
```

### Приложение 3: Отображение рисков портфеля

Обучение отображения из весов портфеля в меры риска по горизонтам:

```
Входная функция:   w(актив)          -- весовая функция портфеля
Точка запроса:     y = (горизонт, α) -- горизонт риска и уровень доверия
Выход:             VaR(горизонт, α)  -- Value-at-Risk

G: w(·) → VaR(·, ·)
```

### Приложение 4: Криптотрейдинг с данными Bybit

Применение DeepONet для обучения операторов ценовой динамики на данных биржи Bybit:

```python
import ccxt

def fetch_bybit_data(symbol='BTC/USDT', timeframe='1h', limit=1000):
    """Получение OHLCV данных с биржи Bybit."""
    exchange = ccxt.bybit({
        'enableRateLimit': True,
        'options': {'defaultType': 'linear'}
    })
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df
```

## Physics-Informed DeepONet (PI-DeepONet)

### Мотивация

Финансовые модели описываются дифференциальными уравнениями в частных производных (PDE). PI-DeepONet включает эти физические ограничения непосредственно в функцию потерь, значительно улучшая точность и физическую согласованность.

### Ограничение PDE Блэка-Шоулза

PDE Блэка-Шоулза для европейских опционов:

```
∂C/∂t + (1/2)σ^2 S^2 ∂^2C/∂S^2 + rS ∂C/∂S - rC = 0
```

```python
class PIDeepONet(nn.Module):
    """Physics-Informed DeepONet с невязкой PDE."""

    def __init__(self, branch_net, trunk_net):
        super().__init__()
        self.branch = branch_net
        self.trunk = trunk_net
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u_sensors, y_query):
        b = self.branch(u_sensors)
        t = self.trunk(y_query)
        return torch.sum(b * t, dim=-1, keepdim=True) + self.bias

    def pde_residual(self, u_sensors, S, t, sigma, r):
        """Вычисление невязки PDE Блэка-Шоулза."""
        S.requires_grad_(True)
        t.requires_grad_(True)

        y_query = torch.cat([S, t], dim=-1)
        C = self.forward(u_sensors, y_query)

        dC = torch.autograd.grad(C, [S, t],
                                  grad_outputs=torch.ones_like(C),
                                  create_graph=True)
        dC_dS, dC_dt = dC[0], dC[1]

        d2C_dS2 = torch.autograd.grad(dC_dS, S,
                                       grad_outputs=torch.ones_like(dC_dS),
                                       create_graph=True)[0]

        residual = dC_dt + 0.5 * sigma**2 * S**2 * d2C_dS2 + r * S * dC_dS - r * C
        return residual
```

## Многоточностный DeepONet (Multi-Fidelity)

### Объединение моделей низкой и высокой точности

На практике у нас есть:
- **Данные низкой точности**: Дешёвые для генерации (Блэк-Шоулз, биномиальные деревья)
- **Данные высокой точности**: Дорогие для генерации (Monte Carlo по Хестону)

Многоточностный DeepONet обучает оператор коррекции:

```
G_HF(u)(y) = G_LF(u)(y) + G_correction(u)(y)
```

```python
class MultiFidelityDeepONet(nn.Module):
    """Многоточностный DeepONet, объединяющий модели BS и Хестона."""

    def __init__(self, branch_dim, trunk_dim, latent_dim):
        super().__init__()
        # DeepONet низкой точности (предобучен на данных BS)
        self.lf_branch = MLPBranch(branch_dim, [256, 256], latent_dim)
        self.lf_trunk = TrunkNet(trunk_dim, [128, 128], latent_dim)
        self.lf_bias = nn.Parameter(torch.zeros(1))

        # DeepONet коррекции (обучен на остатках)
        self.corr_branch = MLPBranch(branch_dim, [128, 128], latent_dim)
        self.corr_trunk = TrunkNet(trunk_dim, [64, 64], latent_dim)
        self.corr_bias = nn.Parameter(torch.zeros(1))

        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, u_sensors, y_query):
        lf_pred = self.forward_lf(u_sensors, y_query)
        correction = self.forward_correction(u_sensors, y_query)
        return self.alpha * lf_pred + correction
```

## Перенос между активами и рыночными режимами

### Межактивный перенос обучения

DeepONet естественно поддерживает перенос обучения, поскольку операторы кодируют структурные зависимости:

1. Сохранить trunk-сеть замороженной (геометрия выходного пространства общая)
2. Дообучить branch-сеть (входная функция отличается)
3. Опционально добавить адаптерные слои

### Режимно-осведомлённый DeepONet

Использует классификатор режимов для выбора/смешивания branch-сетей, обученных на различных рыночных режимах (бычий, медвежий, боковой, высокая волатильность).

## Сравнение с FNO и стандартными НС

| Аспект | DeepONet | FNO |
|--------|----------|-----|
| Архитектура | Branch + Trunk | Фурье-слои |
| Дискретизация входа | Фиксированные сенсоры | Требуется регулярная сетка |
| Вычисление выхода | Запрос в любой точке | Полный выход на сетке |
| Спектральное смещение | Отсутствует | Предпочитает низкие частоты |
| Интеграция PDE | PI-DeepONet | PINO |
| Нерегулярные данные | Естественная поддержка | Требуется интерполяция |
| Масштабирование памяти | O(mp + pd) | O(N log N) для FFT |
| Лучше для | Точечные запросы, нерегулярные данные | Периодические задачи, полные поля |

## Структура проекта

```
153_deeponet_finance/
├── README.md                        # Англоязычная документация
├── README.ru.md                     # Этот файл
├── readme.simple.md                 # Упрощённое объяснение (английский)
├── readme.simple.ru.md              # Упрощённое объяснение (русский)
├── python/
│   ├── __init__.py                  # Инициализация пакета
│   ├── model.py                     # Архитектуры модели DeepONet
│   ├── train.py                     # Конвейер обучения
│   ├── data_loader.py               # Загрузка данных (акции + Bybit крипто)
│   ├── visualize.py                 # Утилиты визуализации
│   ├── backtest.py                  # Фреймворк бэктестинга
│   └── requirements.txt             # Зависимости Python
└── rust_deeponet/
    ├── Cargo.toml                   # Конфигурация проекта Rust
    ├── src/
    │   ├── lib.rs                   # Основная библиотека
    │   └── bin/
    │       ├── train.rs             # Бинарный файл обучения
    │       ├── predict.rs           # Бинарный файл предсказания
    │       └── fetch_data.rs        # Бинарный файл загрузки данных
    └── examples/
        ├── option_pricing.rs        # Пример ценообразования опционов
        ├── crypto_forecast.rs       # Пример прогнозирования крипто
        └── yield_curve.rs           # Пример кривой доходности
```

## Запуск кода

### Python

```bash
cd python
pip install -r requirements.txt

# Обучение DeepONet для ценообразования опционов
python train.py --mode option_pricing --epochs 500

# Обучение DeepONet для прогнозирования крипто (Bybit)
python train.py --mode crypto --symbol BTCUSDT --epochs 200

# Бэктест торговой стратегии
python backtest.py --model checkpoints/best_deeponet.pth --symbol BTCUSDT

# Визуализация результатов
python visualize.py --results results/backtest_results.json
```

### Rust

```bash
cd rust_deeponet

# Загрузка рыночных данных с Bybit
cargo run --bin fetch_data -- --symbol BTCUSDT --interval 60 --limit 5000

# Обучение модели DeepONet
cargo run --bin train -- --config config.json

# Запуск предсказаний
cargo run --bin predict -- --model model.bin --symbol BTCUSDT

# Запуск примеров
cargo run --example option_pricing
cargo run --example crypto_forecast
cargo run --example yield_curve
```

## Ссылки

1. Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E. (2021). Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. *Nature Machine Intelligence*, 3(3), 218-229.

2. Chen, T., & Chen, H. (1995). Universal approximation to nonlinear operators by neural networks with arbitrary activation functions and its application to dynamical systems. *IEEE Transactions on Neural Networks*, 6(4), 911-917.

3. Wang, S., Wang, H., & Perdikaris, P. (2021). Learning the solution operator of parametric partial differential equations with physics-informed DeepONets. *Science Advances*, 7(40).

4. Howard, A. A., Perego, M., Karniadakis, G. E., & Stinis, P. (2022). Multifidelity deep operator networks. *arXiv preprint arXiv:2204.09157*.

5. Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhatt, K., Stuart, A., & Anandkumar, A. (2021). Fourier neural operator for parametric partial differential equations. *ICLR 2021*.

---

*Глава 153 из Machine Learning for Trading. DeepONet позволяет обучать операторы для финансовых приложений, отображая целые функциональные пространства в функциональные пространства -- принципиально более мощная парадигма, чем традиционные точечные отображения нейронных сетей.*
