# Yandex Maps Geocoder

Полнофункциональное Python-приложение для геокодирования с использованием Yandex Maps API с поддержкой валидации, кэширования, многопоточной обработки и экспорта результатов.

## Содержание

- [Обзор](#обзор)
- [Возможности](#возможности)
- [Установка](#установка)
- [Конфигурация](#конфигурация)
- [Быстрый старт](#быстрый-старт)
- [Использование](#использование)
  - [Интерактивный режим](#интерактивный-режим)
  - [Командная строка](#командная-строка)
  - [Программное использование](#программное-использование)
- [Архитектура](#архитектура)
- [API документация](#api-документация)
  - [Класс YandexGeocoder](#класс-yandexgeocoder)
  - [Методы геокодирования](#методы-геокодирования)
  - [Обработка ошибок](#обработка-ошибок)
- [Форматы данных](#форматы-данных)
- [Производительность](#производительность)
- [Тестирование](#тестирование)
- [Развертывание](#развертывание)
- [Вопросы и ответы](#вопросы-и-ответы)
- [Лицензия](#лицензия)

## Обзор

Yandex Maps Geocoder — это мощная Python-библиотека для работы с геокодированием через Yandex Maps API. Приложение предоставляет:

- **Прямое геокодирование**: преобразование адреса в координаты
- **Обратное геокодирование**: преобразование координат в адрес
- **Пакетная обработка**: одновременная обработка тысяч адресов
- **Валидация адресов**: проверка корректности введенных адресов
- **Кэширование**: снижение количества запросов к API
- **Экспорт результатов**: сохранение в Excel, CSV, JSON
- **Статистика**: мониторинг производительности и качества

## Возможности

### ✅ Основные функции
- Поддержка прямого и обратного геокодирования
- Валидация российских адресов с нормализацией
- Многопоточная пакетная обработка
- Интеллектуальное кэширование запросов
- Подробное логирование и статистика

### ✅ Обработка ошибок и надежность
- Автоматические повторные попытки при сбоях
- Ограничение частоты запросов (rate limiting)
- Таймауты и обработка сетевых ошибок
- Валидация входных параметров
- Резервное сохранение данных

### ✅ Экспорт и интеграция
- Экспорт в Excel с автоформатированием
- Экспорт в CSV и JSON
- Поддержка прокси-серверов
- CLI интерфейс для автоматизации
- REST API совместимость

### ✅ Производительность
- Оптимизированная обработка компонентов адреса
- Настраиваемые лимиты запросов
- Балансировка нагрузки при пакетной обработке
- Минимальные задержки между запросами

## Установка

### Требования
- Python 3.8+
- API ключ Yandex Maps

### Установка через pip

```bash
# Клонирование репозитория
git clone https://github.com/Skaylic/yandex-geocoder.git
cd yandex-geocoder

# Создание виртуального окружения
python -m venv venv

# Активация виртуального окружения
# На Windows:
venv\Scripts\activate
# На Linux/Mac:
source venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt
```

### Зависимости

Основные зависимости:
- `requests>=2.28.0` - HTTP-запросы
- `pandas>=1.5.0` - обработка данных
- `openpyxl>=3.0.0` - экспорт в Excel
- `python-dotenv>=0.21.0` - управление переменными окружения

Опциональные зависимости:
- `ratelimit>=2.2.1` - ограничение частоты запросов
- `pytest>=7.0.0` - тестирование

## Конфигурация

### Получение API ключа

1. Зарегистрируйтесь на [Yandex Cloud](https://cloud.yandex.ru/)
2. Создайте новый каталог
3. Перейдите в раздел "API-ключи"
4. Создайте новый ключ
5. Скопируйте ключ

### Настройка переменных окружения

Создайте файл `.env` в корне проекта:

```env
# Обязательные параметры
YANDEX_API_KEY=ваш_api_ключ_здесь

# Опциональные параметры
HTTP_PROXY=http://proxy.example.com:8080
HTTPS_PROXY=http://proxy.example.com:8080
LOG_LEVEL=INFO
LOG_FILE=geocoder.log
MAX_WORKERS=4
RATE_LIMIT=0.5
TIMEOUT=30
RETRY_COUNT=3
```

## Быстрый старт

### Простой пример

```python
from geocoder_app import YandexGeocoder

# Инициализация геокодера
geocoder = YandexGeocoder()

# Геокодирование адреса
results = geocoder.geocode("Москва, Красная площадь, 1", max_results=1)

if results:
    result = results[0]
    print(f"Адрес: {result.full_address}")
    print(f"Координаты: {result.latitude}, {result.longitude}")
    print(f"Почтовый индекс: {result.postal_code}")
```

### Обратное геокодирование

```python
# Обратное геокодирование
result = geocoder.reverse_geocode(55.755277, 37.617680)

if result:
    print(f"Адрес: {result.full_address}")
```

### Пакетная обработка

```python
# Обработка списка адресов
addresses = [
    "Москва, Красная площадь, 1",
    "Санкт-Петербург, Невский проспект, 28",
    "Екатеринбург, площадь 1905 года"
]

results = geocoder.batch_geocode(addresses, max_workers=2)
```

## Использование

### Интерактивный режим

Запустите приложение в интерактивном режиме:

```bash
python geocoder_app.py
```

Вы увидите меню:
```
=== Яндекс Геокодер ===
1. Адрес → Координаты
2. Координаты → Адрес
3. Пакетная обработка файла
4. Экспорт в Excel
5. Статистика
6. Выход
```

### Командная строка

Используйте CLI утилиту для автоматизации:

```bash
# Геокодирование одного адреса
python geocoder_cli.py --address "Москва, Красная площадь"

# Обратное геокодирование
python geocoder_cli.py --reverse 55.755277 37.617680

# Пакетная обработка CSV файла
python geocoder_cli.py --input addresses.csv --output results.xlsx

# Проверка адреса
python geocoder_cli.py --validate "ул. Ленина"

# Показать статистику
python geocoder_cli.py --stats --export stats.json

# Очистить кэш
python geocoder_cli.py --clear-cache
```

#### Опции командной строки

| Параметр | Описание | Пример |
|----------|----------|---------|
| `--address` | Адрес для геокодирования | `--address "Москва"` |
| `--reverse` | Координаты для обратного геокодирования | `--reverse 55.75 37.61` |
| `--input` | Входной файл с адресами | `--input addresses.csv` |
| `--output` | Выходной файл | `--output results.xlsx` |
| `--format` | Формат вывода (excel/csv/json) | `--format csv` |
| `--workers` | Количество потоков | `--workers 4` |
| `--rate-limit` | Лимит запросов в секунду | `--rate-limit 0.5` |
| `--timeout` | Таймаут запроса | `--timeout 30` |
| `--proxy` | URL прокси-сервера | `--proxy http://proxy:8080` |
| `--clear-cache` | Очистить кэш | `--clear-cache` |
| `--verbose` | Подробный вывод | `--verbose` |

### Программное использование

#### Инициализация с настройками

```python
from geocoder_app import YandexGeocoder, RetryStrategy

# Расширенная инициализация
geocoder = YandexGeocoder(
    api_key="your_api_key",
    timeout=30,
    rate_limit=0.5,  # 30 запросов в минуту
    retry_strategy=RetryStrategy(
        max_retries=3,
        base_delay=1.0,
        backoff_factor=2.0
    ),
    proxy="http://proxy.example.com:8080",
    use_cache=True
)
```

#### Валидация адресов

```python
from geocoder_app import RussianAddressValidator

validator = RussianAddressValidator()
validation = validator.validate("Москва, ул. Ленина, д. 10")

print(f"Корректность: {validation.is_valid}")
print(f"Уверенность: {validation.confidence:.2f}")
print(f"Проблемы: {validation.issues}")
print(f"Нормализованный: {validation.normalized_address}")
```

#### Экспорт результатов

```python
from geocoder_app import export_to_excel

# Экспорт в Excel с полной информацией
df = export_to_excel(
    geocoder,
    addresses,
    output_file="results.xlsx",
    include_components=True,
    max_results_per_address=1
)

# Экспорт статистики
stats = geocoder.export_statistics("statistics.json")
```

## Архитектура

### Компоненты системы

```
┌─────────────────────────────────────────────────────────────┐
│                       Основное приложение                    │
├─────────────────────────────────────────────────────────────┤
│  YandexGeocoder ──┬── RateLimiter ──┬── RetryStrategy       │
│                   │                  │                       │
│                   ├── Validator ─────┼── ComponentProcessor  │
│                   │                  │                       │
│                   └── Session ───────┴── CacheManager        │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      Yandex Maps API                        │
│                    (geocode-maps.yandex.ru)                 │
└─────────────────────────────────────────────────────────────┘
```

### Поток данных

1. **Валидация** → Проверка и нормализация входного адреса
2. **Кэширование** → Поиск в кэше, если доступен
3. **Rate limiting** → Контроль частоты запросов
4. **HTTP запрос** → Отправка запроса к API
5. **Retry логика** → Повтор при ошибках
6. **Обработка ответа** → Парсинг и структурирование
7. **Кэширование** → Сохранение результата
8. **Экспорт** → Форматирование и сохранение

### Классовая структура

```python
# Основные классы
YandexGeocoder           # Основной класс геокодера
RussianAddressValidator  # Валидатор российских адресов
RateLimiter              # Ограничитель частоты запросов
RetryStrategy            # Стратегия повторных попыток
ComponentProcessor       # Процессор компонентов адреса
GeocodingResult          # Результат геокодирования
AddressValidationResult  # Результат валидации
```

## API документация

### Класс YandexGeocoder

#### Конструктор

```python
def __init__(self, 
             api_key: Optional[str] = None,
             timeout: int = 30,
             retry_strategy: Optional[RetryStrategy] = None,
             rate_limit: float = 0.5,
             proxy: Optional[str] = None,
             use_cache: bool = True):
```

**Параметры:**
- `api_key` (str): API ключ Yandex Maps
- `timeout` (int): Таймаут запросов в секундах (по умолчанию 30)
- `retry_strategy` (RetryStrategy): Стратегия повторных попыток
- `rate_limit` (float): Лимит запросов в секунду (по умолчанию 0.5)
- `proxy` (str): URL прокси-сервера
- `use_cache` (bool): Использовать кэширование (по умолчанию True)

#### Основные методы

### Метод `geocode()`

```python
def geocode(self, 
            address: str, 
            max_results: int = 5,
            lang: str = 'ru_RU') -> List[GeocodingResult]:
```

Прямое геокодирование: преобразует адрес в координаты.

**Параметры:**
- `address` (str): Адрес для поиска
- `max_results` (int): Максимальное количество результатов (1-100)
- `lang` (str): Язык ответа ('ru_RU', 'en_US', и т.д.)

**Возвращает:**
- `List[GeocodingResult]`: Список объектов геокодирования

**Пример:**
```python
results = geocoder.geocode("Москва, Красная площадь", max_results=3)
for result in results:
    print(f"{result.full_address}: {result.latitude}, {result.longitude}")
```

### Метод `reverse_geocode()`

```python
def reverse_geocode(self, 
                    lat: float, 
                    lon: float,
                    lang: str = 'ru_RU') -> Optional[GeocodingResult]:
```

Обратное геокодирование: преобразует координаты в адрес.

**Параметры:**
- `lat` (float): Широта (-90 до 90)
- `lon` (float): Долгота (-180 до 180)
- `lang` (str): Язык ответа

**Возвращает:**
- `GeocodingResult` или `None`: Результат геокодирования

**Пример:**
```python
result = geocoder.reverse_geocode(55.755277, 37.617680)
if result:
    print(f"Адрес: {result.full_address}")
```

### Метод `batch_geocode()`

```python
def batch_geocode(self, 
                  addresses: List[str],
                  max_workers: int = 4,
                  batch_size: int = 10,
                  delay_between_batches: float = 1.0) -> List[List[GeocodingResult]]:
```

Пакетное геокодирование с многопоточностью.

**Параметры:**
- `addresses` (List[str]): Список адресов для обработки
- `max_workers` (int): Максимальное количество потоков
- `batch_size` (int): Размер пачки для одного потока
- `delay_between_batches` (float): Задержка между пачками в секундах

**Возвращает:**
- `List[List[GeocodingResult]]`: Список списков результатов

**Пример:**
```python
results = geocoder.batch_geocode(
    addresses,
    max_workers=4,
    batch_size=50,
    delay_between_batches=0.5
)
```

### Метод `get_stats()`

```python
def get_stats(self) -> Dict[str, Any]:
```

Получить статистику работы геокодера.

**Возвращает:**
- `Dict[str, Any]`: Статистика с метриками производительности

**Пример:**
```python
stats = geocoder.get_stats()
print(f"Успешных запросов: {stats['successful_requests']}")
print(f"Эффективность кэша: {stats['cache_hit_rate']:.1f}%")
```

### Метод `clear_cache()`

```python
def clear_cache(self):
```

Очистить кэш запросов.

**Пример:**
```python
geocoder.clear_cache()
```

### Методы геокодирования

#### Сырое геокодирование

```python
def geocode_raw(self, 
                address: str, 
                format: str = 'json', 
                results: int = 10,
                lang: str = 'ru_RU') -> Dict[str, Any]:
```

Возвращает сырой ответ API без обработки.

#### Получение метаданных

```python
def get_metadata(self, address: str) -> Dict[str, Any]:
```

Получить метаданные запроса (found, request, suggest, fix).

### Обработка ошибок

Приложение использует многоуровневую обработку ошибок:

```python
try:
    results = geocoder.geocode(address)
except ConnectionError as e:
    print(f"Ошибка соединения: {e}")
except ValueError as e:
    print(f"Некорректный запрос: {e}")
except Exception as e:
    print(f"Неизвестная ошибка: {e}")
```

#### Типы исключений

- `ConnectionError`: Ошибки сети, таймауты, недоступность API
- `ValueError`: Некорректные параметры, неверный формат ответа
- `KeyError`: Отсутствие ожидаемых полей в ответе API

## Форматы данных

### Входные форматы

#### Текстовый файл (TXT)
```
Москва, Красная площадь, 1
Санкт-Петербург, Невский проспект, 28
Екатеринбург, площадь 1905 года
```

#### CSV файл
```csv
id,address,comment
1,"Москва, Красная площадь, 1",основной
2,"Санкт-Петербург, Невский проспект, 28",филиал
```

#### Excel файл
Поддерживаются форматы .xlsx и .xls

### Выходные форматы

#### Excel (.xlsx)
Содержит несколько листов:
- **Геокодирование**: Основные результаты
- **Статистика**: Метрики выполнения
- **Детальная статистика**: Расширенные метрики

#### CSV (.csv)
```csv
Запрос,Статус,Найденный адрес,Широта,Долгота,Точность,Тип объекта
"Москва, Красная площадь",Успешно,"Россия, Москва, Красная площадь",55.755277,37.617680,exact,house
```

#### JSON (.json)
```json
[
  {
    "request_id": "abc123",
    "query": "Москва, Красная площадь",
    "full_address": "Россия, Москва, Красная площадь",
    "latitude": 55.755277,
    "longitude": 37.617680,
    "precision": "exact",
    "kind": "house",
    "country_code": "RU",
    "postal_code": "101000",
    "confidence": 0.95,
    "timestamp": "2024-01-15T12:00:00"
  }
]
```

### Структура GeocodingResult

```python
@dataclass
class GeocodingResult:
    request_id: str          # Уникальный ID запроса
    query: str               # Исходный запрос
    full_address: str        # Полный адрес
    latitude: float          # Широта
    longitude: float         # Долгота
    precision: str           # Точность (exact, number, range, etc.)
    kind: str                # Тип объекта (house, street, etc.)
    country_code: str        # Код страны (RU, US, etc.)
    postal_code: str         # Почтовый индекс
    components: Dict[str, str]  # Компоненты адреса
    confidence: float        # Уверенность (0.0-1.0)
    processing_time: float   # Время обработки (сек)
    timestamp: datetime      # Время создания
    cache_hit: bool          # Попадание в кэш
    retry_count: int = 0     # Количество повторных попыток
```

## Производительность

### Оптимизация запросов

#### Кэширование
- LRU кэш на 1024 запроса
- Хеширование адресов для быстрого поиска
- Автоматическая очистка устаревших записей

#### Rate limiting
- Настраиваемый лимит запросов в секунду
- Умная задержка между запросами
- Предотвращение блокировки API

#### Пакетная обработка
- Многопоточное выполнение
- Балансировка нагрузки
- Контроль задержек между пачками

### Рекомендации по производительности

1. **Размер пачки**: 50-100 адресов на поток
2. **Количество потоков**: 2-4 для стандартного API ключа
3. **Rate limit**: 0.5 запроса/сек (30 запросов/минуту)
4. **Таймаут**: 30 секунд для стабильного соединения

### Мониторинг

```python
# Получение статистики
stats = geocoder.get_stats()

print(f"Всего запросов: {stats['total_requests']}")
print(f"Успешно: {stats['successful_requests']} ({stats['success_rate']:.1f}%)")
print(f"Кэш попадания: {stats['cache_hits']} ({stats['cache_hit_rate']:.1f}%)")
print(f"Среднее время: {stats['avg_processing_time']:.3f} сек")
print(f"Повторные попытки: {stats['retry_attempts']}")
```

## Тестирование

### Запуск тестов

```bash
# Все тесты
pytest test_geocoder.py -v

# Конкретный тест
pytest test_geocoder.py::TestYandexGeocoder::test_geocode_request -v

# С покрытием кода
pytest --cov=geocoder_app test_geocoder.py

# С отчетом о покрытии
pytest --cov=geocoder_app --cov-report=html test_geocoder.py
```

### Структура тестов

```
test_geocoder.py
├── TestAddressValidator
│   ├── test_validate_valid_address
│   ├── test_validate_invalid_address
│   └── test_normalize_address
├── TestComponentProcessor
│   ├── test_process_components
│   ├── test_extract_specific_component
│   └── test_clear_cache
├── TestYandexGeocoder
│   ├── test_initialization
│   ├── test_stats_tracking
│   ├── test_geocode_request
│   └── test_reverse_geocode_validation
├── TestUtils
│   ├── test_validate_coordinates
│   └── test_calculate_distance
├── TestRetryStrategy
│   ├── test_should_retry
│   └── test_get_delay
└── TestExportStatistics
    └── test_export_statistics
```

### Мокирование API

Тесты используют мокирование для изоляции от реального API:

```python
@patch('geocoder_app.requests.Session')
def test_geocode_request(self, mock_session):
    # Настройка мока
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {...}
    
    # Выполнение теста
    results = self.geocoder.geocode("Москва, Красная площадь")
    
    # Проверка результатов
    self.assertGreater(len(results), 0)
```

## Развертывание

### Docker

Создайте Dockerfile:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY . .

# Переменные окружения
ENV PYTHONUNBUFFERED=1

# Запуск
CMD ["python", "geocoder_cli.py", "--help"]
```

Сборка и запуск:

```bash
docker build -t yandex-geocoder .
docker run -e YANDEX_API_KEY=your_key yandex-geocoder --address "Москва"
```

### Виртуальное окружение

```bash
# Создание виртуального окружения
python -m venv venv

# Активация (Windows)
venv\Scripts\activate

# Активация (Linux/Mac)
source venv/bin/activate

# Установка в режиме разработки
pip install -e .
```

### Контейнеризация с Docker Compose

```yaml
version: '3.8'

services:
  geocoder:
    build: .
    environment:
      - YANDEX_API_KEY=${YANDEX_API_KEY}
      - HTTP_PROXY=${HTTP_PROXY}
      - HTTPS_PROXY=${HTTPS_PROXY}
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    command: ["python", "geocoder_cli.py", "--input", "/data/addresses.csv"]
```

## Вопросы и ответы

### Частые вопросы

**Q: Как получить API ключ?**  
A: Зарегистрируйтесь на Yandex Cloud, создайте каталог и сгенерируйте ключ в разделе "API-ключи".

**Q: Какие лимиты у API?**  
A: Бесплатный тариф: 25,000 запросов в день. Подробнее на [developer.tech.yandex.ru](https://developer.tech.yandex.ru/).

**Q: Как увеличить производительность?**  
A: Используйте пакетную обработку, кэширование и настройте оптимальные rate limits.

**Q: Поддерживаются ли другие страны?**  
A: Да, Yandex Maps API поддерживает геокодирование по всему миру.

**Q: Как обрабатывать ошибки 429?**  
A: Приложение автоматически обрабатывает ошибки лимитов с экспоненциальной задержкой.

### Решение проблем

#### Ошибка "API ключ не найден"
1. Проверьте наличие файла `.env`
2. Убедитесь, что переменная `YANDEX_API_KEY` установлена
3. Проверьте правильность ключа

#### Медленная работа
1. Увеличьте `max_workers` для пакетной обработки
2. Проверьте настройки rate limit
3. Используйте кэширование

#### Ошибки сети
1. Проверьте подключение к интернету
2. Настройте прокси при необходимости
3. Увеличьте таймаут

## Лицензия

Этот проект распространяется под лицензией MIT. См. файл [LICENSE](LICENSE) для подробностей.

## Вклад в проект

Мы приветствуем вклады! Пожалуйста:

1. Форкните репозиторий
2. Создайте ветку для вашей функции
3. Добавьте тесты
4. Отправьте pull request

### Требования к коду
- Соответствие PEP 8
- Наличие документации
- Тестовое покрытие новых функций
- Логирование и обработка ошибок

## Контакты

- Автор: [Ваше Имя]
- Email: ваш.email@example.com
- Issues: [GitHub Issues](https://github.com/yourusername/yandex-geocoder/issues)
- Документация API: [Yandex Maps API](https://yandex.ru/dev/maps/geocoder/)

---

*Последнее обновление: Январь 2024*