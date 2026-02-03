import os
import sys
import json
import time
import logging
import threading
import hashlib
import re
from functools import lru_cache
from typing import Optional, Dict, Tuple, List, Any, Union, Generator
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, quote

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from ratelimit import limits, sleep_and_retry
from dotenv import load_dotenv

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('geocoder.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()


@dataclass
class AddressValidationResult:
    """Результат валидации адреса"""
    is_valid: bool
    normalized_address: str
    issues: List[str]
    confidence: float  # 0.0 - 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GeocodingResult:
    """Результат геокодирования"""
    request_id: str
    query: str
    full_address: str
    latitude: float
    longitude: float
    precision: str
    kind: str
    country_code: str
    postal_code: str
    components: Dict[str, str]
    confidence: float
    processing_time: float
    timestamp: datetime
    cache_hit: bool
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


class AddressValidator(ABC):
    """Абстрактный класс валидатора адресов"""

    @abstractmethod
    def validate(self, address: str) -> AddressValidationResult:
        pass

    @abstractmethod
    def normalize(self, address: str) -> str:
        pass


class RussianAddressValidator(AddressValidator):
    """Валидатор российских адресов"""

    def __init__(self):
        # Паттерны для валидации
        self.patterns = {
            'postal_code': r'\b\d{6}\b',
            'house_number': r'д\.?\s*\d+[а-я]?\b|\b\d+[а-я]?\b',
            'street': r'(ул\.?|улица|пр\.?|проспект|пер\.?|переулок|б-р|бульвар|наб\.|набережная|ш\.|шоссе)\s+[^\d,;]+',
            'city': r'\b(г\.?|город|пгт|посёлок|с\.|село|д\.|деревня)\s+[^,;]+',
            'region': r'\b(обл\.?|область|край|респ\.?|республика|ао|автономный округ)\s+[^,;]+'
        }

        # Словари для нормализации
        self.normalization_map = {
            'ул.': 'улица',
            'улица': 'улица',
            'пр.': 'проспект',
            'проспект': 'проспект',
            'пер.': 'переулок',
            'переулок': 'переулок',
            'г.': 'город',
            'город': 'город',
            'обл.': 'область',
            'область': 'область',
            'д.': 'дом',
            'дом': 'дом',
            'корп.': 'корпус',
            'корпус': 'корпус',
            'стр.': 'строение',
            'строение': 'строение'
        }

    def validate(self, address: str) -> AddressValidationResult:
        """Валидация адреса"""
        normalized = self.normalize(address)
        issues = []
        confidence = 1.0

        # Проверка пустого адреса
        if not address or len(address.strip()) < 3:
            return AddressValidationResult(
                is_valid=False,
                normalized_address=normalized,
                issues=['Адрес слишком короткий'],
                confidence=0.0
            )

        # Проверка наличия города
        if not re.search(self.patterns['city'], normalized, re.IGNORECASE):
            issues.append('Не указан населенный пункт')
            confidence *= 0.7

        # Проверка наличия улицы
        if not re.search(self.patterns['street'], normalized, re.IGNORECASE):
            issues.append('Не указана улица')
            confidence *= 0.8

        # Проверка номера дома
        if not re.search(self.patterns['house_number'], normalized, re.IGNORECASE):
            issues.append('Не указан номер дома')
            confidence *= 0.9

        # Проверка почтового индекса (опционально)
        if re.search(self.patterns['postal_code'], normalized):
            confidence *= 1.1  # Бонус за индекс

        # Ограничение confidence в диапазоне 0.0-1.0
        confidence = max(0.0, min(1.0, confidence))

        return AddressValidationResult(
            is_valid=len(issues) < 3,  # Допускаем до 2 предупреждений
            normalized_address=normalized,
            issues=issues,
            confidence=confidence
        )

    def normalize(self, address: str) -> str:
        """Нормализация адреса"""
        if not address:
            return ""

        # Приведение к нижнему регистру
        normalized = address.lower()

        # Удаление лишних пробелов
        normalized = ' '.join(normalized.split())

        # Замена сокращений на полные названия
        for short, full in self.normalization_map.items():
            normalized = re.sub(rf'\b{re.escape(short)}\b', full, normalized)

        # Стандартизация разделителей
        normalized = normalized.replace(',', ', ')
        normalized = re.sub(r'\s+,\s+', ', ', normalized)

        # Удаление повторяющихся слов
        words = normalized.split()
        unique_words = []
        for word in words:
            if word not in unique_words[-3:]:  # Проверяем последние 3 слова
                unique_words.append(word)

        return ' '.join(unique_words).strip()


class RateLimiter:
    """Ограничитель частоты запросов"""

    def __init__(self, calls_per_second: float = 0.5):
        """
        Args:
            calls_per_second: Количество вызовов в секунду
        """
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call_time = 0
        self.lock = threading.Lock()

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self.lock:
                current_time = time.time()
                elapsed = current_time - self.last_call_time

                if elapsed < self.min_interval:
                    sleep_time = self.min_interval - elapsed
                    logger.debug(f"Rate limiting: sleeping {sleep_time:.2f} seconds")
                    time.sleep(sleep_time)

                self.last_call_time = time.time()

            return func(*args, **kwargs)

        return wrapper


class RetryStrategy:
    """Стратегия повторных попыток"""

    def __init__(self,
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 10.0,
                 backoff_factor: float = 2.0,
                 retry_on_status: List[int] = None):
        """
        Args:
            max_retries: Максимальное количество попыток
            base_delay: Базовая задержка в секундах
            max_delay: Максимальная задержка в секундах
            backoff_factor: Коэффициент экспоненциальной задержки
            retry_on_status: Список HTTP статусов для повторных попыток
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.retry_on_status = retry_on_status or [429, 500, 502, 503, 504]

    def should_retry(self, status_code: int) -> bool:
        """Проверить, нужно ли повторять запрос"""
        return status_code in self.retry_on_status

    def get_delay(self, attempt: int) -> float:
        """Получить задержку для попытки"""
        delay = self.base_delay * (self.backoff_factor ** attempt)
        return min(delay, self.max_delay)


class ComponentProcessor:
    """Процессор для оптимизированной обработки компонентов адреса"""

    # Кэш для компонентов
    _component_cache = {}
    _cache_lock = threading.Lock()

    # Маппинг kind → русское название
    COMPONENT_MAPPING = {
        'country': 'Страна',
        'province': 'Область',
        'area': 'Район',
        'region': 'Регион',
        'locality': 'Город',
        'district': 'Район города',
        'street': 'Улица',
        'house': 'Дом',
        'postal_code': 'Почтовый индекс',
        'entrance': 'Подъезд',
        'apartment': 'Квартира',
        'flat': 'Квартира',
        'room': 'Комната',
        'metro': 'Метро'
    }

    # Приоритеты компонентов для сортировки
    COMPONENT_PRIORITY = {
        'country': 1,
        'province': 2,
        'region': 3,
        'area': 4,
        'locality': 5,
        'district': 6,
        'street': 7,
        'house': 8,
        'postal_code': 9,
        'entrance': 10,
        'apartment': 11,
        'flat': 12,
        'room': 13,
        'metro': 14
    }

    @classmethod
    def process_components(cls, components: List[Dict[str, str]]) -> Dict[str, str]:
        """Обработать компоненты адреса"""
        cache_key = hashlib.md5(
            json.dumps(components, sort_keys=True).encode()
        ).hexdigest()

        with cls._cache_lock:
            if cache_key in cls._component_cache:
                return cls._component_cache[cache_key]

        result = {}
        for component in components:
            kind = component.get('kind', '')
            name = component.get('name', '')

            if kind in cls.COMPONENT_MAPPING:
                ru_name = cls.COMPONENT_MAPPING[kind]
                result[ru_name] = name

            # Также сохраняем оригинальный kind
            result[f'kind_{kind}'] = name

        # Сортируем по приоритету
        sorted_result = {}
        for kind in sorted(result.keys(),
                           key=lambda x: cls.COMPONENT_PRIORITY.get(
                               x.replace('kind_', ''), 100)):
            sorted_result[kind] = result[kind]

        with cls._cache_lock:
            cls._component_cache[cache_key] = sorted_result

        return sorted_result

    @classmethod
    def extract_specific_component(cls,
                                   components: List[Dict[str, str]],
                                   target_kind: str) -> Optional[str]:
        """Извлечь конкретный компонент"""
        for component in components:
            if component.get('kind') == target_kind:
                return component.get('name')
        return None

    @classmethod
    def clear_cache(cls):
        """Очистить кэш компонентов"""
        with cls._cache_lock:
            cls._component_cache.clear()


class YandexGeocoder:
    """Улучшенный геокодер Yandex с поддержкой всех функций"""

    # Корректные URL API
    GEOCODE_URL = "https://geocode-maps.yandex.ru/1.x/"
    REVERSE_GEOCODE_URL = "https://geocode-maps.yandex.ru/1.x/"
    SEARCH_URL = "https://search-maps.yandex.ru/v1/"

    def __init__(self,
                 api_key: Optional[str] = None,
                 timeout: int = 30,
                 retry_strategy: Optional[RetryStrategy] = None,
                 rate_limit: float = 0.5,  # 0.5 запроса в секунду = 30 в минуту
                 proxy: Optional[str] = None,
                 use_cache: bool = True):
        """
        Инициализация геокодера

        Args:
            api_key: API ключ Yandex
            timeout: Таймаут запросов в секундах
            retry_strategy: Стратегия повторных попыток
            rate_limit: Лимит запросов в секунду
            proxy: URL прокси-сервера
            use_cache: Использовать кэширование
        """
        self.api_key = api_key or os.getenv('YANDEX_API_KEY')
        if not self.api_key:
            raise ValueError(
                "API ключ не найден. "
                "Укажите в .env файле (YANDEX_API_KEY) или передайте напрямую"
            )

        self.timeout = timeout
        self.retry_strategy = retry_strategy or RetryStrategy()
        self.rate_limit = rate_limit
        self.use_cache = use_cache

        # Валидатор адресов
        self.validator = RussianAddressValidator()

        # Ограничитель частоты
        self.rate_limiter = RateLimiter(calls_per_second=rate_limit)

        # Инициализация сессии с retry
        self.session = self._create_session(proxy)

        # Статистика
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'retry_attempts': 0,
            'total_processing_time': 0.0
        }
        self.stats_lock = threading.Lock()

    def _create_session(self, proxy: Optional[str]) -> requests.Session:
        """Создать сессию с настройками retry и прокси"""
        session = requests.Session()

        # Настройка retry
        retry = Retry(
            total=self.retry_strategy.max_retries,
            backoff_factor=self.retry_strategy.base_delay,
            status_forcelist=self.retry_strategy.retry_on_status,
            allowed_methods=['GET']
        )

        adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=10,
            pool_maxsize=100
        )

        session.mount('http://', adapter)
        session.mount('https://', adapter)

        # Настройка прокси
        if proxy:
            session.proxies = {
                'http': proxy,
                'https': proxy
            }

        # Заголовки
        session.headers.update({
            'User-Agent': 'YandexGeocoder/2.0 (+https://github.com/your-repo)',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate'
        })

        return session

    def _update_stats(self,
                      success: bool = True,
                      cache_hit: bool = False,
                      processing_time: float = 0.0,
                      retry_count: int = 0):
        """Обновить статистику"""
        with self.stats_lock:
            self.stats['total_requests'] += 1
            if success:
                self.stats['successful_requests'] += 1
            else:
                self.stats['failed_requests'] += 1

            if cache_hit:
                self.stats['cache_hits'] += 1
            else:
                self.stats['cache_misses'] += 1

            self.stats['retry_attempts'] += retry_count
            self.stats['total_processing_time'] += processing_time

    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику"""
        with self.stats_lock:
            stats = self.stats.copy()

        if stats['total_requests'] > 0:
            stats['success_rate'] = (
                    stats['successful_requests'] / stats['total_requests'] * 100
            )
            stats['cache_hit_rate'] = (
                    stats['cache_hits'] / max(stats['total_requests'], 1) * 100
            )
            stats['avg_processing_time'] = (
                    stats['total_processing_time'] / max(stats['total_requests'], 1)
            )
        else:
            stats.update({
                'success_rate': 0.0,
                'cache_hit_rate': 0.0,
                'avg_processing_time': 0.0
            })

        return stats

    def _make_request(self,
                      url: str,
                      params: Dict[str, Any],
                      operation: str = "geocode") -> Dict[str, Any]:
        """Выполнить запрос к API с учетом всех ограничений"""
        start_time = time.time()
        cache_hit = False
        retry_count = 0

        try:
            # Применить ограничение частоты
            @self.rate_limiter
            def execute_request():
                logger.debug(f"Выполнение {operation} запроса: {params.get('geocode', '')[:50]}...")
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.timeout
                )

                # Обработка специфических ошибок
                if response.status_code == 429:
                    raise ConnectionError("Превышен лимит запросов к API")
                elif response.status_code == 403:
                    raise ConnectionError("Неверный или просроченный API ключ")
                elif response.status_code == 400:
                    raise ValueError(f"Некорректный запрос: {response.text[:100]}")

                response.raise_for_status()
                return response.json()

            result = execute_request()

            processing_time = time.time() - start_time
            self._update_stats(
                success=True,
                cache_hit=cache_hit,
                processing_time=processing_time,
                retry_count=retry_count
            )

            return result

        except requests.exceptions.Timeout:
            logger.error(f"Таймаут запроса ({self.timeout} секунд)")
            raise ConnectionError(f"Таймаут запроса")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Ошибка соединения: {e}")
            raise ConnectionError(f"Ошибка соединения: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Неверный формат ответа: {e}")
            raise ValueError(f"Неверный формат ответа API")
        except Exception as e:
            logger.error(f"Ошибка при выполнении запроса: {e}")
            processing_time = time.time() - start_time
            self._update_stats(success=False, processing_time=processing_time)
            raise

    @lru_cache(maxsize=1024)
    def _cached_geocode(self,
                        address_hash: str,
                        results: int = 10,
                        lang: str = 'ru_RU') -> Dict[str, Any]:
        """Кэшированное геокодирование"""
        # Этот метод никогда не вызывается напрямую
        pass

    def geocode_raw(self,
                    address: str,
                    format: str = 'json',
                    results: int = 10,
                    lang: str = 'ru_RU') -> Dict[str, Any]:
        """
        Прямое геокодирование: адрес → координаты (сырой ответ)

        Args:
            address: Адрес для поиска
            format: Формат ответа
            results: Количество результатов
            lang: Язык ответа

        Returns:
            Сырой ответ API
        """
        # Валидация адреса
        validation = self.validator.validate(address)
        if not validation.is_valid:
            logger.warning(f"Адрес не прошел валидацию: {address}")
            logger.warning(f"Проблемы: {validation.issues}")

        params = {
            'apikey': self.api_key,
            'geocode': validation.normalized_address,
            'format': format,
            'results': min(max(results, 1), 100),
            'lang': lang
        }

        return self._make_request(self.GEOCODE_URL, params, "geocode")

    def geocode(self,
                address: str,
                max_results: int = 5,
                lang: str = 'ru_RU') -> List[GeocodingResult]:
        """
        Геокодирование с полной обработкой

        Args:
            address: Адрес для поиска
            max_results: Максимальное количество результатов
            lang: Язык ответа

        Returns:
            Список результатов геокодирования
        """
        start_time = time.time()
        cache_hit = False

        # Проверка кэша
        address_hash = hashlib.md5(
            f"{address}_{max_results}_{lang}".encode()
        ).hexdigest()

        if self.use_cache:
            cached_result = getattr(self, '_cache_dict', {}).get(address_hash)
            if cached_result:
                logger.debug(f"Кэш-попадание для адреса: {address[:50]}...")
                cache_hit = True
                self._update_stats(cache_hit=True)
                return cached_result

        # Выполнение запроса
        raw_data = self.geocode_raw(address, results=max_results, lang=lang)
        results = []

        try:
            features = raw_data['response']['GeoObjectCollection']['featureMember']

            for feature in features[:max_results]:
                geo_object = feature['GeoObject']
                meta = geo_object['metaDataProperty']['GeocoderMetaData']

                # Координаты
                pos = geo_object['Point']['pos']  # "lon lat"
                lon, lat = map(float, pos.split())

                # Адресная информация
                address_info = meta.get('Address', {})
                components_raw = address_info.get('Components', [])

                # Обработка компонентов
                components = ComponentProcessor.process_components(components_raw)

                # Извлечение конкретных полей
                country_code = address_info.get('country_code', '')
                postal_code = ComponentProcessor.extract_specific_component(
                    components_raw, 'postal_code'
                ) or ''

                # Создание результата
                result = GeocodingResult(
                    request_id=address_hash[:8],
                    query=address,
                    full_address=meta.get('text', ''),
                    latitude=lat,
                    longitude=lon,
                    precision=meta.get('precision', ''),
                    kind=meta.get('kind', ''),
                    country_code=country_code,
                    postal_code=postal_code,
                    components=components,
                    confidence=self.validator.validate(address).confidence,
                    processing_time=time.time() - start_time,
                    timestamp=datetime.now(),
                    cache_hit=cache_hit
                )

                results.append(result)

        except (KeyError, IndexError, ValueError) as e:
            logger.error(f"Ошибка при обработке ответа: {e}")

        # Сохранение в кэш
        if self.use_cache and results:
            if not hasattr(self, '_cache_dict'):
                self._cache_dict = {}
            self._cache_dict[address_hash] = results
            self._cache_dict[address_hash + '_timestamp'] = time.time()

        return results

    def reverse_geocode(self,
                        lat: float,
                        lon: float,
                        lang: str = 'ru_RU') -> Optional[GeocodingResult]:
        """
        Обратное геокодирование: координаты → адрес

        Args:
            lat: Широта
            lon: Долгота
            lang: Язык ответа

        Returns:
            Результат геокодирования или None
        """
        # Валидация координат
        if not (-90 <= lat <= 90):
            raise ValueError("Широта должна быть в диапазоне от -90 до 90")
        if not (-180 <= lon <= 180):
            raise ValueError("Долгота должна быть в диапазоне от -180 до 180")

        coords = f"{lon},{lat}"

        params = {
            'apikey': self.api_key,
            'geocode': coords,
            'format': 'json',
            'lang': lang
        }

        try:
            raw_data = self._make_request(
                self.REVERSE_GEOCODE_URL, params, "reverse_geocode"
            )

            features = raw_data['response']['GeoObjectCollection']['featureMember']
            if not features:
                return None

            feature = features[0]
            geo_object = feature['GeoObject']
            meta = geo_object['metaDataProperty']['GeocoderMetaData']
            address_info = meta.get('Address', {})
            components_raw = address_info.get('Components', [])

            # Обработка компонентов
            components = ComponentProcessor.process_components(components_raw)

            # Извлечение почтового индекса
            postal_code = ComponentProcessor.extract_specific_component(
                components_raw, 'postal_code'
            ) or ''

            return GeocodingResult(
                request_id=hashlib.md5(coords.encode()).hexdigest()[:8],
                query=coords,
                full_address=meta.get('text', ''),
                latitude=lat,
                longitude=lon,
                precision=meta.get('precision', ''),
                kind=meta.get('kind', ''),
                country_code=address_info.get('country_code', ''),
                postal_code=postal_code,
                components=components,
                confidence=1.0,
                processing_time=0.0,
                timestamp=datetime.now(),
                cache_hit=False
            )

        except Exception as e:
            logger.error(f"Ошибка при обратном геокодировании: {e}")
            return None

    def batch_geocode(self,
                      addresses: List[str],
                      max_workers: int = 4,
                      batch_size: int = 10,
                      delay_between_batches: float = 1.0) -> List[List[GeocodingResult]]:
        """
        Пакетное геокодирование с многопоточностью

        Args:
            addresses: Список адресов
            max_workers: Максимальное количество потоков
            batch_size: Размер пачки для одного потока
            delay_between_batches: Задержка между пачками

        Returns:
            Список списков результатов
        """
        logger.info(f"Начало пакетного геокодирования {len(addresses)} адресов")

        all_results = []
        total_batches = (len(addresses) + batch_size - 1) // batch_size

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(addresses))
                batch_addresses = addresses[start_idx:end_idx]

                # Задержка между отправкой пачек
                if batch_num > 0:
                    time.sleep(delay_between_batches)

                # Отправка пачки на обработку
                future = executor.submit(self._process_batch, batch_addresses, batch_num)
                futures.append((batch_num, future))

            # Сбор результатов
            for batch_num, future in futures:
                try:
                    batch_results = future.result(timeout=self.timeout * 2)
                    all_results.extend(batch_results)
                    logger.info(f"Пачка {batch_num + 1}/{total_batches} обработана")
                except Exception as e:
                    logger.error(f"Ошибка при обработке пачки {batch_num}: {e}")
                    all_results.append([])  # Пустой список для неудачной пачки

        logger.info(f"Пакетное геокодирование завершено")
        return all_results

    def _process_batch(self,
                       addresses: List[str],
                       batch_num: int) -> List[List[GeocodingResult]]:
        """Обработать пачку адресов"""
        batch_results = []

        for i, address in enumerate(addresses):
            try:
                results = self.geocode(address)
                batch_results.append(results)

                # Логирование прогресса
                if (i + 1) % 5 == 0:
                    logger.debug(
                        f"Пачка {batch_num}: обработано {i + 1}/{len(addresses)}"
                    )

            except Exception as e:
                logger.error(f"Ошибка при обработке адреса в пачке {batch_num}: {e}")
                batch_results.append([])

        return batch_results

    def clear_cache(self):
        """Очистить кэш"""
        if hasattr(self, '_cache_dict'):
            self._cache_dict.clear()
        ComponentProcessor.clear_cache()
        logger.info("Кэш геокодера очищен")

    def export_statistics(self, filename: str = "geocoder_stats.json"):
        """Экспортировать статистику в файл"""
        stats = self.get_stats()
        stats['timestamp'] = datetime.now().isoformat()

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        logger.info(f"Статистика экспортирована в {filename}")
        return stats


# Утилиты для работы с данными
def validate_coordinates(lat: float, lon: float) -> bool:
    """Проверить корректность координат"""
    return (-90 <= lat <= 90) and (-180 <= lon <= 180)


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Рассчитать расстояние между двумя точками (км)"""
    from math import radians, sin, cos, sqrt, atan2

    R = 6371.0  # Радиус Земли в км

    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


# Пример использования
if __name__ == "__main__":
    # Демонстрация всех функций
    try:
        # Инициализация с продвинутыми настройками
        geocoder = YandexGeocoder(
            timeout=30,
            rate_limit=0.3,  # ~20 запросов в минуту
            retry_strategy=RetryStrategy(
                max_retries=3,
                base_delay=1.0,
                backoff_factor=2.0
            ),
            use_cache=True
        )

        print("=== Демонстрация работы улучшенного геокодера ===")

        # 1. Валидация адреса
        print("\n1. Валидация адресов:")
        test_addresses = [
            "Москва, Красная площадь, 1",
            "ул. Ленина",
            "г. Санкт-Петербург, Невский пр-т, д. 28",
            "некорректный адрес"
        ]

        for addr in test_addresses:
            validation = geocoder.validator.validate(addr)
            status = "✓" if validation.is_valid else "✗"
            print(f"  {status} {addr}")
            if validation.issues:
                print(f"    Проблемы: {', '.join(validation.issues)}")

        # 2. Одиночное геокодирование
        print("\n2. Геокодирование одиночного адреса:")
        results = geocoder.geocode("Москва, Красная площадь, 1", max_results=1)

        if results:
            result = results[0]
            print(f"  Адрес: {result.full_address}")
            print(f"  Координаты: {result.latitude:.6f}, {result.longitude:.6f}")
            print(f"  Точность: {result.precision}")
            print(f"  Почтовый индекс: {result.postal_code}")
            print(f"  Уверенность: {result.confidence:.2f}")

        # 3. Пакетное геокодирование
        print("\n3. Пакетное геокодирование (демо):")
        addresses = [
            "Санкт-Петербург, Невский проспект, 28",
            "Екатеринбург, площадь 1905 года",
            "Казань, Кремль",
            "Новосибирск, Красный проспект, 1"
        ]

        batch_results = geocoder.batch_geocode(
            addresses,
            max_workers=2,
            batch_size=2,
            delay_between_batches=0.5
        )

        print(f"  Обработано {len(batch_results)} пачек")

        # 4. Статистика
        print("\n4. Статистика работы:")
        stats = geocoder.get_stats()
        print(f"  Всего запросов: {stats['total_requests']}")
        print(f"  Успешно: {stats['successful_requests']}")
        print(f"  Кэш-попадания: {stats['cache_hits']} ({stats.get('cache_hit_rate', 0):.1f}%)")
        print(f"  Среднее время: {stats.get('avg_processing_time', 0):.3f} сек")

        # 5. Экспорт статистики
        geocoder.export_statistics("statistics.json")
        print("\n5. Статистика экспортирована в statistics.json")

        # 6. Очистка кэша
        if input("\nОчистить кэш? (y/N): ").lower() == 'y':
            geocoder.clear_cache()

    except Exception as e:
        logger.error(f"Ошибка при демонстрации: {e}")