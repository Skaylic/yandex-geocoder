import unittest
import tempfile
import json
from unittest.mock import Mock, patch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from geocoder_app import (
    YandexGeocoder,
    RussianAddressValidator,
    RetryStrategy,
    ComponentProcessor,
    validate_coordinates,
    calculate_distance
)


class TestAddressValidator(unittest.TestCase):

    def setUp(self):
        self.validator = RussianAddressValidator()

    def test_validate_valid_address(self):
        """Тест валидации корректного адреса"""
        result = self.validator.validate("Москва, ул. Ленина, д. 10")
        self.assertTrue(result.is_valid)
        self.assertGreater(result.confidence, 0.5)

    def test_validate_invalid_address(self):
        """Тест валидации некорректного адреса"""
        result = self.validator.validate("ул.")
        self.assertFalse(result.is_valid)
        self.assertEqual(result.confidence, 0.0)

    def test_normalize_address(self):
        """Тест нормализации адреса"""
        normalized = self.validator.normalize("Москва, ул. Ленина, д. 10")
        self.assertIn("улица", normalized)
        self.assertNotIn("ул.", normalized)


class TestComponentProcessor(unittest.TestCase):

    def setUp(self):
        self.components = [
            {"kind": "country", "name": "Россия"},
            {"kind": "locality", "name": "Москва"},
            {"kind": "street", "name": "Ленина"},
            {"kind": "house", "name": "10"}
        ]

    def test_process_components(self):
        """Тест обработки компонентов"""
        result = ComponentProcessor.process_components(self.components)

        self.assertEqual(result["Страна"], "Россия")
        self.assertEqual(result["Город"], "Москва")
        self.assertEqual(result["Улица"], "Ленина")
        self.assertEqual(result["Дом"], "10")

    def test_extract_specific_component(self):
        """Тест извлечения конкретного компонента"""
        country = ComponentProcessor.extract_specific_component(
            self.components, "country"
        )
        self.assertEqual(country, "Россия")

        missing = ComponentProcessor.extract_specific_component(
            self.components, "metro"
        )
        self.assertIsNone(missing)

    def test_clear_cache(self):
        """Тест очистки кэша"""
        # Сначала обработаем компоненты
        result1 = ComponentProcessor.process_components(self.components)

        # Очистим кэш
        ComponentProcessor.clear_cache()

        # Обработаем снова
        result2 = ComponentProcessor.process_components(self.components)

        # Результаты должны быть одинаковыми
        self.assertEqual(result1, result2)


class TestYandexGeocoder(unittest.TestCase):

    def setUp(self):
        # Мок API ключа для тестов
        self.api_key = "test_api_key"

        # Создаем геокодер с мок-сессией
        self.geocoder = YandexGeocoder(
            api_key=self.api_key,
            use_cache=False  # Отключаем кэш для чистоты тестов
        )

    def test_initialization(self):
        """Тест инициализации геокодера"""
        self.assertEqual(self.geocoder.api_key, self.api_key)
        self.assertIsInstance(self.geocoder.validator, RussianAddressValidator)

    def test_stats_tracking(self):
        """Тест отслеживания статистики"""
        stats = self.geocoder.get_stats()
        self.assertEqual(stats['total_requests'], 0)
        self.assertEqual(stats['successful_requests'], 0)

    @patch('geocoder_app.requests.Session')
    def test_geocode_request(self, mock_session):
        """Тест запроса геокодирования с моком"""
        # Настраиваем мок
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": {
                "GeoObjectCollection": {
                    "featureMember": [
                        {
                            "GeoObject": {
                                "name": "Красная площадь, 1",
                                "Point": {"pos": "37.617680 55.755277"},
                                "metaDataProperty": {
                                    "GeocoderMetaData": {
                                        "text": "Россия, Москва, Красная площадь, 1",
                                        "precision": "exact",
                                        "kind": "house",
                                        "Address": {
                                            "country_code": "RU",
                                            "Components": [
                                                {"kind": "country", "name": "Россия"},
                                                {"kind": "locality", "name": "Москва"},
                                                {"kind": "street", "name": "Красная площадь"},
                                                {"kind": "house", "name": "1"}
                                            ]
                                        }
                                    }
                                }
                            }
                        }
                    ]
                }
            }
        }

        mock_session_instance = Mock()
        mock_session_instance.get.return_value = mock_response
        mock_session.return_value = mock_session_instance

        # Заменяем сессию в геокодере
        self.geocoder.session = mock_session_instance

        # Выполняем геокодирование
        results = self.geocoder.geocode("Москва, Красная площадь, 1")

        # Проверяем результаты
        self.assertGreater(len(results), 0)
        result = results[0]
        self.assertEqual(result.full_address, "Россия, Москва, Красная площадь, 1")
        self.assertEqual(result.latitude, 55.755277)
        self.assertEqual(result.longitude, 37.617680)
        self.assertEqual(result.country_code, "RU")

    def test_reverse_geocode_validation(self):
        """Тест валидации координат при обратном геокодировании"""
        # Некорректные координаты
        with self.assertRaises(ValueError):
            self.geocoder.reverse_geocode(100, 200)

        # Корректные координаты
        try:
            self.geocoder.reverse_geocode(55.755277, 37.617680)
        except Exception:
            # Ожидаем ошибку API (нет мока), но не ошибку валидации
            pass

    def test_batch_geocode_empty(self):
        """Тест пакетного геокодирования с пустым списком"""
        results = self.geocoder.batch_geocode([], max_workers=1)
        self.assertEqual(len(results), 0)


class TestUtils(unittest.TestCase):

    def test_validate_coordinates(self):
        """Тест валидации координат"""
        self.assertTrue(validate_coordinates(55.755277, 37.617680))
        self.assertTrue(validate_coordinates(-90, -180))
        self.assertTrue(validate_coordinates(90, 180))

        self.assertFalse(validate_coordinates(100, 200))
        self.assertFalse(validate_coordinates(55.755277, 200))
        self.assertFalse(validate_coordinates(100, 37.617680))

    def test_calculate_distance(self):
        """Тест расчета расстояния"""
        # Москва - Санкт-Петербург
        moscow = (55.755277, 37.617680)
        spb = (59.934280, 30.335098)

        distance = calculate_distance(*moscow, *spb)

        # Расстояние примерно 630 км
        self.assertAlmostEqual(distance, 630, delta=50)

        # Нулевое расстояние
        distance_same = calculate_distance(*moscow, *moscow)
        self.assertAlmostEqual(distance_same, 0.0, delta=0.1)


class TestRetryStrategy(unittest.TestCase):

    def test_should_retry(self):
        """Тест проверки необходимости повторной попытки"""
        strategy = RetryStrategy()

        self.assertTrue(strategy.should_retry(429))
        self.assertTrue(strategy.should_retry(500))
        self.assertFalse(strategy.should_retry(200))
        self.assertFalse(strategy.should_retry(404))

    def test_get_delay(self):
        """Тест расчета задержки"""
        strategy = RetryStrategy(base_delay=1.0, backoff_factor=2.0)

        self.assertEqual(strategy.get_delay(0), 1.0)
        self.assertEqual(strategy.get_delay(1), 2.0)
        self.assertEqual(strategy.get_delay(2), 4.0)

        # Проверка ограничения максимальной задержки
        strategy_max = RetryStrategy(base_delay=10.0, max_delay=20.0)
        self.assertEqual(strategy_max.get_delay(3), 20.0)


class TestExportStatistics(unittest.TestCase):

    def test_export_statistics(self):
        """Тест экспорта статистики"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tmp_file = tmp.name

        try:
            geocoder = YandexGeocoder(api_key="test", use_cache=False)

            # Имитируем некоторую статистику
            geocoder._update_stats(success=True, processing_time=0.5)
            geocoder._update_stats(success=False, processing_time=0.2)

            # Экспортируем статистику
            stats = geocoder.export_statistics(tmp_file)

            # Проверяем, что файл создан и содержит данные
            with open(tmp_file, 'r', encoding='utf-8') as f:
                saved_stats = json.load(f)

            self.assertEqual(saved_stats['total_requests'], 2)
            self.assertEqual(saved_stats['successful_requests'], 1)
            self.assertEqual(saved_stats['failed_requests'], 1)
            self.assertIn('timestamp', saved_stats)

        finally:
            os.unlink(tmp_file)


if __name__ == '__main__':
    unittest.main(verbosity=2)