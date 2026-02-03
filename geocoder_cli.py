#!/usr/bin/env python3
"""
Командная строка для геокодера
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from geocoder_app import YandexGeocoder, export_to_excel, batch_geocode


def parse_arguments():
    """Разбор аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description='Геокодирование адресов с использованием Yandex Maps API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s --address "Москва, Красная площадь"
  %(prog)s --input addresses.csv --output results.xlsx
  %(prog)s --batch --input addresses.txt --workers 4
  %(prog)s --stats --export stats.json
        """
    )

    # Основные команды
    command_group = parser.add_mutually_exclusive_group(required=True)
    command_group.add_argument('--address', type=str, help='Адрес для геокодирования')
    command_group.add_argument('--reverse', nargs=2, type=float,
                               metavar=('LAT', 'LON'), help='Обратное геокодирование')
    command_group.add_argument('--input', type=str, help='Файл с адресами для пакетной обработки')
    command_group.add_argument('--stats', action='store_true', help='Показать статистику')
    command_group.add_argument('--validate', type=str, help='Проверить адрес')

    # Параметры
    parser.add_argument('--output', type=str, default='geocoding_results.xlsx',
                        help='Выходной файл для результатов')
    parser.add_argument('--format', choices=['excel', 'csv', 'json'], default='excel',
                        help='Формат вывода')
    parser.add_argument('--api-key', type=str, help='API ключ Yandex')
    parser.add_argument('--proxy', type=str, help='URL прокси-сервера')
    parser.add_argument('--workers', type=int, default=4,
                        help='Количество потоков для пакетной обработки')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Размер пачки для обработки')
    parser.add_argument('--rate-limit', type=float, default=0.5,
                        help='Лимит запросов в секунду')
    parser.add_argument('--timeout', type=int, default=30,
                        help='Таймаут запросов в секундах')
    parser.add_argument('--retries', type=int, default=3,
                        help='Количество повторных попыток')
    parser.add_argument('--clear-cache', action='store_true',
                        help='Очистить кэш перед выполнением')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Подробный вывод')

    return parser.parse_args()


def read_addresses_from_file(filename: str) -> list:
    """Прочитать адреса из файла"""
    file_path = Path(filename)

    if not file_path.exists():
        print(f"Ошибка: файл {filename} не найден")
        sys.exit(1)

    if file_path.suffix.lower() == '.csv':
        try:
            df = pd.read_csv(filename)
            # Автоопределение колонки с адресами
            address_columns = [col for col in df.columns
                               if any(keyword in col.lower() for keyword in
                                      ['адрес', 'address', 'улица', 'street'])]

            if not address_columns:
                print("Ошибка: не найдена колонка с адресами в CSV файле")
                sys.exit(1)

            addresses = df[address_columns[0]].dropna().astype(str).tolist()
            return addresses

        except Exception as e:
            print(f"Ошибка при чтении CSV файла: {e}")
            sys.exit(1)

    elif file_path.suffix.lower() in ['.txt', '.text']:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                addresses = [line.strip() for line in f if line.strip()]
            return addresses
        except Exception as e:
            print(f"Ошибка при чтении текстового файла: {e}")
            sys.exit(1)

    elif file_path.suffix.lower() in ['.xlsx', '.xls']:
        try:
            df = pd.read_excel(filename)
            # Автоопределение колонки с адресами
            address_columns = [col for col in df.columns
                               if any(keyword in col.lower() for keyword in
                                      ['адрес', 'address', 'улица', 'street'])]

            if not address_columns:
                print("Ошибка: не найдена колонка с адресами в Excel файле")
                sys.exit(1)

            addresses = df[address_columns[0]].dropna().astype(str).tolist()
            return addresses

        except Exception as e:
            print(f"Ошибка при чтении Excel файла: {e}")
            sys.exit(1)

    else:
        print(f"Ошибка: неподдерживаемый формат файла {file_path.suffix}")
        sys.exit(1)


def main():
    """Основная функция"""
    args = parse_arguments()

    # Настройка геокодера
    geocoder = YandexGeocoder(
        api_key=args.api_key,
        timeout=args.timeout,
        rate_limit=args.rate_limit,
        proxy=args.proxy,
        use_cache=True
    )

    # Очистка кэша если нужно
    if args.clear_cache:
        geocoder.clear_cache()
        print("Кэш очищен")

    try:
        # Выполнение команд
        if args.address:
            print(f"Геокодирование адреса: {args.address}")
            results = geocoder.geocode(args.address, max_results=3)

            if results:
                print("\nРезультаты:")
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. {result.full_address}")
                    print(f"   Координаты: {result.latitude:.6f}, {result.longitude:.6f}")
                    print(f"   Точность: {result.precision}")
                    print(f"   Почтовый индекс: {result.postal_code}")
            else:
                print("Адрес не найден")

        elif args.reverse:
            lat, lon = args.reverse
            print(f"Обратное геокодирование: {lat}, {lon}")
            result = geocoder.reverse_geocode(lat, lon)

            if result:
                print(f"\nАдрес: {result.full_address}")
                print(f"Точность: {result.precision}")
                print(f"Почтовый индекс: {result.postal_code}")
            else:
                print("Адрес не найден для указанных координат")

        elif args.input:
            addresses = read_addresses_from_file(args.input)
            print(f"Найдено {len(addresses)} адресов для обработки")

            if args.format == 'excel':
                df = export_to_excel(
                    geocoder,
                    addresses,
                    output_file=args.output,
                    include_components=True
                )
                print(f"\nРезультаты сохранены в {args.output}")
                print(f"Обработано {len(df)} записей")

            elif args.format == 'csv':
                results = geocoder.batch_geocode(
                    addresses,
                    max_workers=args.workers,
                    batch_size=args.batch_size
                )

                # Преобразование результатов в плоский список
                flat_results = []
                for batch in results:
                    for result_list in batch:
                        if result_list:
                            for result in result_list:
                                flat_results.append(result.to_dict())

                df = pd.DataFrame(flat_results)
                df.to_csv(args.output, index=False, encoding='utf-8-sig')
                print(f"\nРезультаты сохранены в {args.output}")
                print(f"Обработано {len(df)} записей")

            else:  # json
                results = geocoder.batch_geocode(
                    addresses,
                    max_workers=args.workers,
                    batch_size=args.batch_size
                )

                import json
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

                print(f"\nРезультаты сохранены в {args.output}")

        elif args.stats:
            stats = geocoder.get_stats()
            print("\nСтатистика работы геокодера:")
            print(f"  Всего запросов: {stats['total_requests']}")
            print(f"  Успешно: {stats['successful_requests']}")
            print(f"  Неудачно: {stats['failed_requests']}")
            print(f"  Кэш-попадания: {stats['cache_hits']}")
            print(f"  Промахи кэша: {stats['cache_misses']}")
            print(f"  Попытки повтора: {stats['retry_attempts']}")
            print(f"  Процент успеха: {stats.get('success_rate', 0):.1f}%")
            print(f"  Эффективность кэша: {stats.get('cache_hit_rate', 0):.1f}%")
            print(f"  Среднее время: {stats.get('avg_processing_time', 0):.3f} сек")

            if args.output != 'geocoding_results.xlsx':
                geocoder.export_statistics(args.output)
                print(f"\nСтатистика экспортирована в {args.output}")

        elif args.validate:
            validation = geocoder.validator.validate(args.validate)
            print(f"\nВалидация адреса: {args.validate}")
            print(f"  Корректность: {'✓' if validation.is_valid else '✗'}")
            print(f"  Уверенность: {validation.confidence:.2f}")
            print(f"  Нормализованный: {validation.normalized_address}")

            if validation.issues:
                print(f"  Проблемы: {', '.join(validation.issues)}")

        # Вывод статистики после выполнения (если не была запрошена отдельно)
        if not args.stats and args.address or args.input or args.reverse:
            stats = geocoder.get_stats()
            print(f"\nСтатистика выполнения:")
            print(f"  Запросов: {stats['total_requests']}")
            print(f"  Время: {stats.get('avg_processing_time', 0):.3f} сек/запрос")

    except KeyboardInterrupt:
        print("\n\nПрервано пользователем")
        sys.exit(0)
    except Exception as e:
        print(f"\nОшибка: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()