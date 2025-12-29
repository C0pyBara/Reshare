# Telegram -> ClickHouse


Простой проект на Telethon для запуска под учётной записью пользователя (не Bot API): читает новые сообщения из каналов и сохраняет их в ClickHouse.


## Требования
- Python 3.10+
- (локально, без Docker) (рекомендовано для ClickHouse)


## Быстрый старт (локально с (локально, без Docker))


1. Скопируйте проект
2. Создайте `.env` на основе `.env.example` и заполните поля (`API_ID`, `API_HASH`, `TELEGRAM_SESSION` при необходимости)
3. Запустите ClickHouse: `docker compose up -d`
4. Установите зависимости: `pip install -r requirements.txt`
5. Отредактируйте `config.py` и укажите каналы
6. Запустите: `python main.py`


## Примечание
- Telethon при первом запуске попросит код подтверждения — получите его в вашем Telegram.
- ClickHouse клиент слушает по умолчанию на `localhost:9000`.