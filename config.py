import os
from dotenv import load_dotenv

load_dotenv()

API_ID = int(os.getenv("API_ID"))
API_HASH = os.getenv("API_HASH")
SESSION_NAME = os.getenv("TELEGRAM_SESSION", "telegram_session")

CHANNELS = [
    "https://t.me/svodka38",
    "https://t.me/irkutsk_smi",
    "https://t.me/smi138",
    "https://t.me/smi_irkutsk",
    # "https://t.me/joinchat/qIxIXJnzY2kzNDAy",
    "https://t.me/irkdtp",
    "https://t.me/babr_mash",
   #  "https://tgstat.ru/channel/@irk01",
    "https://t.me/vestiirkutsk",
    "https://t.me/liveirkytsk",
   #  "https://t.me/joinchat/KunuAv8fWTEyZGYy",
    "https://t.me/novosti_irkutsk",
    "https://t.me/chp_irkutsk",
   #  "https://t.me/+xexYgU-DXaQ5ZWVi",
    # "https://t.me/irkseichas",
    "https://t.me/irkru",
    "https://t.me/irkutsk_poleznoe",
    "https://t.me/ircity_ru",
    # "https://telega.in/c/+XgJJut5uWa8zZjUy",
    "https://t.me/irk_edition",
    "https://t.me/onlirk",
   #  "https://t.me/joinchat/zEIdZLc-lugwNWUy",
   # "https://t.me/joinchat/KnFvGuPcwfFiOTNi",
    "https://t.me/irk_gorod",
    "https://t.me/irk_24",
    "https://t.me/aisttvru",
    "https://t.me/vpolezreniyaproekt",
    "https://t.me/odno_slovo",
    "https://t.me/baykaling",
   # "https://telega.in/c/+LfFnhOBn9R0zNzcy",
    "https://t.me/nts_irkutsk",
    "https://t.me/irkblog",
    "https://t.me/verbludvogne",
    "https://t.me/irkutskcorruption",
    "https://t.me/irktlgr"
]

TARGET_GROUP = os.getenv("TARGET_GROUP")  # -100... или @groupname

CHECK_INTERVAL = 300  # 5 минут

WORKERS = 3
QUEUE_MAXSIZE = 1000

# Конфигурация моделей для classifier.py
RUBERT_MODEL = 'cointegrated/rubert-tiny2'
USE_RUBERT = True
USE_QWEN = False  # Отключено, используем Qwen3-8B через spam_model.py
QWEN_MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'  # Fallback модель (не используется, если USE_QWEN=False)