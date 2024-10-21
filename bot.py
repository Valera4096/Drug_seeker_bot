from aiogram import Bot, Dispatcher
import logging
from app.handlers import router
from config import TOKEN


from aiogram.fsm.storage.memory import MemoryStorage


storage = MemoryStorage()


# Создаем объекты бота и диспетчера
bot = Bot(token=TOKEN)
dp = Dispatcher(storage=storage)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    dp.include_router(router)
    dp.run_polling(bot)