import asyncio
from aiogram.filters import Command
import os
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram import F
from aiogram.enums import ParseMode
from aiogram.utils.keyboard import ReplyKeyboardBuilder
from engine import ask


load_dotenv()

API_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GPT_MODEL = os.getenv("GPT_MODEL")

bot = Bot(token=API_TOKEN)
dp = Dispatcher()


@dp.message(Command("help"))
async def cmd_hello(message: Message):
    """
    Хэндлер на команду /help.
    Сообщает пользователю о назначении бота.
    :param message: Message
    :return: Message
    """
    await message.answer(
        f"<b>{message.from_user.last_name}.\nБот отвечает на вопросы о 37 церемонии Оскар.</b>",
        parse_mode=ParseMode.HTML,
    )


@dp.message(Command("start"))
async def cmd_start(message: types.Message) -> None:
    """
    Хэндлер на команду /start.
    :param message: Message
    :return: Message
    """
    builder = ReplyKeyboardBuilder()
    await message.answer(
        "Задайте вопрос боту.", reply_markup=builder.as_markup(resize_keyboard=True)
    )


@dp.message()
async def echo_handler(message: Message) -> None:
    """
    Хэндлер общения с ботом.
    :param message: Message
    :return: Message
    """
    await message.answer(ask(message.text))


async def main():
    """
    Запуск процесса поллинга новых апдейтов,
    :return:
    """

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
