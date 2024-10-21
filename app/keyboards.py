from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.utils.keyboard import ReplyKeyboardBuilder, InlineKeyboardBuilder

main = ReplyKeyboardMarkup(keyboard= [
    [KeyboardButton(text = 'Поиск по фото')],[KeyboardButton(text ='Поиск аналогов по названию')],
    [KeyboardButton(text ='Поиск лекарств по описанию симптомов')],[KeyboardButton(text ='Назад')]
],
                           resize_keyboard=True,
                           input_field_placeholder= 'Выберите действие')

async def found_medicines(lst):
    keyboard =InlineKeyboardBuilder()
    for i in lst:
        keyboard.add(InlineKeyboardButton(text= i, callback_data= i))
    return keyboard.adjust(2).as_markup()


voic = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text = 'Да', callback_data= 'actice_voice')], [InlineKeyboardButton(text= 'Нет', callback_data='deactive_voice')],

 ])
