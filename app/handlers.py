from aiogram.types import Message, CallbackQuery, FSInputFile
import os
import logging
from aiogram.filters import CommandStart, Command
from aiogram import Router, F
import app.keyboards as kb
from funcs.funcs import load_model_yolo, get_prediction, search_drug_by_action, analogue_search, load_model_vectr, load_df, text, load_voice_model, speech_recognition, text_voice, synthesize


from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext

from aiogram.enums import ParseMode



#####################################
model_yolo = load_model_yolo()      #
model_vector = load_model_vectr()   #
model_vector.to("cpu")              #--------------> Загрузка моделей и Датафрейма 
voice_model = load_voice_model()
df = load_df()                      #
#####################################


###############################-----> Объявление класса для контекста
class LEN_ANALOG(StatesGroup):
    length_analog = State()
    length_medicals = State()
    photo = State()
    res = State()
    text_label = State()
    text_pain = State()
    description_or_name = State()
###############################

router = Router()

###################################################################################################################################
# Это хендлер будет срабатывать на команду "/start"                                                                               #
@router.message(CommandStart())                                                                                                   #
async def process_start_command(message: Message):                                                                                #
    user_name = message.from_user.full_name                                                                                       #
    user_id = message.from_user.id                                                                                                #--->>> Команда старт
                                                                                                                                  #
    #################################################################################                                             #--->>> Команда старт
    logging.info(f'Этот пользователь {user_name}, с ИД {user_id} запустил бота')    # ------------> Временное логирование         #
    #################################################################################                                             #--->>> Команда старт
                                                                                                                                  #
    await message.answer(f'''Привет! {user_name}                                                                                  
это бот для помощи подбора лекарств,
Для того что бы начать поиск, выбери в меню, каким образом будем искать лекарство ?                                                
если тебе нужна помощь в работа бота, набери команду /help ''',                                                                    #
                         reply_markup = kb.main)                                                                                   #
    await message.bot.send_message(chat_id=662286809, text=f'Бот был запущен пользователем {user_name} ({user_id})')
####################################################################################################################################



#####################################################################################################################################################
# Это хендлер будет срабатывать на команду "/help"                                                                                                  #
@router.message(Command('help'))                                                                                                                    #
async def get_help(message: Message):                                                                                                               # ----> Команда ХЕЛП
    await message.answer('''Для работы бота, выберите в меню способ поиска (если меню не отображается нажмите квадрат справа внизу от клавиатуры)
После чего выберите каким способ искать лекарства, далее следуйте запросам бота''')                                                                 #
#####################################################################################################################################################


################################# Действия при нажатий инлайнкнопки НАЗАД #######################################################################
@router.message(F.text == 'Назад' )
async def Answer_if_no_menu_is_selected(message: Message, state: FSMContext):
    await state.clear()
    await message.answer(f'Для подобора аналогов, выберите в меню, каким образом бот будет искать')
################################# Действия при нажатий инлайнкнопки НАЗАД #######################################################################


################################################### Обработка реплай клавиатуры ФОТО #############################################################################  
@router.message(F.text == 'Поиск по фото')
async def load_photo(message: Message , state: FSMContext):
    await state.set_state(LEN_ANALOG.photo)
    await message.answer(text = 'Пришлите фотографию')
################################################### Обработка реплай клавиатуры ФОТО #############################################################################     

    
################################################### ПОЛУЧЕНИЕ ФОТО ОТ ПОЛЬЗОВАТЕЛЯ #############################################################################   
@router.message(LEN_ANALOG.photo, F.photo)
async def send_photo(message: Message, state: FSMContext):
    
    # Получить ID фотографии и информацию о файле
    image_id = message.photo[-1].file_id
    file_info = await message.bot.get_file(image_id)
    
    # Скачать фотографию и сохранить локально
    file_content = await message.bot.download_file(file_info.file_path)
    file_path = 'imgs/image.jpg'
    with open(file_path, 'wb') as f:
        file_content.seek(0)
        f.write(file_content.read())
        
    ###############################################################################    
    user_name = message.from_user.full_name                                       #
    user_id = message.from_user.id                                                # -----> Временное логирование
    logging.info(f'Этот пользователь {user_name}, с ИД {user_id} отправил фото')  #
    ###############################################################################
        
    # Получить предсказание и путь к файлу с результатами
    predicted_text, Probability,  prediction_image_path = get_prediction(model_yolo, file_path)
    if predicted_text:
        if len(predicted_text) > 1:
            predict_res = ', '.join(predicted_text)
            await message.reply(text= (f'На фото обнаруженно несколько лекарств: {predict_res}\nПо какому лекарству вы хотите найти аналоги ?'))
            await message.bot.send_photo(message.chat.id, photo= FSInputFile(prediction_image_path), reply_markup = await kb.found_medicines(predicted_text))
            
            await state.clear()
        else:
            await message.reply(f'Вижу c вероятностью {Probability[predicted_text[0]]} что это скорее всего {predicted_text[0]}')
            await message.bot.send_photo(message.chat.id, photo= FSInputFile(prediction_image_path))
            await state.clear()
            result_analog = analogue_search((predicted_text[0]), df=df, model=model_vector, num=40, similarity_threshold=0.7)
            await state.update_data(res = result_analog)
            await state.set_state(LEN_ANALOG.length_analog)
            if len(result_analog) != 0:                
                await message.answer (text = f'Найденых аналогов по этому лекарству: {len(result_analog)-1}\nСколько аналогов показать ?')
            else:
                await message.answer (text = f'Аналогов по этому лекарству не найдено, попробуйте загрузить другую фотошрафию, или попробуйте поискать другим методом')
                await state.set_state(LEN_ANALOG.photo)
            
    else:
        await message.reply(f'''Не удалось обнаружить объекты с достаточной уверенностью.
Попробуйте загрузить другую фотографию, или выберите в меню "назад" и попробуйте поискать другими методоми поиска''')
        await state.set_state(LEN_ANALOG.photo)                            
    # Удалить временные файлы
    os.remove(file_path)
    os.remove(prediction_image_path)    
    
####################Если пользователь прислал не фото ##################################
@router.message(LEN_ANALOG.photo, F.voice | F.text)
async def load_photo(message: Message , state: FSMContext):
    await state.set_state(LEN_ANALOG.photo)
    await message.answer(text = 'Что бы начать поиск мне нужна фотография!\nЕсли вы хотите изменить метод поиска, выберите в меню "Назад"')
####################Если пользователь прислал не фото ##################################
################################################### ПОЛУЧЕНИЕ ФОТО ОТ ПОЛЬЗОВАТЕЛЯ #############################################################################  
    
@router.callback_query(F.data == 'actice_voice')
async def actv_voice(query: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    if data['description_or_name']:
        start = 0
        end = int(data['length_medicals'])
    else:
        start = 1
        end = int(data["length_analog"]) + 1
    await query.answer('')
    await query.bot.send_message(query.message.chat.id, text= '<b>------Озвучиваю результат-----</b>', parse_mode= ParseMode.HTML)
    for i in range(start, end):
        ################ Синтез речи #########################
        path_text_yandex = 'voice/voice.ogg'                                                           
        _ = synthesize(text_voice(data['res'][i]),path_text_yandex )
        name_drugs = data['res'][i]['rus_name']
        await query.bot.send_message(query.message.chat.id, text= f'<b>Название лекарства:</b> {name_drugs}', parse_mode= ParseMode.HTML)                                          
        await query.bot.send_voice(query.message.chat.id, voice = FSInputFile(path_text_yandex))          
        os.remove(path_text_yandex)
        ################ Синтез речи ######################### 
    await state.clear()     
    
@router.callback_query(F.data == 'deactive_voice')
async def actv_voice(query: CallbackQuery, state: FSMContext):
    await query.answer('')
    await query.bot.send_message(query.message.chat.id, text = 'Если вы хотите попробовать другой метод поиска, выберите в меню, как будем искать')
    await state.clear() 

########################################################  ОБРАБОТКА ИНЛАЙН КЛАВИАТУРЫ ############################################################################
@router.callback_query()  # Мы не используем специфический фильтр здесь
async def handle_callbacks(query: CallbackQuery, state: FSMContext):
    data = query.data
    await query.answer(f'Вы выбрали опцию: {data}')
    result_analog = analogue_search(data, df=df, model=model_vector, num=100, similarity_threshold=0.7)
    await state.update_data(res = result_analog)
    await state.set_state(LEN_ANALOG.length_analog)
    await query.bot.send_message (query.message.chat.id,text= f'Найденых аналогов по этому лекарству: {len(result_analog)-1}\nСколько аналогов показать ?')
########################################################  ОБРАБОТКА ИНЛАЙН КЛАВИАТУРЫ ############################################################################


################################################### ОБРАБОТКА ПОИСКА ПО НАЗВАНИЮ #####################################################  
################################################### Обработка реплай клавиатуры ВВОД ПО НАЗВАНИЮ #####################################################
@router.message(F.text == 'Поиск аналогов по названию')
async def title_text_medicals(message: Message , state: FSMContext):
    await state.set_state(LEN_ANALOG.text_label)
    await message.answer(text = 'Напишите название лекарста\nИли скажите название лекарства в голосовом сообщений\n\nЕсли вы хотите изменить метод поиска,выберите в меню "Назад"')
################################################### Обработка реплай клавиатуры ВВОД ПО НАЗВАНИЮ #####################################################

@router.message(LEN_ANALOG.text_label, F.text)
async def analog_lable(message: Message, state: FSMContext):
    result_analog = analogue_search(message.text, df=df, model=model_vector, num=100, similarity_threshold=0.7)
    await state.update_data(res = result_analog)
    await state.set_state(LEN_ANALOG.length_analog)
    if len(result_analog) <= 0:
        await message.answer (text = f'К сожалению, аналогов по этому лекарству не найдено((((\nВы можете попробовать найти другое лекарство, или выберите в меню "Назад" для того что бы сменить способ поиска')
        await state.set_state(LEN_ANALOG.text_label)
    else:
        await message.reply(text= f'Найденых аналогов по этому лекарству: {len(result_analog)-1}\nСколько аналогов показать ?')
    
    
@router.message(LEN_ANALOG.text_label, F.voice)
async def input_voice_title(message: Message, state: FSMContext):
    voice_id = message.voice.file_id
    file_info = await message.bot.get_file(voice_id)
    
    # Скачать фотографию и сохранить локально
    file_content = await message.bot.download_file(file_info.file_path)
    file_path = 'voice/voice_user.wav'
    with open(file_path, 'wb') as f:
        file_content.seek(0)
        f.write(file_content.read())
    text = speech_recognition(file_path, voice_model)
    result_analog = analogue_search(text, df=df, model=model_vector, num=100, similarity_threshold=0.7)
    await state.update_data(res = result_analog)
    await state.set_state(LEN_ANALOG.length_analog)
    if len(result_analog) <= 0:
        await message.answer (text = f'К сожалению, аналогов по этому лекарству не найдено((((\nВы можете попробовать найти другое лекарство, или выберите в меню "Назад" для того что бы сменить способ поиска')
        await state.set_state(LEN_ANALOG.text_label)
    else:
        await message.reply(text= f'Найденых аналогов по этому лекарству: {len(result_analog)-1}\nСколько аналогов показать ?')
    os.remove(file_path)
    
#####################Ответ если пользователь прислал ФОТКУ ##############################################  
@router.message(LEN_ANALOG.text_label, F.photo)
async def input_voice_title(message: Message, state: FSMContext):
    await state.set_state(LEN_ANALOG.text_label)
    await message.answer(text= f'В этом методе я не могу обработать фото, напишите название лекарства или скажите его название в голосовом сообщений\nЕсли вы хотите изменить метод поиска, выберите в меню "Назад"')
##################### Ответ если пользователь прислал ФОТКУ ##############################################  
################################################### ОБРАБОТКА ПОИСКА ПО НАЗВАНИЮ #####################################################    
    
    
################################################### ОБРАБОТКА ПОИСКА ПО ОПИСАНИЮ #####################################################
################################################### Обработка реплай клавиатуры ВВОД ПО ОПИСАНИЮ #####################################################
@router.message(F.text == 'Поиск лекарств по описанию симптомов')
async def txt_pain(message: Message , state: FSMContext):
    await state.set_state(LEN_ANALOG.text_pain)
    await message.answer(text = 'Напишите симптомы по которым искать действия лекарств\nИли скажите в голосовом сообщений\n\nЕсли вы хотите изменить метод поиска,выберите в меню "Назад"')
    
################################################### Обработка реплай клавиатуры ВВОД ПО ОПИСАНИЮ #####################################################

@router.message(LEN_ANALOG.text_pain, F.text)
async def analog_lable(message: Message, state: FSMContext):
    search_results = search_drug_by_action(message.text, df, model_vector, num=100, similarity_threshold=0.7)
    await state.update_data(res = search_results)
    await state.set_state(LEN_ANALOG.length_medicals)
    if len(search_results) <= 0:
        await message.reply(text= f'К сожалению, лекарст по этому описанию не найдено, попробуйте оописать симптомы более подробно или нажмите "Назад" в меню, что бы посикать другим методом')
        await state.set_state(LEN_ANALOG.text_pain)
    else:        
        await message.reply(text= f'Сколько лекрств показать?\nНайдено: {len(search_results)}')
    
@router.message(LEN_ANALOG.text_pain, F.voice)
async def input_voice_description(message: Message, state: FSMContext):
    voice_id = message.voice.file_id
    file_info = await message.bot.get_file(voice_id)
    
    file_content = await message.bot.download_file(file_info.file_path)
    file_path = 'voice/voice_user.wav'
    with open(file_path, 'wb') as f:
        file_content.seek(0)
        f.write(file_content.read())
    text = speech_recognition(file_path, voice_model)
    search_results = search_drug_by_action(text, df, model_vector, num=100, similarity_threshold=0.7)
    await state.update_data(res = search_results)
    await state.set_state(LEN_ANALOG.length_medicals)
    if len(search_results) <= 0:
        await message.reply(text= f'К сожалению, лекарст по этому описанию не найдено, попробуйте оописать симптомы более подробно или нажмите "Назад" в меню, что бы посикать другим методом')
        await state.set_state(LEN_ANALOG.text_pain)
    else:        
        await message.reply(text= f'Сколько лекрств показать?\nНайдено: {len(search_results)}')
    os.remove(file_path)

#####################Ответ если пользователь прислал ФОТКУ ##############################################  
@router.message(LEN_ANALOG.text_pain, F.photo)
async def input_voice_title(message: Message, state: FSMContext):
    await state.set_state(LEN_ANALOG.text_pain)
    await message.answer(text= f'В этом методе я не могу обработать фото, напишите название лекарства или скажите его название в голосовом сообщений\nЕсли вы хотите изменить метод поиска, выберите в меню "Назад"')
##################### Ответ если пользователь прислал ФОТКУ ##############################################      

################################################### РЕЗУЛЬТАТ ПОИСКА ПО ОПИСАНИЮ ##################################################### 
@router.message(LEN_ANALOG.length_medicals)
async def Output_analogues(message:Message, state: FSMContext):
    await state.update_data(length_medicals = message.text)
    data = await state.get_data()
    try:
        if int(data['length_medicals']) > len(data['res']) :
            await state.set_state(LEN_ANALOG.length_medicals)
            await message.reply(text = f'Введенное число превышает количество найденых аналогов, введите корректную циферку\nЕсли вы хотите прервать поиск, выберите "Назад" в меню')
        elif int(data['length_medicals']) == 0:
            await state.set_state(LEN_ANALOG.length_medicals)
            await message.reply(text = f'Вы ввели 0, введите корректное число.\nЕсли вы хотите прервать поиск, выберите "назад" в меню ')            
        else:    
            for i in range(int(data["length_medicals"])):
                await message.answer(text= text(data['res'][i]), parse_mode= ParseMode.HTML)
            await message.answer(text='Озвучить текст?', reply_markup= kb.voic)
            await state.update_data(description_or_name = True)
            #await state.clear()
    except ValueError:
        await message.reply (text = f'По словам я не могу понять сколько лекарств показать(((\nВведите число что бы я понял\nЕсли вы хотите прервать поиск, выберите "назад" в меню')
        await state.set_state(LEN_ANALOG.length_medicals)
################################################### РЕЗУЛЬТАТ ПОИСКА ПО ОПИСАНИЮ ##################################################### 
        
################################################### РЕЗУЛЬТАТ ПОИСКА ПО ФОТО/НАЗВАНИЮ #####################################################

@router.message(LEN_ANALOG.length_analog)
async def Output_analogues(message:Message, state: FSMContext):
    await state.update_data(length_analog = message.text)
    data = await state.get_data()
    try:
        if int(data['length_analog']) > len(data['res']) -1:
            await state.set_state(LEN_ANALOG.length_analog)
            await message.reply(text = f'Введенное число превышает количество найденых аналогов, введите корректную циферку\nЕсли вы хотите прервать поиск, выберите "назад" в меню')
        elif int(data['length_analog']) == 0:
            await state.set_state(LEN_ANALOG.length_analog)
            await message.reply(text = f'Вы ввели 0, введите корректное число.\nЕсли вы хотите прервать поиск, выберите "назад" в меню ')            
        else:    
            for i in range(int(data["length_analog"]) + 1 ):
                if i == 0:
                    await message.answer(text= '<b>------Лекарство по которому идет поиск-----</b>', parse_mode= ParseMode.HTML)
                    await message.answer(text= text(data['res'][i]), parse_mode= ParseMode.HTML)
                elif i == 1:
                    await message.answer(text= '<b>-----------------Аналоги----------------</b>', parse_mode= ParseMode.HTML)
                    await message.answer(text= text(data['res'][i]), parse_mode= ParseMode.HTML)
                else:
                    await message.answer(text= text(data['res'][i]), parse_mode= ParseMode.HTML)
            await message.answer(text='Озвучить текст?', reply_markup= kb.voic)
            await state.update_data(description_or_name = False)
            #await state.clear()
    except ValueError:
        await message.reply (text = f'По словам я не могу понять сколько лекарств показать(((\nВведите число, чтобы я понял')
        await state.set_state(LEN_ANALOG.length_analog)
################################################### РЕЗУЛЬТАТ ПОИСКА ПО ФОТО/НАЗВАНИЮ #####################################################        

        
################################################### Обработка ВВОДА БЕЗ ВЫБОРА МЕНЮ #####################################################    
@router.message(F.text | F.photo | F.voice)
async def Answer_if_no_menu_is_selected(message: Message):
    await message.answer(f'Для подобора аналогов, выберите в меню, каким образом бот будет искать')
################################################### Обработка ВВОДА БЕЗ ВЫБОРА МЕНЮ ##################################################### 
