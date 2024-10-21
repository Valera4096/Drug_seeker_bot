import torch
from torchvision import transforms as T
from torchvision.io import read_image
from sentence_transformers import SentenceTransformer

def load_model_yolo():
    model = torch.hub.load(
        repo_or_dir = 'yolov5', # будем работать с локальной моделью 
        model = 'custom', # непредобученная
        path='training/best.pt', # путь к нашим весам
        source='local' # откуда берем модель – наша локальная
        )
    return model



def get_prediction(model, path: str) -> str:
    
    dictionary = {'Afrin':'Африн', 'Arbidol':'Арбидол', 'Canephron':'Канефрон', 'Cardiomagnyl':'Кардиомагнил' ,'Detralex': 'Детралекс',
                  'Edarbi':'Эдарби', 'Eliquis': 'Эликвис', 'Grammidin':'Граммидин', 'Heptral': 'Гептрал', 'Ingavirin': 'Ингавирин',
                  'Jess': 'Джес', 'Kagocel':'Кагоцел', 'Mexidol': 'Мексидол', 'Miramistin':'Мирамистин', 'Nimesil': 'Нимесил', 'Nurofen':'Нурофен',
                  'Pentalgin':'Пенталгин', 'Strepsils':'Стрепсилс', 'Teraflu':'Терафлю', 'Ursosan':'Урсосан', 'Xarelto':'Ксарелто',
                  'Sinupret':'Сунупрет', 'Rinonorm': 'Ринонорм', 'Faringosept': 'Фарингосепт', 'Pinosol':'Пиносол', 'Rhinostop':'Риностоп', 
                  'Anaferon':'Анаферон', 'Tantum_verde':'Тантум верде', 'Engystol': 'Энгистол', 'Rengalin': 'Ренгалин', 'Snoop':'Снуп', 
                  'Ibuklin' :'Ибуклин', 'AnviMax':'АнвиМакс', 'Tizin':'Тизин', 'Ambrobene':'Амбробене', 'Doctor_Mom':'Доктор Мом', 
                  'ACC':'АЦЦ,', 'Antigrippin': 'Антигриппин', 'Aquanasal':'Акваназаль' , 'Dolphin':'Долфин', 'Mezim':'Мезим' ,
                  'Microlax':'Микролакс', 'Polysorb':'Полисорб', 'Enterosgel':'Энтеросгель', 'Phosphogliv':'Фосфоглив', 'Gastal': 'Гастал'}
    
    model.conf = 0.20  # Устанавливаем порог
    # Image
    img = T.ToPILImage()(read_image(path))
    model.eval()
    # Inference
    results = model(img)
    # Рисуем ограничивающие рамки на картинке и сохраняем ее
    results.render()  # Apply the predictions on image
    rendered_img = T.ToPILImage()(results.ims[0])  # Используем 'ims' вместо 'imgs'
    rendered_img_path = "imgs/image_with_boxes.jpg"
    rendered_img.save(rendered_img_path)

    prediction_df = results.pandas().xyxy[0]
    predictions = {}
    res = []

    for index, row in prediction_df.iterrows():
        if dictionary[row['name']] in res:
            pass
        else:
            predictions[dictionary[row['name']]] =  round(row['confidence'],2)
            res.append(dictionary[row['name']])

    return res , predictions, rendered_img_path




def text(slovar):
    text= f'''
<b>Русскоязычное название:</b> {slovar['rus_name']}

<b>Активное вещество:</b>  {', '.join(slovar['active_substance'])}

<b>Дозировка:</b>  {', '.join(slovar['dose'])}
<b>Единицы измерения:</b>  {', '.join(slovar['dose_UoM'])}

<b>Клинико-фармакологическая группа:</b>  {slovar['pharmacological_group']}

<b>Фармако-терапевтическая группа:</b>  {slovar['therapeutic_group']}

<b>Показания активных веществ препарата:</b>  {slovar['rus_name']}: {slovar['ind_active_substance']}

{', '.join(slovar['manufacturer'])}

<b>Подробную информацию см. по ссылке:</b>  {slovar['url']}'''
    return text


def text_voice(slovar):
    text= f'''
Русскоязычное название: {slovar['rus_name']}

Активное вещество:  {', '.join(slovar['active_substance'])}

Клинико-фармакологическая группа:  {slovar['pharmacological_group']}

Фармако-терапевтическая группа:  {slovar['therapeutic_group']}

Показания активных веществ препарата:  {slovar['rus_name']}: {slovar['ind_active_substance']}'''
    return text


import whisper

def speech_recognition(path, speech_model):
    speech_model = speech_model
    result = speech_model.transcribe(path)

    return result['text']

# Загрузка модели
def load_voice_model():
    model = whisper.load_model('small', in_memory='True', device='cpu')
    return model
    
############################################ Озвучка результатов ##################################################################
################################################################################################################################  

from argparse import ArgumentParser

from speechkit import model_repository, configure_credentials, creds

# Аутентификация через API-ключ.
configure_credentials(
   yandex_credentials=creds.YandexCredentials(
      api_key='You API YANDEX'
   )
)

def synthesize(text, export_path):
   model = model_repository.synthesis_model()

   # Задайте настройки синтеза.
   model.voice = 'masha'
   model.role = 'good'
   model.speed = 1.1


   # Синтез речи и создание аудио с результатом.
   result = model.synthesize(text, raw_format=False)
   result.export(export_path, 'ogg')

############################################ Озвучка результатов ##################################################################
################################################################################################################################    


#########################################     Функций для работы векторизаций        ###########################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
#########################################     Функций для работы векторизаций        ###########################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
#########################################     Функций для работы векторизаций        ###########################################



# Импорт модулей
import re
import spacy
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.spatial.distance import cosine, euclidean
import faiss

device = torch.device('cpu')
def load_model_vectr():
    model = SentenceTransformer("intfloat/multilingual-e5-large").to(device=device)
    return model


def load_df():
    df = pd.read_csv("data/vidal_drugs.csv")
    embeddings_rus_name = np.load('embeddings/embeddings_rus_name_e5.npy')
    embeddings_eng_name = np.load('embeddings/embeddings_eng_name_e5.npy')
    embeddings_active_substance = np.load("embeddings/embeddings_active_substance_e5.npy")
    embeddings_ind_active_substance = np.load("embeddings/embeddings_ind_active_substance_e5.npy")
    df["rus_name_embeddings"] = list(embeddings_rus_name)
    df["eng_name_embeddings"] = list(embeddings_eng_name)
    df["active_substance_embeddings"] = list(embeddings_active_substance)
    df["ind_active_substance_embeddings"] = list(embeddings_ind_active_substance)
    initialize_faiss_indices(df)
    return df


# Глобальные переменные для хранения индексов FAISS
global_indices = {
    'rus_name': None,
    'eng_name': None,
    'active_substance': None,
    'ind_active_substance': None,
}


# Загрузка списка стоп-слов
stop_words = set(stopwords.words("russian"))

# Загрузка модели spaCy для русского языка
nlp = spacy.load("ru_core_news_lg")

def preprocess_function(text):
    # Приведение к нижнему регистру
    text = text.lower()
    # Замена дефисов на пробелы
    text = re.sub(r"-", " ", text)
    # Удаление знаков пунктуации
    text = re.sub(r"[^\w\s]", "", text)
    # # Удаление знаков пунктуации, кроме дефиса
    # text = re.sub(r"[^\w\s\-]", "", text)
    # Удаление чисел
    # text = re.sub(r"\d+", "", text)
    # Токенизация
    words = word_tokenize(text)
    # Удаление стоп-слов
    words = [word for word in words if word not in stop_words]
    # Лемматизация
    lemmatized = [token.lemma_ for token in nlp(" ".join(words))]
    # Объединение слов в строку
    text = " ".join(lemmatized)
    return text


def create_dict_from_series(series: pd.Series) -> dict:
    """Вспомогательная функция для создания словаря из строки pandas Series с заданной структурой."""
    return {
        'id': series.name,  # Используем индекс строки как уникальный идентификатор
        'rus_name': series.get('rus_name', ''),
        'rus_name': series.get('rus_name', ''),
        'eng_name': series.get('eng_name', ''),
        'active_substance': series.get('active_substance', '').split(', ') if series.get('active_substance', '') != 'Not found' else [],
        'dose': series.get('dose', '').split(', ') if series.get('dose', '') != 'Not found' else [],
        'dose_UoM': series.get('dose_UoM', '').split(', ') if series.get('dose_UoM', '') != 'Not found' else [],
        'volume': series.get('volume', ''),
        'vol_UoM': series.get('vol_UoM', ''),
        'auxiliary_substance': series.get('auxiliary_substance', ''),
        'pharmacological_group': series.get('pharmacological_group', ''),
        'therapeutic_group': series.get('therapeutic_group', ''),
        'pharmacological_action': series.get('pharmacological_action', ''),
        'ind_active_substance': series.get('ind_active_substance', ''),
        'manufacturer': series.get('manufacturer', '').split(', ') if series.get('manufacturer', '') != 'Not found' else [],
        'country': series.get('country', '').split(', ') if series.get('country', '') != 'Not found' else [],
        'active_license': series.get('active_license', False),
        'url': series.get('url', ''),
        'similarity': series.get('similarity', '')
    }


def build_faiss_index(embeddings):
    """Вспомогательная функция для построения индекса FAISS."""
    dimension = embeddings.shape[1]  # Получаем размерность векторов
    # Нормализация векторов
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatL2(dimension)  # Используем L2 расстояние для индекса
    index.add(embeddings)  # Добавляем эмбеддинги в индекс
    return index

def initialize_faiss_indices(df):
    """Строит и сохраняет FAISS индексы в глобальной переменной."""
    # Предполагаем, что df уже содержит необходимые векторные представления
    global_indices['rus_name'] = build_faiss_index(np.vstack(df["rus_name_embeddings"]))
    global_indices['eng_name'] = build_faiss_index(np.vstack(df["eng_name_embeddings"]))
    global_indices['active_substance'] = build_faiss_index(np.vstack(df["active_substance_embeddings"]))
    global_indices['ind_active_substance'] = build_faiss_index(np.vstack(df["ind_active_substance_embeddings"]))


# Функция поиска по названию (русском и английском) (используя функцию build_faiss_index)
def search_drug_by_name(name_query: str, df, model, num=5, similarity_threshold=0.7):
    name_query = (name_query + ' ') * 10 # Для увеличения веса слов в запросе пользователя (для лучшей семантики)
    name_query = preprocess_function(name_query)
    query_embedding = model.encode([name_query], normalize_embeddings=True)

    # query_embedding = transform_query(name_query, tfidf_vectorizer, svd, model)

    # Использование существующих индексов из памяти
    index_rus = global_indices['rus_name']
    index_eng = global_indices['eng_name']

    # Поиск в FAISS для русских и английских названий и объединение результатов
    D_rus, I_rus = index_rus.search(query_embedding, num)
    D_eng, I_eng = index_eng.search(query_embedding, num)
    
    # Соединяем индексы и расстояния из обоих поисков
    combined_indices = np.concatenate([I_rus.flatten(), I_eng.flatten()])
    combined_scores = np.concatenate([(1 - D_rus).flatten(), (1 - D_eng).flatten()])

    # Фильтруем результаты по порогу схожести и убираем дубликаты индексов
    filtered_indices_scores = list({idx: score for idx, score in zip(combined_indices, combined_scores) if score >= similarity_threshold}.items())

    # Сортировка результатов по схожести и выбор топ-N результатов
    filtered_indices_scores_sorted = sorted(filtered_indices_scores, key=lambda x: x[1], reverse=True)
    top_indices_scores = filtered_indices_scores_sorted[:num]  # Ограничиваем количество результатов до num

    # Если после фильтрации остались результаты, преобразуем в DataFrame
    if top_indices_scores:
        indices, scores = zip(*top_indices_scores)  # Распаковка данных

        # Создаем итоговый DataFrame
        results_df = df.iloc[list(indices)].copy()
        results_df['similarity'] = scores

        return results_df
    else:
        # Возвращаем пустой DataFrame, если подходящих результатов нет
        return pd.DataFrame()

# Функция поиска по активному веществу (используя функцию build_faiss_index)
def search_drug_by_active_substance(active_substance_query: str, most_similar_id, df, model, num=5, similarity_threshold=0.7):
    active_substance_query = (active_substance_query + ' ') * 10 # Для увеличения веса слов в запросе пользователя (для лучшей семантики)
    active_substance_query = preprocess_function(active_substance_query)
    query_embedding = model.encode([active_substance_query], normalize_embeddings=True)

    # query_embedding = transform_query(active_substance_query, tfidf_vectorizer, svd, model)

    # Использование существующих индексов из памяти
    index = global_indices['active_substance']
    D, I = index.search(query_embedding, num)  # D - расстояния, I - индексы ближайших элементов

    scores = 1 - D.flatten()

    # Создаём датафрейм для результатов
    results_df = pd.DataFrame()

    # Поиск и добавление исходного лекарства по ID в начало списка результатов
    if most_similar_id is not None and most_similar_id in df.index:
        original_drug_df = df.loc[[most_similar_id]]
        original_drug_df['similarity'] = 1.0  # Максимальная схожесть
        results_df = pd.concat([original_drug_df, results_df])

    # Фильтруем и добавляем результаты согласно порогу схожести и наличию лицензии
    for idx, score in zip(I.flatten(), scores):
        if score >= similarity_threshold and df.iloc[idx]['active_license'] and idx != most_similar_id:  # Исключаем исходное лекарство из повторного добавления
            df_row = df.iloc[[idx]].copy()
            df_row['similarity'] = score
            results_df = pd.concat([results_df, df_row])

    # Сортировка результатов по убыванию схожести (кроме первого, которое уже исходное лекарство)
    if len(results_df) > 1:
        first_row = results_df.iloc[:1]
        rest = results_df.iloc[1:].sort_values(by='similarity', ascending=False)
        results_df = pd.concat([first_row, rest])

    return results_df

# Функция поиска лекарств по действию лекарства (используя функцию build_faiss_index)
def search_drug_by_action(action_query: str, df, model, num=5, similarity_threshold=0.7):
    action_query = (action_query + ' ') * 10 # Для увеличения веса слов в запросе пользователя (для лучшей семантики)
    action_query = preprocess_function(action_query)
    query_embedding = model.encode([action_query], normalize_embeddings=True)

    # query_embedding = transform_query(action_query, tfidf_vectorizer, svd, model)
    
    # Использование существующих индексов из памяти
    index = global_indices['ind_active_substance']
    
    # Поиск в FAISS
    D, I = index.search(query_embedding, num)  # Поиск num ближайших соседей
    
    results_list = []
    for idx, score in zip(I.flatten(), 1 - D.flatten()):  # Преобразуем расстояния в сходства
        if score >= similarity_threshold:
            # Добавляем дополнительную проверку на активную лицензию
            if df.iloc[idx]['active_license']:
                series = df.iloc[idx]
                result_dict = create_dict_from_series(series)
                result_dict["similarity"] = score
                results_list.append(result_dict)

    return results_list


# Функция поиска аналогов по точному совпадению наименования без использования ИИ

def find_drug_analogues(exact_match: pd.Series, df: pd.DataFrame, num: int = 5) -> list:
    # Проверяем, что exact_match не пуст и содержит необходимые данные
    if exact_match.empty or 'active_substance' not in exact_match or 'dose' not in exact_match:
        return []  # Возвращаем пустой список, если данные отсутствуют или не полны

    # Строим словарь для первого совпадения
    exact_match_dict = create_dict_from_series(exact_match)

    # Помещаем исходное лекарство на первую позицию
    results = [exact_match_dict]

    # Получаем действующее вещество и дозу
    active_substance = exact_match["active_substance"]
    dose = exact_match["dose"]

    # Ищем аналоги по действующему веществу и дозе, не включая исходное лекарство
    analogues = df[
        (df["active_substance"] == active_substance)
        & (df["dose"] == dose)
        & (df["active_license"])
        & (df.index != exact_match.name)
    ].copy()

    
    # Добавляем аналоги до достижения указанного числа
    for _, row in analogues.iterrows():
        if len(results) >= num + 1:
            break

        analogue_dict = create_dict_from_series(row)

        results.append(analogue_dict)

    return results[: num + 1]  # Включаем исходное лекарство + num аналогов

# Функция поиска аналогов по точному совпадению наименования без использования ИИ

def convert_ai_drug_search_results_to_list(ai_results_df: pd.DataFrame) -> list:
    # Преобразует результаты поиска ИИ в список словарей для вывода
    results_list = []
    for _, row in ai_results_df.iterrows():
        # Используем вспомогательную функцию для каждой строки
        result_dict = create_dict_from_series(row)
        results_list.append(result_dict)
    return results_list

def analogue_search(user_query: str, df: pd.DataFrame, model, num: int = 5, similarity_threshold=0.7) -> list:
    user_query = user_query.strip().lower()

    exact_matches = df[(df['rus_name'].str.lower() == user_query) |
                       ((df['eng_name'].str.lower() == user_query) &
                       (df['active_license']))]
    
    if not exact_matches.empty:
        # print("Точное совпадение по наименованию найдено.")
        exact_match = exact_matches.iloc[0]
        analogues = find_drug_analogues(exact_match, df, num)
        if analogues and len(analogues) > 1:
            # print(f"Найдены аналоги по точному совпадению активного вещества: {len(analogues) - 1}")
            return analogues
        else:
            # print("Аналоги по точному совпадению активного вещества не найдены. Идет поиск по активному веществу с использованием ИИ.")
            active_substance = exact_match["active_substance"]
            exact_match_id = exact_match.name
            ai_results = search_drug_by_active_substance(active_substance, exact_match_id, df, model, num, similarity_threshold)
            # print(f"Найдено {len(ai_results)} аналогов по активному веществу с помощью ИИ.")
            return convert_ai_drug_search_results_to_list(ai_results)
    else:
        # print("Точное совпадение по наименованию не найдено. Идет поиск по наименованию с ИИ.")
        ai_name_results = search_drug_by_name(user_query, df, model, num, similarity_threshold)
        if not ai_name_results.empty:
            # print("Найдены совпадения по наименованию с ИИ.")
            most_similar = ai_name_results.iloc[0]
            analogues = find_drug_analogues(most_similar, df, num)
            # print(analogues)
            if analogues and len(analogues) > 1:
                # print(f"Найдены аналоги по совпадению наименования с ИИ и точному совпадению активного вещества: {len(analogues) - 1}")
                return analogues
            else:
                # print("Аналоги по совпадению наименования с ИИ и точному совпадению активного вещества не найдены. Идет поиск по активному веществу с ИИ.")
                active_substance = most_similar["active_substance"]
                most_similar_id = most_similar.name
                ai_results = search_drug_by_active_substance(active_substance, most_similar_id, df, model, num, similarity_threshold)
                # print(f"Найдено {len(ai_results)} аналогов по наименованию и активному веществу с ИИ.")
                return convert_ai_drug_search_results_to_list(ai_results)
        else:
            # print("Результаты поиска по наименованию с ИИ пусты.")
            return []
