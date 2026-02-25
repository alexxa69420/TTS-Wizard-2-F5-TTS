import os
import re
import io
import time
import base64
import urllib.parse
import logging
import threading

from flask import Flask, request, Response, jsonify
from pydub import AudioSegment
from num2words import num2words
from pymystem3 import Mystem
from gradio_client import Client, handle_file

# --- КОНФИГУРАЦИЯ ---
FLASK_HOST = "127.0.0.1" 
FLASK_PORT = 8124

F5_TTS_URL = "http://192.168.0.112:7860/" 

REF_AUDIO_PATH = r"C:\\SOLO\\pickme.mp3"
REF_TEXT = "Привет, я твой новый голос, я буду жить на этой странице донатов"

CUSTOM_CKPT_PATH = "hf://hotstone228/F5-TTS-Russian/model_last.safetensors"
CUSTOM_VOCAB_PATH = "hf://hotstone228/F5-TTS-Russian/vocab.txt"
CUSTOM_MODEL_CFG = '{"dim": 1024, "depth": 22, "heads": 16, "ff_mult": 2, "text_dim": 512, "text_mask_padding": false, "conv_layers": 4, "pe_attn_head": 1}'

# --- Настройки логирования ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - (%(funcName)s) - %(message)s')
logger = logging.getLogger(__name__)

# --- Инициализация Mystem ---
try:
    mystem = Mystem(grammar_info=True, entire_input=False)
    logger.info("Mystem инициализирован успешно.")
    _ = mystem.analyze("тест") 
except Exception as e:
    logger.error(f"Не удалось инициализировать Mystem: {e}. Коррекция рода числительных будет отключена.")
    mystem = None

# --- Инициализация Gradio Client для F5-TTS ---
f5_client = None
try:
    logger.info(f"Подключение к F5-TTS по адресу {F5_TTS_URL}...")
    f5_client = Client(F5_TTS_URL)
    
    logger.info("Применение настроек кастомной модели F5-TTS...")
    model_setup_result = f5_client.predict(
        custom_ckpt_path=CUSTOM_CKPT_PATH,
        custom_vocab_path=CUSTOM_VOCAB_PATH,
        custom_model_cfg=CUSTOM_MODEL_CFG,
        api_name="/set_custom_model"
    )
    logger.info(f"Модель установлена. Ответ сервера: {model_setup_result}")
except Exception as e:
    logger.error(f"Ошибка подключения или настройки F5-TTS. Ошибка: {e}")


# --- Инициализация Flask ---
app = Flask(__name__)

# --- Функция коррекции числительных ---
def correct_numeral_gender_mystem(text):
    if not mystem:
        return text

    words = text.split(' ')
    corrected_words = list()
    i = 0
    
    while i < len(words):
        # Используем __getitem__ вместо квадратных скобок для защиты от фильтра чата
        word = words.__getitem__(i)
        corrected_word = word

        if word in ("один", "два"):
            if i + 1 < len(words):
                next_word = words.__getitem__(i + 1)
                cleaned_next_word = re.sub(r'\W+$', '', next_word).strip()

                if cleaned_next_word:
                    try:
                        analysis = mystem.analyze(cleaned_next_word)
                        if analysis:
                            first_analysis = analysis.__getitem__(0)
                            if 'analysis' in first_analysis and first_analysis.get('analysis'):
                                gr = first_analysis.get('analysis').__getitem__(0).get('gr')
                                if gr.startswith('S,'): 
                                    gender = None
                                    if 'жен' in gr:
                                        gender = 'femn'
                                    elif 'сред' in gr:
                                        gender = 'neut'
                                    elif 'муж' in gr:
                                        gender = 'masc'

                                    if gender:
                                        if word == "один":
                                            if gender == 'femn': corrected_word = "одна"
                                            elif gender == 'neut': corrected_word = "одно"
                                        elif word == "два":
                                            if gender == 'femn': corrected_word = "две"
                    except Exception as e:
                        logger.warning(f"Ошибка анализа Mystem для '{cleaned_next_word}': {e}")

        corrected_words.append(corrected_word)
        i += 1

    return ' '.join(corrected_words)


# --- Основной роут Flask для Voice Wizard ---
# Используем круглые кортежи вместо списков для методов
@app.route('/synthesize/', methods=("GET",))
@app.route('/synthesize/<path:text>', methods=("GET",))
def handle_synthesize_request(text=''):
    request_start_time = time.time()
    logger.info(f"Получен запрос от Voice Wizard ({request.remote_addr})")

    if not text:
        qs = request.query_string.decode('utf-8', errors='ignore')
        if qs:
            args = urllib.parse.parse_qs(qs)
            text_param_list = args.get('text')
            if text_param_list:
                text = text_param_list.__getitem__(0)
            else:
                text = qs
            
    if not text:
        return jsonify({"status": "error", "message": "Текст не предоставлен"}), 400
        
    try:
        decoded_text = urllib.parse.unquote(text).strip()
        if not decoded_text:
            return jsonify({"status": "error", "message": "Пустой текст"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": "Ошибка декодирования URL"}), 400

    try:
        def replace_with_words(match):
            number_str = match.group(0)
            try:
                return num2words(int(number_str), lang='ru')
            except ValueError:
                return number_str
                
        processed_text = re.sub(r"-?\d+", replace_with_words, decoded_text)
    except Exception as e:
        logger.error(f"Ошибка num2words: {e}")
        processed_text = decoded_text

    try:
        final_text = correct_numeral_gender_mystem(processed_text)
    except Exception as e:
        logger.error(f"Ошибка коррекции рода (Mystem): {e}")
        final_text = processed_text
        
    logger.info(f"Финальный текст для генерации: '{final_text}'")

    if not f5_client:
        return jsonify({"status": "error", "message": "Клиент F5-TTS не подключен"}), 503

    try:
        logger.info("Отправка запроса в F5-TTS...")
        result = f5_client.predict(
            ref_audio_input=handle_file(REF_AUDIO_PATH),
            ref_text_input=REF_TEXT,
            gen_text_input=final_text,
            remove_silence=False,
            randomize_seed=True,
            seed_input=0,
            cross_fade_duration_slider=0.15,
            nfe_slider=64,
            speed_slider=1,
            api_name="/basic_tts"
        )
        
        # Индекс 0 берем через метод, чтобы избежать вырезания скобок фильтром
        generated_audio_path = result.__getitem__(0)
        logger.info(f"Аудио успешно сгенерировано сервером.")

    except Exception as e:
        logger.error(f"Ошибка при обращении к F5-TTS API: {e}", exc_info=True)
        return jsonify({"status": "error", "message": f"Ошибка F5 API: {str(e)}"}), 500

    try:
        logger.info("Конвертация аудио в формат для Voice Wizard (Mono WAV)...")
        audio = AudioSegment.from_file(generated_audio_path)
        
        standard_audio = audio.set_frame_rate(44100).set_sample_width(2).set_channels(1)
        
        with io.BytesIO() as wav_io:
            standard_audio.export(wav_io, format="wav")
            audio_data = wav_io.getvalue()
            
        audio_base64_string = base64.b64encode(audio_data).decode("utf-8")
        
        total_time = time.time() - request_start_time
        logger.info(f"Успех! Общее время: {total_time:.2f} сек.")
        
        return Response(audio_base64_string, mimetype="text/plain")

    except Exception as e:
        logger.error(f"Ошибка конвертации аудио: {e}", exc_info=True)
        return jsonify({"status": "error", "message": "Ошибка конвертации аудио"}), 500


if __name__ == "__main__":
    logger.info("F5-TTS Voice Wizard Bridge Запущен!")
    app.run(host=FLASK_HOST, port=FLASK_PORT, threaded=True, use_reloader=False)