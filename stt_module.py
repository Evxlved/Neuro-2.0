import os
import json
from typing import Optional, Callable
import queue
import threading
import logging
import re
import time

try:
    import sounddevice as sd
except ImportError:
    raise ImportError("pip install sounddevice")

try:
    from vosk import Model, KaldiRecognizer
except ImportError:
    raise ImportError("pip install vosk")

logger = logging.getLogger(__name__)


class STTManager:
    def __init__(self, model_path: str, sample_rate: int = 16000,
                 silence_timeout: float = 1.6):
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.silence_timeout = silence_timeout   # сколько секунд тишины = конец фразы
        self.model = None
        self.recognizer = None
        self._load_model()

        self.text_queue = queue.Queue(maxsize=2)   # очередь ГОТОВЫХ фраз
        self.is_running = False
        self.listener_thread = None
        self._reset_event = threading.Event()      # запрос на сброс распознавателя

        # накопитель фразы: куски до паузы склеиваем, отдаём после тишины
        self._phrase_buffer: list[str] = []
        self._last_phrase_time = 0.0

    def _load_model(self):
        logger.info(f"Загружаю модель STT из {self.model_path}...")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Модель не найдена: {self.model_path}")
        self.model = Model(self.model_path)
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
        logger.info("Модель STT загружена.")

    def start_listening(self, callback: Optional[Callable[[str], None]] = None):
        """
        Запускает фоновое прослушивание микрофона.
        callback — функция для обработки распознанного текста (опционально).
        """
        if self.is_running:
            return

        self.is_running = True

        def listener():
            while self.is_running:               # внешний цикл: переживаем ошибки аудиопотока
                try:
                    with sd.InputStream(
                        samplerate=self.sample_rate,
                        channels=1,
                        dtype='int16',
                        blocksize=2048
                    ) as stream:
                        logger.info("STT: прослушивание началось...")
                        while self.is_running:
                            # сброс распознавателя выполняем здесь же, в потоке слушателя
                            if self._reset_event.is_set():
                                self._phrase_buffer.clear()
                                try:
                                    self.recognizer.Reset()
                                except Exception:
                                    logger.exception("Не удалось сбросить KaldiRecognizer")
                                self._reset_event.clear()

                            data, _ = stream.read(2048)   # ~128 мс аудио при 16 кГц
                            data_bytes = data.tobytes()

                            if self.recognizer.AcceptWaveform(data_bytes):
                                result = json.loads(self.recognizer.Result())
                                text = result.get("text", "").strip()

                                if (text and
                                    len(text) >= 3 and
                                    not re.match(r'^(э|а|м|хм|ээ|аа|ммм|\s)+$', text, re.IGNORECASE)):
                                    logger.debug(f"STT фрагмент: '{text}'")
                                    self._phrase_buffer.append(text)
                                    self._last_phrase_time = time.time()

                            # тишина дольше silence_timeout — фраза закончена
                            if (self._phrase_buffer and
                                    time.time() - self._last_phrase_time >= self.silence_timeout):
                                full_phrase = " ".join(self._phrase_buffer)
                                self._phrase_buffer.clear()

                                try:
                                    self.text_queue.put(full_phrase, block=False)
                                except queue.Full:
                                    self.text_queue.get()
                                    self.text_queue.put(full_phrase, block=False)

                except Exception:
                    if self.is_running:
                        logger.exception("Ошибка в потоке STT")
                        time.sleep(2)        # пауза, затем внешний while пересоздаст аудиопоток

        self.listener_thread = threading.Thread(target=listener, daemon=True)
        self.listener_thread.start()

    def _restart_listener(self):
        """Оставлен для совместимости: перезапуск теперь живёт внутри listener."""
        pass

    def stop_listening(self):
        """Останавливает прослушивание."""
        self.is_running = False
        if self.listener_thread:
            self.listener_thread.join(timeout=1.0)
        logger.info("STT: прослушивание остановлено.")

    def reset(self):
        """Сброс распознавателя: флаг уходит слушателю, недосказанный буфер и очередь чистим."""
        self._phrase_buffer.clear()
        while True:
            try:
                self.text_queue.get_nowait()
            except queue.Empty:
                break
        self._reset_event.set()

    def get_text(self) -> Optional[str]:
        """Получает следующую ГОТОВУЮ фразу из очереди (None, если пусто)."""
        try:
            return self.text_queue.get(block=False)
        except queue.Empty:
            return None
