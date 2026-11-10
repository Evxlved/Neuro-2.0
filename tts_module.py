import torch
import sounddevice as sd
from threading import Thread, Lock
import queue
import logging
import time
import os
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TTSManager:
    def __init__(self, language='ru', model_id='v5_5_ru', device='cpu',
                 speaker='baya', sample_rate=48000, volume=0.5):
        self.language = language
        self.model_id = model_id
        self.device = device
        self.speaker = speaker
        self.sample_rate = sample_rate
        # Добавляем volume с ограничением [0.0, 1.0]
        self.volume = max(0.0, min(1.0, volume))
        self.model = None
        self.model_lock = Lock()   # Silero не потокобезопасна — все вызовы apply_tts под этим локом
        self.audio_queue = queue.Queue(maxsize=50)  # Ограничение очереди
        self.playing = False
        self.lock = Lock()
        self._pending = 0          # сколько фраз сейчас синтезируется (для ожидания в main)
        self._load_model()
        self._start_player()

    def _load_model(self):
        logger.info("Загружаю модель Silero TTS...")
        try:
            cache_dir = os.path.join(torch.hub.get_dir(), 'snakers4_silero-models')
            if os.path.exists(cache_dir):
                self.model, _ = torch.hub.load(
                    repo_or_dir=cache_dir,
                    model='silero_tts',
                    language=self.language,
                    speaker=self.model_id,
                    device=self.device,
                    source='local'
                )
            else:
                self.model, _ = torch.hub.load(
                    repo_or_dir='snakers4/silero-models',
                    model='silero_tts',
                    language=self.language,
                    speaker=self.model_id,
                    device=self.device
                )
            logger.info("Модель TTS загружена.")
        except Exception as e:
            logger.error(f"Ошибка загрузки TTS: {e}")
            raise

    def _start_player(self):
        """Фоновый плеер с регулировкой громкости и конвертацией типа."""

        def player():
            while True:
                try:
                    audio = self.audio_queue.get(timeout=1)
                    if audio is None:
                        break

                    # 1. Конвертируем torch.Tensor → numpy.ndarray
                    if isinstance(audio, torch.Tensor):
                        audio = audio.cpu().numpy()

                    # 2. Убедимся, что тип — float32 (как ожидает sounddevice)
                    if audio.dtype != np.float32:
                        audio = audio.astype(np.float32)

                    # 3. Масштабируем громкость
                    audio_scaled = (audio * self.volume).astype(np.float32)

                    with self.lock:
                        self.playing = True
                        sd.play(audio_scaled, samplerate=self.sample_rate)
                        sd.wait()
                        self.playing = False

                except queue.Empty:
                    continue
                except Exception:
                    logger.exception("Ошибка воспроизведения")

        Thread(target=player, daemon=True).start()

    def speak(self, text: str):
        """Асинхронный синтез с контролем очереди."""
        if not text or self.audio_queue.full():
            return

        with self.lock:
            self._pending += 1

        def synthesize():
            try:
                with self.model_lock:   # модель дёргаем только по очереди, без гонок
                    audio = self.model.apply_tts(
                        text=text,
                        speaker=self.speaker,
                        sample_rate=self.sample_rate,
                        put_accent=True,
                        put_yo=True
                    )
                self.audio_queue.put(audio, block=False)
            except queue.Full:
                logger.warning("Очередь TTS переполнена. Пропускаем фрагмент.")
            except Exception:
                logger.exception("Ошибка синтеза")
            finally:
                with self.lock:
                    self._pending -= 1

        Thread(target=synthesize, daemon=True).start()

    def wait_idle(self, timeout: float | None = None) -> bool:
        """
        Ждёт, пока всё синтезированное прозвучит.
        True — всё отыграло; False — не дождались за timeout секунд.
        """
        deadline = time.time() + timeout if timeout else None
        while True:
            with self.lock:
                pending = self._pending
            if pending == 0 and not self.playing and self.audio_queue.empty():
                return True
            if deadline and time.time() > deadline:
                return False
            time.sleep(0.1)

    def stop(self):
        """Остановка плеера."""
        self.audio_queue.put(None)
        logger.info("TTS остановлен.")

    def set_volume(self, volume: float):
        """Установить громкость (0.0 … 1.0)."""
        self.volume = max(0.0, min(1.0, volume))
        logger.info(f"Громкость TTS установлена: {self.volume:.2f}")
