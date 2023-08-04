import time

from elevenlabs import generate, set_api_key
import pygame
import nltk
from threading import Thread, Event, Lock
from queue import Queue
from completion import completion

import io

with open("elevenlabs_api_key.txt", "r") as f:
    set_api_key(f.read().strip())


class SpeechSynthesizer:

    def __init__(self, voice_id='f983VwDGfSWLHQit66A0', max_sentences_before_summarize=5):
        self.voice_id = voice_id
        self.max_sentences = max_sentences_before_summarize
        self.max_tokens_per_sentence = 50
        self.tts_queue: Queue[str] = Queue()
        self.synth_queue: Queue[str] = Queue()
        self.playback_queue: Queue[bytes] = Queue()
        self.playback_thread_event = Event()
        self.tts_thread_event = Event()
        self.tts_thread = Thread(target=self._tts_worker)
        self.playback_thread = Thread(target=self._playback_worker)
        self.tts_thread.start()
        self.playback_thread.start()
        self.lock = Lock()

        # Initialize the mixer
        pygame.mixer.init()

    def is_busy(self):
        """
        Python queues have atomic mutations, but not atomic reads.
        Therefore, we need to lock the queue before reading its size.
        :return:
        """
        with self.lock:
            return (self.tts_queue.qsize() + self.playback_queue.qsize() + self.synth_queue.qsize()) > 0 \
                   or pygame.mixer.music.get_busy()

    def tts_blocking(self, gpt_output):
        """
        Start TTS, blocking until all playback is finished.
        :param gpt_output: output from GPT
        :return:
        """
        self.start_tts(gpt_output)
        while self.is_busy():
            time.sleep(0.1)

    def start_tts(self, gpt_output):
        sentences = nltk.sent_tokenize(gpt_output)
        if len(sentences) > self.max_sentences:
            sentences = completion.summarize({
                "role": "user",
                "content": gpt_output
            }, prompt="describe what you have accomplished or your findings very briefly; "
                      "clarify that you can provide additional details if necessary.")
        for sentence in sentences:
            tokens = nltk.word_tokenize(sentence)
            if len(tokens) < self.max_tokens_per_sentence:
                self.tts_queue.put(sentence)

        # Start the TTS thread and playback thread.
        self.tts_thread_event.set()
        self.playback_thread_event.set()

    def stop_tts(self):
        """
        Stops TTS and playback immediately, cancel pending TTS requests and pending playback tracks.
        :return:
        """
        self.tts_thread_event.clear()
        self.playback_thread_event.clear()
        self.tts_queue.queue.clear()
        self.playback_queue.queue.clear()
        pygame.mixer.music.stop()

    def _playback_worker(self):
        while True:
            self.playback_thread_event.wait()
            if not self.playback_queue.empty():
                mp3_bytes = self.playback_queue.get()
                # Write the bytes to a BytesIO object
                buffer = io.BytesIO(mp3_bytes)
                # Load the MP3 data
                pygame.mixer.music.load(buffer)
                # Play the audio
                pygame.mixer.music.play()

                # Block while current audio is playing
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                    self.playback_thread_event.wait()

            time.sleep(0.1)

    def _tts_worker(self):
        while True:
            self.tts_thread_event.wait()
            while not self.tts_queue.empty():
                # Synchronize queue access and mutations.
                # This way, all tts tasks are accounted for by summing queues at all times.
                with self.lock:
                    text = self.tts_queue.get()
                    self.synth_queue.put(text)

                try:
                    tts_result = self._synthesize(text)
                    with self.lock:
                        self.synth_queue.get()
                        self.playback_queue.put(tts_result)
                except Exception as e:
                    # Remove item from synth queue if synthesis fails.
                    self.synth_queue.get()
                    print(e)

            time.sleep(0.1)

    def _synthesize(self, text) -> bytes:
        return generate(voice=self.voice_id, text=text, stream=False)


