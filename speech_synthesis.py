import time
import pygame
import nltk
import io

from elevenlabs import generate, set_api_key
from typing import Optional
from threading import Thread, Event, Lock
from queue import Queue
from chat_completion_interface import completion


nltk.download('punkt')

with open("elevenlabs_api_key.txt", "r") as f:
    set_api_key(f.read().strip())


class SpeechSynthesizer:

    def __init__(self, voice_id='f983VwDGfSWLHQit66A0', max_sentences=5, min_synth_tokens=10):
        """
        :param voice_id: ID of the voice.
        :param max_sentences: max sentences before summarization
        :param min_synth_tokens: min tokens aggregated before sending to synthesis
        """
        self.voice_id = voice_id
        self.max_sentences = max_sentences
        self.min_synth_tokens = min_synth_tokens
        self.max_tokens_per_sentence = 50
        self.tts_queue: Queue[str] = Queue()
        self.synth_queue: Queue[str] = Queue()
        self.playback_queue: Queue[bytes] = Queue()
        self.playback_thread_event = Event()
        self.tts_thread_event = Event()
        self.tts_thread: Optional[Thread] = None
        self.playback_thread: Optional[Thread] = None
        self.stream_buffer = ""
        self.lock = Lock()

    def init(self):
        # Initialize the mixer
        pygame.mixer.init()
        # Initialize threads
        self.tts_thread = Thread(target=self._tts_worker)
        self.tts_thread.daemon = True
        self.playback_thread = Thread(target=self._playback_worker)
        self.playback_thread.daemon = True
        self.tts_thread.start()
        self.playback_thread.start()

    def is_busy(self):
        """
        Python queues have atomic mutations, but not atomic reads.
        Therefore, we need to lock the queue before reading its size.
        :return:
        """
        with self.lock:
            return (self.tts_queue.qsize() + self.playback_queue.qsize() + self.synth_queue.qsize()) > 0 \
                   or pygame.mixer.music.get_busy()

    def wait_for_completion(self):
        """
        Block until all TTS and playback is finished.
        :return:
        """
        while self.is_busy():
            time.sleep(0.1)

    def stream_tts(self, chunk: str | None):
        """
        Stream TTS as chunks of text come in.
        :param chunk: chunk from GPT response stream, None if end of stream
        :return:
        """
        if not chunk:
            # End of stream
            if len(self.stream_buffer) > 0:
                self.tts_thread_event.set()
                self.playback_thread_event.set()
                self.tts_queue.put(self.stream_buffer)
                self.stream_buffer = ""
            return
        self.stream_buffer += chunk
        sentences = nltk.sent_tokenize(self.stream_buffer)
        tokens = nltk.word_tokenize(self.stream_buffer)

        # When at least one sentence is complete / too long, start streaming.
        if len(sentences) > 1 or len(tokens) > self.max_tokens_per_sentence:
            sentence_tokens = nltk.word_tokenize(sentences[0])
            if len(sentence_tokens) > self.max_tokens_per_sentence:
                incomplete_sent = " ".join(tokens[:self.max_tokens_per_sentence])
                self.tts_queue.put(incomplete_sent)
                remaining = " ".join(tokens[self.max_tokens_per_sentence:])
                self.stream_buffer = remaining
            else:
                self.tts_queue.put(sentences[0])
                self.stream_buffer = " ".join(sentences[1:])

            # Begin streaming TTS if not already started.
            self.tts_thread_event.set()
            self.playback_thread_event.set()

    def start_tts(self, gpt_output):
        sentences = nltk.sent_tokenize(gpt_output)
        if len(sentences) > self.max_sentences:
            summary = completion.summarize([{
                "role": "user",
                "content": gpt_output
            }], prompt="describe what you have accomplished or your findings very briefly; "
                      "clarify that you can provide additional details if necessary.")
            print("Summarized for TTS: " + summary)
            sentences = nltk.sent_tokenize(summary)
        buffer = ""
        for sentence in sentences:
            tokens = nltk.word_tokenize(sentence)
            if len(tokens) < self.max_tokens_per_sentence:
                buffer += sentence + " "
                if len(buffer) > self.min_synth_tokens:
                    self.tts_queue.put(buffer)
                    buffer = ""
            else:
                # Todo: process sentences that are too long.
                pass

        self.tts_queue.put(buffer)

        # Start the TTS thread and playback thread.
        self.tts_thread_event.set()
        self.playback_thread_event.set()

    def stop_tts(self):
        """
        Stops TTS and playback immediately, cancel pending TTS requests and pending playback tracks.
        :return:
        """
        self.tts_queue.queue.clear()
        self.synth_queue.queue.clear()
        self.playback_queue.queue.clear()
        self.tts_thread_event.clear()
        self.playback_thread_event.clear()
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


