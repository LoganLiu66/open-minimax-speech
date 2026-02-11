from minimaxspeech.modules.tokenizer.xtts_tokenizer import VoiceBpeTokenizer

class MiniMaxSpeechFrontend:
    def __init__(self, vocab_file: str):
        self.text_tokenizer = VoiceBpeTokenizer(vocab_file=vocab_file)

    def encode(self, text: str, lang: str='en'):
        return self.text_tokenizer.encode(text, lang=lang)

    def decode(self, tokens: list):
        return self.text_tokenizer.decode(tokens)
