# Local inference for edge devices

class EdgeAnalyzer:
    def analyze(self, audio_path):
        # TODO: quantized Whisper + heuristic fallback
        return {'sentiment': 'neutral', 'confidence': 0.85, 'offline': True}