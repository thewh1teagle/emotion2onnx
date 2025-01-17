"""
wget https://github.com/thewh1teagle/emotion2onnx/releases/download/model-files/emotion2vec.onnx
uv run examples/usage.py
"""
from emotion2onnx import EmotionToOnnx
import soundfile as sf 

audio, sample_rate = sf.read('short.wav') # 16khz
model = EmotionToOnnx('emotion2vec.onnx')
scores = model.extract_emotions(audio)
print(scores)
