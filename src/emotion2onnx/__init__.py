import onnxruntime as rt
import numpy as np

class EmotionToOnnx:
    def __init__(self, model_path: str):
        self.session = rt.InferenceSession(model_path)
        self.emotions = ["angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown"]

    def extract_emotions(self, audio: np.ndarray):        
        # Convert audio data to float32
        audio = audio.astype(np.float32)
        inputs = {self.session.get_inputs()[0].name: audio}
        outputs = self.session.run(None, inputs)
        emotion_scores = outputs[0]  # Adjust this index if necessary        

        # Check that the output shape matches the emotions list length
        scores = {}
        if len(emotion_scores) == len(self.emotions):
            for i, score in enumerate(emotion_scores):
                scores[self.emotions[i]] = score
        else:
            print(f"Error: Mismatch between emotion scores length ({len(emotion_scores)}) and emotions list length ({len(self.emotions)})")
        return scores
