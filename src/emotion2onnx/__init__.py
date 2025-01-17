import onnxruntime as rt
import numpy as np

class EmotionToOnnx:
    def __init__(self, model_path: str):
        self.session = rt.InferenceSession(model_path)
        self.emotions = ["angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown"]

    def extract_emotions(self, audio: np.ndarray):        
        # Ensure audio is a numpy array and convert to float32
        audio = np.asarray(audio, dtype=np.float32)
        audio = audio.reshape(1, -1)
        
        inputs = {self.session.get_inputs()[0].name: audio}
        outputs = self.session.run(None, inputs)
        emotion_scores = outputs[0]  # Adjust this index if necessary        

        # Average across all time steps (axis 1) to get a single vector of 768 features
        if len(emotion_scores.shape) == 3:  # If the output is (1, 98, 768)
            emotion_scores = np.mean(emotion_scores[0], axis=0)  # Average across the time steps (98)
        
        # Check the length of emotion_scores to match the emotions list length
        scores = {}
        if len(emotion_scores) == len(self.emotions):
            for i, score in enumerate(emotion_scores):
                scores[self.emotions[i]] = score
        elif len(emotion_scores) > len(self.emotions):
            for i in range(len(self.emotions)):
                score: np.ndarray = emotion_scores[i]
                scores[self.emotions[i]] = score.tolist()
        else:
            print(f"Error: Mismatch between emotion scores length ({len(emotion_scores)}) and emotions list length ({len(self.emotions)})")
        
        return scores