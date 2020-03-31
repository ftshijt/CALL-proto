from call_scoring.Score import Score
import os

file_root = "/home/nan/CALL-proto/Fun-emes/django_project"

new_file = os.path.join(file_root, "microphone-results.wav")

score_obj = Score("", "", 16000, "/home/nan/CALL-proto/backend/asr_train", 0)
score_obj.KaldiInfer(new_file)
