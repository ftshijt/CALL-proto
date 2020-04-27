from django.shortcuts import render
from django.contrib.auth.forms import UserCreationForm
from django.db.models import F   

# Create your views here.
import os
import json

scores = []
json_files = ["score1.json", "score2.json", "score3.json", "score4.json", "score5.json"]
score_dir = os.path.dirname((__file__)) + "/data/"
print(score_dir)
for file in json_files:
    path = os.path.join(score_dir, file)
    with open(path, 'r') as f:
        scores.append(json.load(f))
        
scores = json.dumps(scores)

tests = []
test_files = ["test1.json", "test2.json", "test3.json", "test4.json", "test5.json"]
score_dir = os.path.dirname((__file__)) + "/data/"
for file in test_files:
    path = os.path.join(score_dir, file)
    with open(path, 'r') as f:
        tests.append(json.load(f))

tests = json.dumps(tests)

clicks = "0"
# initialize Score object
# from callBackend import Score
# Score_test = Score(args.lexiconaddr, args.phonesaddr, args.sr, args.kaldi_workspace, args.utt_id)

def home(request):
    return render(request, 'blog/home.html')

def register(request):
    form = UserCreationForm()
    return render(request, 'users/register.html', {'form':form})

def own_sentence(request):
    return render(request, 'blog/own_sentence.html', {'title': 'Use Your Own Sentence'})

def profile(request):
    return render(request, 'blog/profile.html')

def feedback(request):
    return render(request, 'blog/feedback.html')

def to_post(request):
    return render(request, 'blog/to_post.html')

def given_sentence(request):
    data = request.POST.get('record')
    clicks = request.POST.get('clicks')
    ids = request.POST.get('IDs')
    sentence = request.POST.get('text')
    sentence_json = json.dumps(sentence)

    if clicks is None:
        clicks = 0
    clicks = json.dumps(clicks)
    print(clicks)

    import speech_recognition as sr
    # get audio from the microphone
    r = sr.Recognizer()
    with sr.Microphone() as source:
        # r.adjust_for_ambient_noise(source)
        print("Speak:")
        audio = r.listen(source)

    try:
        with open(os.path.dirname((__file__))+"/static/audio/user.wav","wb") as f:
            f.write(audio.get_wav_data())

        from pydub import AudioSegment
        wavfile = os.path.dirname((__file__))+"/static/audio/user.wav"
        user_audio = AudioSegment.from_wav(wavfile)
        mp3file = os.path.dirname((__file__))+"/static/audio/user.mp3"
        user_audio.export(mp3file, format="mp3")

        # from scipy.io.wavfile import read
        # import numpy as np
        # a = read("microphone-results.wav") 
        # audio_arr = np.array(a[1],dtype=float)
        # print(audio_arr)
        # text = "Hello how are you doing?"
        # audio = "/Users/garyxian/Desktop/JHU/AI_systems/CALL-proto/Fun-emes/django_project/blog/static/audio/user.wav"
        # score_output = Score_test.CalcScores(audio, text)
        # wav_id = PackZero(args.utt_id, 6)`
        # output = json.dumps(score_output, open("/Users/garyxian/Desktop/JHU/AI_systems/CALL-proto/Fun-emes/django_project/score_wav%s.json"%wav_id, "w", encoding="utf-8"))

        output = " " + r.recognize_google(audio)
    except sr.UnknownValueError:
        output = "Could not understand audio"
    except sr.RequestError as e:
        output = "Could not request results; {0}".format(e)
    data =output

    from gtts import gTTS
    # converts sentence into mp3, play on website
    sentence = "I have to study for my AP Calculus test."
    tts = gTTS(text=sentence, lang='en')
    tts.save(os.path.dirname((__file__))+"/static/audio/sentence.mp3")

    return render(request, 'blog/given_sentence.html', {'title': 'Say This Sentence', 'data':data, 'scores':scores, 'tests': tests, 'sentence': sentence_json, 'clicks': clicks, 'IDs': ids})

# def evaluate(request):
#     # score the wav file
#     # send the resulting JSON file to frontend

