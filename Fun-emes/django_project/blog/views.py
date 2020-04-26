from django.shortcuts import render
from django.contrib.auth.forms import UserCreationForm

# Create your views here.
import os
import json
score_dict = None
elementary = None
highschool1 = None
highschool2 = None
highschool3 = None
hospital = None
score_dir = os.path.dirname((__file__))
path = os.path.join(score_dir, "elementary.json")
with open(path, 'r') as f:
    elementary = json.load(f)
    elementary = json.dumps(elementary)
path = os.path.join(score_dir, "highschool1.json")
with open(path, 'r') as f:
    highschool1 = json.load(f)
    highschool1 = json.dumps(highschool1)
path = os.path.join(score_dir, "highschool2.json")
with open(path, 'r') as f:
    highschool2 = json.load(f)
    highschool2 = json.dumps(highschool2)
path = os.path.join(score_dir, "highschool3.json")
with open(path, 'r') as f:
    highschool3 = json.load(f)
    highschool3 = json.dumps(highschool3)
path = os.path.join(score_dir, "hospital.json")
with open(path, 'r') as f:
    hospital = json.load(f)
    hospital = json.dumps(hospital)


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

def given_sentence(request):
    data = request.POST.get('record')
    sentence = request.POST.get('text')
    sentence_json = json.dumps(sentence)

    import speech_recognition as sr

    # get audio from the microphone
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        print("Speak:")
        # audio = r.listen(source)

    # try:
    # with open("microphone-results.wav", "wb") as f:
    #     f.write(audio.get_wav_data())  
        # from scipy.io.wavfile import read
        # import numpy as np
        # a = read("microphone-results.wav") 
        # audio_arr = np.array(a[1],dtype=float)
        # print(audio_arr)
        # text = "Hello how are you doing?"
        # audio = "/Users/garyxian/Desktop/JHU/AI_systems/CALL-proto/Fun-emes/django_project/microphone-results.wav"
        # score_output = Score_test.CalcScores(audio, text)
        # wav_id = PackZero(args.utt_id, 6)`
        # output = json.dumps(score_output, open("/Users/garyxian/Desktop/JHU/AI_systems/CALL-proto/Fun-emes/django_project/score_wav%s.json"%wav_id, "w", encoding="utf-8"))

        # output = " " + r.recognize_google(audio)
    # except sr.UnknownValueError:
    #     output = "Could not understand audio"
    # except sr.RequestError as e:
    #     output = "Could not request results; {0}".format(e)
    # data =output

    # from gtts import gTTS
    # import os
    # tts = gTTS(text='beep boop', lang='en')
    # tts.save("good.mp3")
    # os.system("mpg321 good.mp3")

    return render(request, 'blog/given_sentence.html', {'title': 'Say This Sentence', 'data':data, 'elementary': elementary, 'highschool1': highschool1, 'highschool2': highschool2, 'highschool3': highschool3, 'hospital' : hospital, 'sentence': sentence_json})

# def evaluate(request):
#     # score the wav file
#     # send the resulting JSON file to frontend

