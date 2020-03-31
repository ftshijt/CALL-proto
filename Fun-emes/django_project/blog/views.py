from django.shortcuts import render

# Create your views here.
import os
import json
score_dict = None
score_dir = os.path.dirname((__file__))
rel_path = "score.json"
abs_file_path = os.path.join(score_dir, rel_path)
with open(abs_file_path, 'r') as f:
    score_dict = json.load(f)
    score_dict = json.dumps(score_dict)

sentence = "This is a sample sentence."
sentence_list = sentence.split()
sentence_json = json.dumps(sentence_list)

def launch(request):
    return render(request, 'blog/launch.html')

def home(request):
    return render(request, 'blog/home.html')
<<<<<<< HEAD
=======

>>>>>>> 1b77ab56325b118de66249bb4e2bbde767140155

def own_sentence(request):
    return render(request, 'blog/own_sentence.html', {'title': 'Use Your Own Sentence'})

def given_sentence(request):
    data = request.POST.get('record')
    sentence = request.POST.get('sentence')
    import speech_recognition as sr

    # get audio from the microphone
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak:")
        audio = r.listen(source)

    try:
        with open("microphone-results.wav", "wb") as f:
            f.write(audio.get_wav_data())
        from scipy.io.wavfile import read
        import numpy as np
        a = read("microphone-results.wav")
        audio_arr = np.array(a[1],dtype=float)
        print(audio_arr)

        output = " " + r.recognize_google(audio)
    except sr.UnknownValueError:
        output = "Could not understand audio"
    except sr.RequestError as e:
        output = "Could not request results; {0}".format(e)
    data =output

    from gtts import gTTS
    import os
    tts = gTTS(text='Good morning', lang='en')
    tts.save("good.mp3")
    os.system("mpg321 good.mp3")

    return render(request, 'blog/given_sentence.html', {'title': 'Say This Sentence', 'data':data, 'score':score_dict, 'sentence': sentence_json})

<<<<<<< HEAD

from django.contrib.auth.models import User
from django.http import JsonResponse


from django.contrib.auth.forms import UserCreationForm
from django.views.generic.edit import CreateView

class SignUpView(CreateView):
    template_name = 'blog/signup.html'
    form_class = UserCreationForm

def validate_username(request):
    username = request.GET.get('username', None)
    data = {
        'is_taken': User.objects.filter(username__iexact=username).exists()
    }
    return JsonResponse(data)
# def evaluate(request):
#     # score the wav file
#     # send the resulting JSON file to frontend

=======
# def audio(request):
#     data = request.POST.get('record')
#     import speech_recognition as sr

#     # get audio from the microphone
#     r = sr.Recognizer()
#     with sr.Microphone() as source:
#         print("Speak:")
#         audio = r.listen(source)

#     try:
#         print(dir(audio))
#         print(audio.sample_rate)
#         print(audio.sample_width)
#         with open("microphone-results.wav", "wb") as f:
#             f.write(audio.get_wav_data())
#         from scipy.io.wavfile import read
#         import numpy as np
#         a = read("microphone-results.wav")
#         audio_arr = np.array(a[1],dtype=float)
#         print(audio_arr)

#         output = " " + r.recognize_google(audio)
#     except sr.UnknownValueError:
#         output = "Could not understand audio"
#     except sr.RequestError as e:
#         output = "Could not request results; {0}".format(e)
#     data =output

#     return render(request,'blog/audio.html',{'data':data})
>>>>>>> 1b77ab56325b118de66249bb4e2bbde767140155
