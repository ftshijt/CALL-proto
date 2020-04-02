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

# initialize Score object
# from callBackend import Score
# Score_test = Score(args.lexiconaddr, args.phonesaddr, args.sr, args.kaldi_workspace, args.utt_id)

def launch(request):
    return render(request, 'blog/launch.html')

def home(request):
    return render(request, 'blog/home.html')

def own_sentence(request):
    return render(request, 'blog/own_sentence.html', {'title': 'Use Your Own Sentence'})

def given_sentence(request):
    data = request.POST.get('record')
    sentence = request.POST.get('text')
    count = request.POST.get('count')
    print(count)
    if count is None:
        print("count is ")
        count = 1
    count = json.dumps(count)
    print(count)

    print(sentence)
    sentence_json = json.dumps(sentence)
    import speech_recognition as sr

    # get audio from the microphone
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        print("Speak:")
        audio = r.listen(source)

    try:
        with open("microphone-results.wav", "wb") as f:
            f.write(audio.get_wav_data())  
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

        output = " " + r.recognize_google(audio)
    except sr.UnknownValueError:
        output = "Could not understand audio"
    except sr.RequestError as e:
        output = "Could not request results; {0}".format(e)
    data =output

    # from gtts import gTTS
    # import os
    # tts = gTTS(text='beep boop', lang='en')
    # tts.save("good.mp3")
    # os.system("mpg321 good.mp3")

    return render(request, 'blog/given_sentence.html', {'title': 'Say This Sentence', 'data':data, 'score':score_dict, 'count': count})

# def evaluate(request):
#     # score the wav file
#     # send the resulting JSON file to frontend

