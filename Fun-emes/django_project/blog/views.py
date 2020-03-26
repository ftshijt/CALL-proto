from django.shortcuts import render

# Create your views here.

posts = [
    {
        'author' : 'CoreyMS',
        'title' :'Blog Post 1',
        'content' : 'First post content',
        'date_posted': 'August 27, 2018'
    },
    {
        'author': 'Jane Doe',
        'title': 'Blog Post 2',
        'content': 'Second post content',
        'date_posted': 'August 28, 2018'
    }
]

def launch(request):
    return render(request, 'blog/launch.html')

def home(request):
    context = {
        'posts': posts
    }
    return render(request, 'blog/home.html', context)


def own_sentence(request):
    return render(request, 'blog/own_sentence.html', {'title': 'Make Your Own Sentence'})

def audio(request):
    data = request.POST.get('record')
    import speech_recognition as sr

    # get audio from the microphone
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak:")
        audio = r.listen(source)

    try:
        print(dir(audio))
        print(audio.sample_rate)   
        print(audio.sample_width) 
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

    return render(request,'blog/audio.html',{'data':data})
