from django.shortcuts import render
from django.contrib.auth.forms import UserCreationForm

# Create your views here.
import os
import json
from gtts import gTTS
from pydub import AudioSegment
import speech_recognition as sr

# initialize Score object
# from callBackend import Score
# Score_test = Score(args.lexiconaddr, args.phonesaddr, args.sr, args.kaldi_workspace, args.utt_id)

def home(request):
	return render(request, 'blog/home.html')

def register(request):
	form = UserCreationForm()
	return render(request, 'users/register.html', {'form':form})

def own_sentence(request):
	start = request.POST.get('start')
	recording = request.POST.get('recording')
	sentence = request.POST.get('text')
	sentence_json = json.dumps(sentence)

	curr_path = os.path.dirname((__file__))

	if sentence != None and recording == "false":
		# converts sentence into mp3, play on website
		tts = gTTS(text=sentence, lang='en')
		tts.save(curr_path+"/static/audio/sentence.mp3")

	if start == None:
		score = None
		score = json.dumps(score)  

	else:
		# get audio from the microphone
		r = sr.Recognizer()
		with sr.Microphone() as source:
			print("Speak:")
			audio = r.listen(source)

		# write audio as .wav file to local 
		with open(curr_path+"/static/audio/user.wav","wb") as f:
			f.write(audio.get_wav_data())

		# convert wav to mp3 so web can play it
		user_audio = AudioSegment.from_wav(curr_path+"/static/audio/user.wav")
		user_audio.export(curr_path+"/static/audio/user.mp3", format="mp3")

		###
		score_dir = curr_path+"/data/score2.json"
		with open(score_dir) as f:
			score = json.load(f)
		score = json.dumps(score)
		###

		#Backend
		# lexiconaddr = '/home/nan/CALL-proto/res/lexicon.txt'
		# phonesaddr = '/home/nan/CALL-proto/res/phone_map.txt'
		# sr = 16000
		# kaldi_workspace = '/home/nan/CALL-proto/backend/asr_train'
		# utt_id = 1
		# audio = '/home/nan/CALL-proto/Fun-emes/django_project/blog/static/audio/user.wav'
		# text = sentence
		# Score_test = Score(lexiconaddr, phonesaddr, sr, kaldi_workspace, utt_id)
		# score_output = Score_test.CalcScores(audio, text)
		# print (score_output)
		# wav_id = PackZero(utt_id, 6)
		# json.dump(score_output, open("/home/nan/CALL-proto/Fun-emes/django_project/test_score_wav%s.json"%wav_id, "w", encoding="utf-8"))

	return render(request, 'blog/own_sentence.html', {'title': 'Use Your Own Sentence', 'sentence': sentence_json, 'score': score})

def profile(request):
	return render(request, 'blog/profile.html', {'title': 'Self Practice'})

def feedback(request):
	return render(request, 'blog/feedback.html')

def to_post(request):
	return render(request, 'blog/to_post.html')

def given_sentence(request):
	start = request.POST.get('start')
	clicks = request.POST.get('clicks')
	recording = request.POST.get('recording')
	IDs = request.POST.get('IDs')
	sentence = request.POST.get('text')
	sentence_json = json.dumps(sentence)

	curr_path = os.path.dirname((__file__))

	if sentence != None and recording == "false":
		# converts sentence into mp3, play on website
		tts = gTTS(text=sentence, lang='en')
		tts.save(curr_path+"/static/audio/sentence.mp3")

	if clicks == None:
		clicks = 0
		clicks = json.dumps(clicks)

	if start == None:
		score = None
		score = json.dumps(score)  

	else:
		# get audio from the microphone
		r = sr.Recognizer()
		with sr.Microphone() as source:
			# r.adjust_for_ambient_noise(source)
			print("Speak:")
			audio = r.listen(source)

		# write audio as .wav file to local 
		with open(curr_path+"/static/audio/user.wav","wb") as f:
			f.write(audio.get_wav_data())

		# convert wav to mp3 so web can play it
		user_audio = AudioSegment.from_wav(curr_path+"/static/audio/user.wav")
		user_audio.export(curr_path+"/static/audio/user.mp3", format="mp3")

		###
		score_dir = curr_path+"/data/score2.json"
		with open(score_dir) as f:
			score = json.load(f)
		score = json.dumps(score)
		###

		#Backend
		# lexiconaddr = '/home/nan/CALL-proto/res/lexicon.txt'
		# phonesaddr = '/home/nan/CALL-proto/res/phone_map.txt'
		# sr = 16000
		# kaldi_workspace = '/home/nan/CALL-proto/backend/asr_train'
		# utt_id = 1
		# audio = '/home/nan/CALL-proto/Fun-emes/django_project/blog/static/audio/user.wav'
		# text = sentence
		# Score_test = Score(lexiconaddr, phonesaddr, sr, kaldi_workspace, utt_id)
		# score_output = Score_test.CalcScores(audio, text)
		# print (score_output)
		# wav_id = PackZero(utt_id, 6)
		# json.dump(score_output, open("/home/nan/CALL-proto/Fun-emes/django_project/test_score_wav%s.json"%wav_id, "w", encoding="utf-8"))   

	return render(request, 'blog/given_sentence.html', {'title': 'Scenario Practice', 'score':score, 'sentence': sentence_json, 'clicks': clicks, 'IDs': IDs})


