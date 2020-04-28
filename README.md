# CALL-proto
A prototype works on computer assisted language learning (CALL)

## Document
doc/task0-assignment6.pdf \
doc/Frontend-Design-Assignment6.pdf

## Environment Install

`cd tools` \
`make KALDI=your_kaldi_dir`
and then install torch by conda

For usage: \
`. tools/env.sh`

Followings are my sample environment:
python version: 3.7.6 (default, Jan  8 2020, 19:59:22)  [GCC 7.3.0] \
pytorch version: 1.1.0 \
cupy version: 6.2.0 \
cuda version: 9010 \
cudnn version: 7501


# Django Application
A web application for Fun-emes language learning project.

## Main Directories and Functions
django_env - python virutal environment for package organization
django_project - main project folder

The project is made from tutorial so the frontend functionalities do not exactly match what we will have later on. Layouts of pages are temporary.

## Environment Setup and Installations
Runs on latest version of python (python3) and need pip to install needed packages so make sure that you have them on your machine

`cd Django/django_project` \
 Install python3 virtual environment package first if you dont already have it, and then use command `python3 -m venv env` \
`source env/bin/activate` to start the virtual environment, make sure to do this before running the server \

Necessary packages and libraries to install before running server for the first time if you do not already have them:
`pip install django`\
`pip install django-crispy-forms`\
`pip install scipy`\
`pip install SpeechRecognition`\
`brew install portaudio`\
`pip install pyaudio`\
`brew install ffmpeg`

Followings are my sample environment (as of Mar 2020) :
python version: 3.7.6 (default, Mar 2020)  [GCC 7.3.0] \
django version: 3.0.4 \
speechrecognition version: 2.1.3 \
portaudio version: V19.6.0 \
pyaudio version: 0.2.11

## Runnning the Development Server

Then inside Django/django_project folder, run \
`python manage.py runserver`

After development server started with no errors, open a browser and use address `localhost:8000/blog` to go to the main page

This route will also be changed later
Other routes to note are
`localhost:8000/blog/audio` - speech recognition page, press Record to start recording your audio and the red text on the screen will update
`localhost:8000/blog/about`- temporary page that has no function, will modify later
