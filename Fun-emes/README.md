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
`pip install SpeechRecognition`\
`brew install portaudio`\
`pip install pyaudio`\

Then inside Django/django_project folder, run \
`python manage.py runserver`\

After development server started with no errors, open a browser and use address `localhost:8000/blog` to go to the main page \

This route will also be changed later \
Other routes to note are
`localhost:8000/blog/audio` - speech recognition page, press Record to start recording your audio and the red text on the screen will update \
`localhost:8000/blog/about`- temporary page that has no function, will modify later \
