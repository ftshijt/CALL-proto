from django.urls import path
from . import views

urlpatterns = [
        path('home/', views.home, name='home'),
        path('', views.launch, name='launch'),
        path('own_sentence/', views.own_sentence, name='own_sentence'),
        path('audio/', views.audio, name='audio'),
]
