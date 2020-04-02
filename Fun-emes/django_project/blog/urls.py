from django.urls import path
from django.conf.urls import url
from . import views

urlpatterns = [
		path('', views.launch, name='launch'),
        path('home/', views.home, name='home'),
        path('own_sentence/', views.own_sentence, name='own_sentence'),
        path('given_sentence/', views.given_sentence, name='given_sentence'),
]
