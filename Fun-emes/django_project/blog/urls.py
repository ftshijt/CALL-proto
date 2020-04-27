from django.urls import path
from django.conf.urls import url
from . import views

urlpatterns = [
        path('', views.home, name='home'),
        path('own_sentence/', views.own_sentence, name='own_sentence'),
        path('given_sentence/', views.given_sentence, name='given_sentence'),
        path('profile/', views.profile, name='profile'),
        path('feedback/', views.feedback, name='feedback'),
        path('to_post/', views.to_post, name='to_post'),
]
