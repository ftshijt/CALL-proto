from enum import Enum
from django_enum_choices.fields import EnumChoiceField

class Occupations(Enum):
	ES = "Elementary School"
	MS = "Middle School"
	HS = "High School"
	CL = "College"
	EP = "Employed"
	UEP = "Unemployed"
	RE = "Retired"

class Scenarios(Enum):
	CS = "Classroom Student"
	CT = "Classroom teacher"
	WE = "Work employee"
	WM = "Work manager"
	RE = "Restaurant"
	HP = "Hospital"
	SO = "Social"

from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User
# Create your models here.
class Text(models.Model):
	occupation = EnumChoiceField(Occupations)
	scenario = EnumChoiceField(Scenarios)
	sentence = models.TextField()

	def __str__(self):
		return str(self.occupation) + '| ' + str(self.scenario) + '| ' + self.sentence
	
	class Meta:
		ordering = ['occupation']

class Score(models.Model):
	user_id = models.ForeignKey(User, on_delete=models.CASCADE)
	sentence_id = models.ForeignKey(Text, on_delete=models.CASCADE)
	score_date = models.DateTimeField(default=timezone.now)
	score = models.CharField(max_length=3)

class Post (models.Model):
    title = models.CharField(max_length=50)
    description = models.TextField()

    def __str__(self):
        return self.title



    
