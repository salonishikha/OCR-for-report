# Create your models here.
from django.db import models

class ProfileImage(models.Model):
    image = models.FileField(upload_to='',default='no_name.png')
    excelfile = models.FileField(upload_to='',default='no_name.xlsx')
