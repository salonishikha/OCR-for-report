'''
Created on Mar 10, 2015

@author: nishant.nawarkhede
'''
from django import forms

class ProfileImageForm(forms.Form):
    image = forms.FileField(label='Upload an Image')
    excelfile = forms.FileField(label='Upload excel file')
