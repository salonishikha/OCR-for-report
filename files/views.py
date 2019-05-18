# Create your views here.
from urllib import request

from django.shortcuts import render
from django.urls import reverse
from django.views import static
from django.views.generic.edit import FormView
from django.http.response import HttpResponse, HttpResponseRedirect
from numpy.distutils.fcompiler import none

from files.forms import ProfileImageForm
from files.models import ProfileImage
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pytesseract
import shutil
import os
import random
try:
 from PIL import Image
except ImportError:
 import Image
from scipy.misc import imread, imresize
from keras.models import load_model
import pandas as pd
profile_image=none
class ProfileImageView(FormView):
    # template_name = 'profile_image_form.html'
    template_name = 'front.html'
    form_class = ProfileImageForm

    def form_valid(self, form):
        profile_image = ProfileImage(
            image=self.get_form_kwargs().get('files')['image'],
            excelfile= self.get_form_kwargs().get('files')['excelfile'],
        )
        profile_image.save()
        return HttpResponseRedirect(reverse('results'))

def ResultsView(request):
    im = cv2.imread('static/user_report.jpg')
    #im = open("static/user_report.jpg", "rb").read()
    # PreProcessing: i. Crop to appropriate size, ii. Thresholding iii.find contours
    print(im)
    im3 = im.copy()
    x1, y1, x2, y2 = 10, 760, 1190, 1590
    imc = im[y1:y2, x1:x2]  # imc = image cropped
    imcc = imc.copy()
    # imgplot = plt.imshow(imc)
    # plt.show()

    height = imc.shape[0]
    width = imc.shape[1]

    # print(height,width)
    gray = cv2.cvtColor(imc, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

    #################      Now finding Contours         ###################

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # find dimension of all the boxes containing digits
    pre_sorted = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            [x, y, w, h] = cv2.boundingRect(cnt)
            if (h > 50 and h < 500):
                pre_sorted.append((x, y, w, h))

    # Sort according to X-coordinate
    sortedx_numpy = np.array(sorted(pre_sorted, key=lambda x: x[0])).reshape(20, 10, 4)

    # Sort according to Y-coordinate
    sorted_required = []
    for i in range(1, 20, 2):
        sorted_required.append(sorted(sortedx_numpy[i], key=lambda x: x[1]))
    sorted_final = np.array(sorted_required).reshape(100, 4)  # convert list to numpy array
    model = load_model('/static/my_model.h5')
    string_list = []
    im23 = imc.copy()
    for a in sorted_final:
        x = a[0]
        y = a[1]
        w = a[2]
        h = a[3]
        digit1 = imc[y:y + h, x:x + w]
        gray1 = cv2.cvtColor(digit1, cv2.COLOR_BGR2GRAY)
        x = imresize(gray1, (28, 28))
        # convert to a 4D tensor to feed into our model
        x = x.reshape(1, 28, 28, 1)
        x = x.astype('float32')
        x /= 255
        out = model.predict(x)
        string_list.append(np.argmax(out))

    # print(string_list)
    # read from excel file and generate float list
    df = pd.read_excel("/static/record.xlsx")
    for rows in df.iterrows():
        my_list = rows
    new_list = [float(my_list[1][i]) for i in range(1, 100)]
    # eorder string list for matching with input excel list
    ocrlist = []
    for i in range(0, 100):
        if (string_list[i] not in range(1, 9)):
            string_list[i] = 0
    ocrlist2 = np.array(string_list).reshape(10, 10)
    ocrlist4 = ocrlist2.transpose()
    for i in range(0, 10):
        for j in range(0, 10):
            ocrlist.append(ocrlist4[i][j])
    # return a list of booleans- Boolean_list with matched vales True

    Boolean_list = [False] * 100
    for i in range(0, 99):
        if (ocrlist[i] == new_list[i]):
            # matched[i]=ocrlist[i]
            Boolean_list[i] = True

    # print("Total number of matches:")
    # print(np.count_nonzero(Boolean_list))
    # To highlight the wrong answers
    counter = 0
    imtest = cv2.imread('user_report.jpg')
    for bl in Boolean_list:
        if not (bl):
            x = sorted_final[counter][0]
            y = sorted_final[counter][1]
            w = sorted_final[counter][2]
            h = sorted_final[counter][3]
            cv2.rectangle(imtest, (x + x1, y + y1), (x + x1 + w, y + y1 + h), (0, 0, 255), 2, )
        counter += 1
    im = cv2.rectangle(imtest, (0, 20), (80, 100), (0, 0, 255), 2)
    im = cv2.putText(imtest, "Wrong answer", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), lineType=cv2.LINE_AA,
                     thickness=7)
    cv2.imwrite('static/wrong.png', imtest)  # write the image containing highlighted wrong answers
    return render(request, "Results.html")
