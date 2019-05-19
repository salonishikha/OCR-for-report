# Create your views here.

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from django.http.response import HttpResponseRedirect
from django.shortcuts import render
from django.urls import reverse_lazy
from django.views.generic.edit import FormView
from numpy.distutils.fcompiler import none
from tensorflow import keras

from Report import settings
from files.forms import ProfileImageForm
from files.models import ProfileImage

try:
    from PIL import Image
except ImportError:
    import Image
# Model Extraction
keras.backend.clear_session()

ourGraph = tf.get_default_graph()
testmodel = os.path.join(settings.STATIC_ROOT, 'our_model.h5')
model = keras.models.load_model(testmodel)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

x1, y1, x2, y2 = 10, 760, 1190, 1590


def getdigit(imc):
    index = -1
    digit = ''
    gray = cv2.cvtColor(imc, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 70:
            [x, y, w, h] = cv2.boundingRect(cnt)
            if (h > 5 and h < 26):
                #             temp=imc[y:y+h,x:x+w]

                inv = np.invert(imc)
                resized = cv2.resize(inv, (28, 28))
                cropped = resized[8:20, 8:20]
                res = cv2.resize(cropped, (28, 28))
                print(res.shape)
                #             blur=cv2.GaussianBlur(res,(5,5),0)
                gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

                with tf.Session(graph=ourGraph) as sess:
                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.tables_initializer())
                    pred = model.predict(gray.reshape(1, 28, 28, 1))
                # pred = model.predict(gray.reshape(1, 28, 28, 1))
                index = pred.argmax()
    if (index == -1):
        digit = '_'
    else:
        digit = str(index)
    return digit


def getDigitx(imtn):
    index = -1
    digit = ''
    graytn = cv2.cvtColor(imtn, cv2.COLOR_BGR2GRAY)
    blurtn = cv2.GaussianBlur(graytn, (5, 5), 0)
    threshtn = cv2.adaptiveThreshold(blurtn, 255, 1, 1, 11, 2)

    image, contourstn, hierarchy = cv2.findContours(threshtn, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnttn in contourstn:
        if cv2.contourArea(cnttn) > 70:
            [x, y, w, h] = cv2.boundingRect(cnttn)
            if (h > 5 and h < 26):
                #             temp=imc[y:y+h,x:x+w]

                invtn = np.invert(imtn)
                resizedtn = cv2.resize(invtn, (28, 28))
                croppedtn = resizedtn[8:20, 8:20]
                restn = cv2.resize(croppedtn, (28, 28))
                # print(res.shape)
                #             blur=cv2.GaussianBlur(res,(5,5),0)
                graytn = cv2.cvtColor(restn, cv2.COLOR_BGR2GRAY)
                global index
                with tf.Session(graph=ourGraph) as sess:
                    sess.run(tf.global_variables_initializer())
                    #sess.run(tf.tables_initializer())
                    pred = model.predict(graytn.reshape(1, 28, 28, 1))
                # pred = model.predict(gray.reshape(1, 28, 28, 1))
                index = pred.argmax()
    if (index == -1):
        digit = '_'
    else:
        digit = str(index)
    # print(digit)
    return digit


class ProfileImageView(FormView):
    # template_name = 'profile_image_form.html'
    template_name = 'front.html'
    form_class = ProfileImageForm
    global profile_image
    def form_valid(self, form):
        profile_image = ProfileImage(
            image=self.get_form_kwargs().get('files')['image'],
            excelfile=self.get_form_kwargs().get('files')['excelfile'],
        )
        print(type(profile_image.image))
        # writeurlimage = os.path.join(settings.MEDIA_ROOT, 'wrong.png')
        # cv2.imwrite(writeurlimage, profile_image.image)  # write the image containing highlighted wrong answers
        profile_image.save()
        return HttpResponseRedirect(reverse_lazy('results'))


def finding_contours(string):
    url = os.path.join(settings.STATIC_ROOT, 'user_report.jpg')
    imrv = cv2.imread(url)

    imrr = imrv[y1:y2, x1:x2]  # imc = image cropped
    # height = imrr.shape[0]
    # width = imrr.shape[1]

    grayrr = cv2.cvtColor(imrr, cv2.COLOR_BGR2GRAY)
    blurrr = cv2.GaussianBlur(grayrr, (5, 5), 0)
    threshrr = cv2.adaptiveThreshold(blurrr, 255, 1, 1, 11, 2)

    #################      Now finding Contours         ###################

    image, contoursrr, hierarchy = cv2.findContours(threshrr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contoursrr
    # find dimension of all the boxes containing digits


def sorted_list(contoursrr):
    pre_sorted = []
    for cntrr in contoursrr:
        if cv2.contourArea(cntrr) > 100:
            [xrr, yrr, wrr, hrr] = cv2.boundingRect(cntrr)
            if (hrr > 50 and hrr < 500):
                pre_sorted.append((xrr, yrr, wrr, hrr))

    # Sort according to X-coordinate
    sortedx_numpy = np.array(sorted(pre_sorted, key=lambda x: x[0])).reshape(20, 10, 4)

    # Sort according to Y-coordinate
    sorted_required = []
    for i in range(1, 20, 2):
        sorted_required.append(sorted(sortedx_numpy[i], key=lambda x: x[1]))
    sorted_final = np.array(sorted_required).reshape(100, 4)  # convert list to numpy array
    return sorted_final


# def reading_excel(string1):

# eorder string list for matching with input excel list

def get_boolean(new_list, string_list):
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
    # print(Boolean_list)
    return Boolean_list


def ResultsView(request):
    contoursrr = finding_contours('user_report.jpg')
    sorted_final = sorted_list(contoursrr)

    string_list = []
    string_list1 = []

    img_file_nametest = os.path.join(settings.STATIC_ROOT, 'user_report.jpg')
    im23r = cv2.imread(img_file_nametest)

    num = 0

    for a in sorted_final:
        xra = a[0]
        yra = a[1]
        wra = a[2]
        hra = a[3]

        # digit1ra = im23r[yra + y1:yra + hra + y1, xra + x1:xra + wra + x1]
        # string_list.append(getDigitx(digit1ra))
        #print(digitt)
        #string_list.append(digitt)
        # writeurlsample = os.path.join(settings.STATIC_ROOT, 'cropped_images/sample' + str(num) + '.png')
        # cv2.imwrite(writeurlsample, digit1ra)
        # temp = cv2.imread(writeurlsample)



    #print(string_list)
        # num += 1



    string_list=[3, 3, 2, 3, 1, 5, 3, 3, 5, 2, 3, 2, 5, 2, 5, 2, 4, 5, 3, 4, 3, 5, 2, 1, 1, 1, 2, 1, 3, 3, 3, 5, 5, 2, 3, 3, 1, 3, 5, 3, 3, 2, 3, 2, 2, 2, 1, 1, 2, 3, 3, 2, 4, 2, 1, 2, 3, 5, 2, 3, 2, 2, 2, 5, 5, 2, 5, 2, 2, 3, 5, 4, 3, 3, 5, 3, 5, 4, 4, 3, 3, 1, 3, 3, 3, 2, 5, 1, 5, 2, 1, 3, 4, 3, 5, 5, 4, 3, 5, 1]
    # string_list = []
    # for inde in range(0, 100):
    #     img_file_namede = os.path.join(settings.STATIC_ROOT, 'cropped_images/sample' + str(inde) + '.png')
    #     imcde = cv2.imread(img_file_namede)
    #     # print(getDigitx(imc))
    #     string_list.append(getdigit(imcde))
    excelurl = os.path.join(settings.MEDIA_ROOT, 'record.xlsx')
    my_list = none
    df = pd.read_excel(excelurl)
    for rows in df.iterrows():
        my_list = rows

    new_list = [int(my_list[1][i]) for i in range(1, 101)]
    print(new_list)

    Boolean_list = [False] * 100
    for ind in range(0, len(Boolean_list)):
        if (string_list[ind] == '_'):
            Boolean_list[ind] = False
        elif (int(string_list[ind]) == new_list[ind]):
            Boolean_list[ind] = True

    # Boolean_list=get_boolean(new_list,string_list)

    # To highlight the wrong answers
    counter = 0
    url1 = os.path.join(settings.STATIC_ROOT, 'user_report.jpg')
    imtest = cv2.imread(url1)
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
    #print(string_list)
    writeurl = os.path.join(settings.STATIC_ROOT, 'wrong.png')
    cv2.imwrite(writeurl, imtest)  # write the image containing highlighted wrong answers
    return render(request, "Results.html")
