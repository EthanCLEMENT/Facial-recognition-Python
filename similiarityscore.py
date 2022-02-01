from __future__ import print_function, unicode_literals

import json

from facepplib import FacePP, exceptions

face_detection=""
faceset_initialize=""
face_search=""
face_landmarks=""
dense_facial_landmarks=""
face_attributes=""
beauty_score_and_emotion_recognition=""

def face_comparing(app):

    img_url1 = "https://i.postimg.cc/J4CrR1kf/test3.jpg"
    img_url2 = "https://i.postimg.cc/J4CrR1kf/test3.jpg"

    cmp_ = app.compare.get(image_url1=img_url1,image_url2=img_url2)

    print('image1', '=', cmp_.image1)
    print('image2', '=', cmp_.image2)

    print('thresholds', '=', json.dumps(cmp_.thresholds, indent=4))
    print('confidence', '=', cmp_.confidence)

if __name__ == '__main__':

    api_key ='xQLsTmMyqp1L2MIt7M3l0h-cQiy0Dwhl'
    api_secret ='TyBSGw8NBEP9Tbhv_JbQM18mIlorY6-D'

    try:
        app_ = FacePP(api_key=api_key, api_secret=api_secret)
        funcs = [
            face_detection,
            face_comparing,
            faceset_initialize,
            face_search,
            face_landmarks,
            dense_facial_landmarks,
            face_attributes,
            beauty_score_and_emotion_recognition
        ]
        face_comparing(app_)


    except exceptions.BaseFacePPError as e:
        print('Error:', e)