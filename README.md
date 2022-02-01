# Face recongition project
## Description
I made this project in January 2022. The goal was to make a program that could differentiate a man from a woman in an image. First, I gathered portraits from both genders with the Python library ```BeautifulSoup``` to create the dataset that is used as model. Additionally I added a premade dataset to increase the accuracy of my program. Then I used the libaries ```TensforFlow```, ```cv2``` and ```sklearn``` to train the model and analyze the pictures.
I added a feature that compares the similiarities between two individuals and sends back a similiarity score. For this I used the library ```faceapp```.

## Install libraries
To run the program you need to make sure that all the libraries are imported. For that you can either open a terminal and type :
```
pip install BeautifulSoup, TensorFlow, cv2, sklearn, faceapp
```
or
```
pip install -r requirements.txt
```
Then you can enter in your terminal to train the model :
```
python main.py
```
and then you can run in your terminal :
```
python genderprediction.py
```
to run analyze an image.

