#import librairies
import requests 
import urllib.request
import os
from bs4 import BeautifulSoup 

#url
url = "C:/Users/ethan/Desktop/Face recognition project/woman"

#url request
r = requests.get(url)
soup = BeautifulSoup(r.content, 'html.parser')

#returns all the tags that matches the filter 
data = soup.find_all('img')

number = 0

#download images
for image in data:
    image_src = image['src']
    fullfilename = os.path.join("C:/Users/ethan/Desktop/Face recognition project/woman", "image "+str(number))
    urllib.request.urlretrieve(image_src,fullfilename )
    number+=11
    
