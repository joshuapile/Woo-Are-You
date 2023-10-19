from bs4 import BeautifulSoup
from urllib.request import urlopen
import requests
from PIL import Image
from pathlib import Path
import cv2
import urllib, json,urllib.request
import numpy as np
import urllib
from sklearn.model_selection import train_test_split
from populatetrainandtest import populatetrainandtest


# url_to_image
#
# Parameters: 
#
# url - String object which holds a link to an images
#
# Returns:
# 
# Returns an image object which corresponds with the image in the link

def url_to_image(url):
	resp = urllib.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	return image

# parsehtml
#
# Parameters: 
#
# link - A string of the website which html will be extracted
#
# headers - The headers that will be used when doing a requesting to the previously
#           address link
#
# Returns:
# 
# Returns a BeautifulSoup object containing the html of the link

def parsehtml(link,header={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}):
        response = requests.get(link,headers=header)
        if response.status_code != 200:
            print("Error fetching page")
            exit()
        else:
            content = response.content
        return BeautifulSoup(response.content, 'html.parser')

# The class is used to populate the test and train datasets from images of species
# from the avibase website in the New York City Area

class birddata:
        
    def __init__(self,contentpath):
         self=self  
         self.contentpath=contentpath

    def collect(self):
        
        # The html from the avibase is extracted
        birdhtml = parsehtml("https://avibase.bsc-eoc.org/checklist.jsp?lang=EN&p2=1&list=clements&synlang=&region=USnyne&version=text&lifelist=&highlight=0")
        
        
        for i,birddata in enumerate(birdhtml.find_all("tr", class_="highlight1")[318:],1):
            
            birdnameandlink = birddata.a
            birdname = birdnameandlink.find("i").text
            
            sublink = birdnameandlink["href"]
            summarylink= "https://avibase.bsc-eoc.org/"+ sublink +"&sec=summary"
            picturelink= "https://avibase.bsc-eoc.org/"+ sublink +"&sec=flickr"

            picturedata = parsehtml(picturelink).find_all("div",class_="divflickr")
            unfilteredpicturepngs = [ pictureelement.find("img")["src"] for pictureelement in picturedata]
            
            if len(unfilteredpicturepngs)<10: #If there isnt enough images for us to successfully train on then the iteration is skipped
                 continue

            trainpictures, testpictures =  train_test_split(unfilteredpicturepngs, test_size =.20,shuffle=True) 
            DirectoryPopulator = populatetrainandtest()

            #The pictures extracted from the html are split into a 80% and 20% set for train and test and then placed into the
            #proper directory
            DirectoryPopulator.populate("train",trainpictures,birdname,self.contentpath)
            DirectoryPopulator.populate("test",testpictures,birdname,self.contentpath)

            print("Bird #"+str(i) + " " + birdname + ": Completed")
        
        