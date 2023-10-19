from os import path
import os 
from PIL import Image
import urllib, json,urllib.request

#This class is used for placing all the picture of a certain species
#which was scraped from Avibase into the appropriately file directory.

class populatetrainandtest:

    def __init__(self):
         self=self  
    
    #populate 
    #
    # Parameters: 
    # 
    # test_train_str - A string object which can either be "train" or "test" depending
    #                  on whether the images are being placed in the test or train
    #                  datasets.
    #
    # linkpicturearray - An array that holds all of the links to images to be placed 
    #                    in the dataset as strings
    #
    # birdname - A string indicating the name of the species of bird
    #
    # contentpath - A string containing the path where the test and train will be 
    #               contained.
    #
    # Returns: None
    #
    # Effects:
    #
    # Creates the directory with the same name as the birdname in the same path of 
    # contentpath + test_train_str. Then for each picture link in linkpicturearray
    # the image is placed in that directory

    def populate(self,test_train_str,linkpicturearray,birdname,contentpath):

        pathname = contentpath +test_train_str+"/"
        path = os.path.join(pathname, birdname) 
        os.mkdir(path) # New directory is created which the name of the species

        for j,picture in enumerate(linkpicturearray,1):
                try:

                    urllib.request.urlretrieve(picture, "gfg.png") #Requesting the png from the png link
                except:
                    print("404 Error")
                    continue
                

                img=Image.open("gfg.png")
                imagepathname=pathname+birdname+"/image"+str(j)+".png"
                img.save(imagepathname)