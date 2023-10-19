## Project Name: 

Woo Are You

## Project Description: 

This project is still in ongoing development.

Woo Are You is an application that allows users to take pictures of birds in the Metropolitan area of New York City and will provide users with the species of bird that was just photographed and, subsequently, other information regarding that specific bird.  This application is essential for any bird watchers in the city but is also utilized as a tool for individuals to identify species of birds that may be endangered and help them take appropriate action to help promote their survival. This project involves web scraping and machine learning with a CNN. Furthermore, front-end aspects will all be implemented.

## Technology Utilized: 

- Python
- PyTorch
- torch-vision
- BeautifulSoup

## Features/Bugs

- Data and Pictures of the bird species are scraped from Avibase and EBird. As a result, all data on them will be up-to-date

- Uses CNN to learn from the pictures and allow for user-inputted pictures to be 
analyzed accurately and return the name of the captured species. Has 54% accuracy due
to overfitting; however, I will be scraping more databases for more images to address
the issue.

- After fixing the overfitting issue, I will be adding functionality to display bird data and add graph 
representations of train and validation set accuracy for each epoch.

## File Breakdown

- birddata.py: This file contains a web scraper class that extracts pictures and data from the databases Avibase and EBird

- populatetrainandtest.py: This file contains a class that creates directories in the test and train dataset files for each species of bird and populates them with pictures of that species.

- trainer.py: This holds the trainingclass which has helper functions for producing the model. The function "train_model" trains resnet 50 with our previously created dataset. "imshow" helps with data visualization after training the model.

- birdmodel.py: Holds the "birdmodel" class. When "savetrainedmodel" is called the resnet 50 model is trained on the data in PyTorchBirdData and saved as "birdclassiferresnet50".

- PyTorchBirdData: holds the directories train and test which house all of the image data.
              
