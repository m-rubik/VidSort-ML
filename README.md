# VidSort-ML

Python project that utilizes machine learning to recognize classifier-trained faces within a set of videos and categorize them based on the classifier's confidence scores. [[inspiration](https://www.analyticsvidhya.com/blog/2018/08/a-simple-introduction-to-facial-recognition-with-python-codes/)]

This project uses [openCV](https://pypi.org/project/opencv-python/) to access video files and load frames from the video such that they can be processed. This library is also used for image manipulation, such as gray-scale conversion, adding overlays, and saving.

In order to process each frame, the [face_recognition library](https://github.com/ageitgey/face_recognition/blob/master/README.md#installation) is used to detect the locations of all faces within a frame (if any), and to extract facial featuresets (encodings) of each face.

The found face encodings are then processed by [Sklearn's Support Vector Classifiers (SVC)](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) (referred to as models) in order to assign names to the faces.

To train the SVC, [Selenium](https://pypi.org/project/selenium/) is used to autonomously obtain images from Google Images via a query with the name of each person within a user-provided list [[inspiration](https://towardsdatascience.com/image-scraping-with-python-a96feda8af2d)]. As images are pulled (using [requests](https://pypi.org/project/requests/)), they undergo a small amount of pre-processing to determine if they are fit for use in model training. Images that are "good enough" (*see image_scraper.py for more details*) as saved to "./images/training", whereas the rest are discared. Though factors such as the person's fame influence the ability to find images of their face, it has been found in early-stage testing that a request for 100 images of some of the top American celebrities resulted in 50-80 of the collected images being appropriate for training. Though not fully testing, it is recommended that **at least 50 images should be used for training a model to recognize any given person**.  It should be also be noted that this number will increase with the number of people within the model.

With a trained model, all faces detected within an image will be compared against the collection of known faces, and assigned a probability of the face belonging each person that the model is trained to recognize. If the probability is sufficiently high for a given person, and if the probability assigned to that person is higher than all the rest, the algorithm declares that the face belongs to that person. There is an even higher threshold at which the algorithm is so confident that the face belongs to a certain person that is takes a snapshot of the face and saves it so it can be used for future training iterations.

![face_detection_picture](https://raw.githubusercontent.com/m-rubik/VidSort-ML/master/examples/images/1.jpg)

![face_detection_picture_2](https://raw.githubusercontent.com/m-rubik/VidSort-ML/master/examples/images/2.jpg)

## Installation

In order to install the *face_recognition* library, ensure that the order of pip install is as follows:
```bash
pipenv install cmake
pipenv install dlib
pipenv install face_recognition
```

Note that this should be **automatically achieved** with the following:
```bash
pipenv install --sequential
```

## Setup
⚠️ Under construction... ⚠️

In the project's root directory:
1. create a folder titled "images" and within that folder create a folder called "testing".
2. create an "objects" folder ???? Only for certain things, not normal operation ???
3. create a "videos" folder and put all your videos in there ??? I don't think this is necessary...


## Usage
⚠️ Under construction... ⚠️

### Obtaining Images
**IF YOU HAVE IMAGES FOR TRAINING**
In the "./images/testing" directory add a folder for each person you want the model to recognize. The name of the folder will be the name that gets assigned to that person. Within each person's folder, name the images as 1.jpg, 2.jpg, ... 

**IF YOU DO NOT HAVE IMAGES**
1. Open *image_scraper.py*
2. Alter the value of *list_of_names* as like so:
```python
list_of_names = ["Ryan Gosling", "Emma Stone"]
```
3. Change the value of *number_images* to be the desired search amount:
```python
search_and_download(name, driver, number_images=100)
```
NOTE: Currently, this is only the number of images it will take from Google. If you request 100, it is **highly likely** that you **will not** get 100/100 images that are suitable for training. Adjust accordingly.

NOTE 2: It is suggested you manually browse through the images obtained from Google to ensure no blatant errors.

4. Run the script.

### Training the Model ###
1. Ensure that "./images/training" is complete with sub-folders full of images for each person you want to train to recognize.
2. Open *./utilities/model_utilities.py*
3. At the very bottom, change the text within the *train_model* call to be whatever you want the model's name to be. If you want to test the model after training, change the first string to be the model's name, and then change the second to be the name of image to test against (NOTE: This image **must** be located in the projects **root** folder)
```python
if __name__ == "__main__":
    train_model("top10")
    # test_model("top9", "1.jpg")
```
4. Run the script.

### Analyzing a set of videos ###
1. In "./videos", create a sub-folder and put all of the videos that you want to analyse within it.
2. Run GUI.py
3. Use the "Select Video Folder" button to navigate to the folder containing all videos you want to analyse
4. Use the "Select Classifier" button to select the model you want to use in the analysis (located in "./models"
5. Press "START"
6. Each video will now be analysed 1-by-1 (though in the future it may be done in parallel). For each video is processes, it will add an entry into a database (found in "./databases") with the same name as the model's name. The overall process is complete when the "START" button is no longer grayed-out.

### Searching databases for people ###
Once your video sets have been analysed:
1. Open **./utilities.database_utilities.py**
2. use the following commands and make adjustments where required ("Test" is the model's name):
```python
db = load_database("Test")
db.search("Person 1")
db.search(["Person 1", "Person 2"])
```
3. To see all entries in database:
```python
db.display_entries()
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)