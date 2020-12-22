# VidSort-ML

This is a Python project that utilizes machine learning to recognize classifier-trained faces within a set of videos and categorize them based on the classifier's confidence scores. [[inspiration](https://www.analyticsvidhya.com/blog/2018/08/a-simple-introduction-to-facial-recognition-with-python-codes/)]

## About

This project uses [openCV](https://pypi.org/project/opencv-python/) to access video files and load frames from the video such that they can be processed. This library is also used for image manipulation, such as gray-scale conversion, adding overlays, and saving.

In order to process each frame, the [face_recognition library](https://github.com/ageitgey/face_recognition/blob/master/README.md#installation) is used to detect the locations of all faces within a frame (if any), and to extract facial featuresets (encodings) of each face.

The found face encodings are then processed by [Sklearn's Multi-layer Perceptron (MLP) Neural Network (NN)](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) (referred to as models) in order to assign names to the faces. By default, the NN architecture is that of a 3 hidden layer deep neural network (DNN) each with 30 neurons. The input layer has 128 neurons, and by splitting the hidden neurons into 3 evenly sized layers with the full count calculated as [two-thirds the amount of input neurons plus the amount of output neurons](https://books.google.ca/books?id=Swlcw7M4uD8C&printsec=frontcover&dq=Introduction+to+Neural+Networks+for+Java,+Second+Edition+The+Number+of+Hidden+Layers&hl=en&sa=X&ved=0ahUKEwiq8675k7fpAhUWsJ4KHXnKB6wQ6AEIKDAA#v=onepage&q=Introduction%20to%20Neural%20Networks%20for%20Java%2C%20Second%20Edition%20The%20Number%20of%20Hidden%20Layers&f=false).

To train the DNN, [Selenium](https://pypi.org/project/selenium/) is used to autonomously obtain images from Google Images via a query with the name of each person within a user-provided list [[inspiration](https://towardsdatascience.com/image-scraping-with-python-a96feda8af2d)]. As images are pulled (using [requests](https://pypi.org/project/requests/)), they undergo a small amount of pre-processing to determine if they are fit for use in model training. Images that are "good enough" (*see image_scraper.py for more details*) are saved to "./images/training", whereas the rest are discared. Though factors such as the person's fame influence the ability to find images of their face, it has been found in early-stage testing that a request for 100 images of some of the top American celebrities resulted in 50-80 of the collected images being appropriate for training. Though not fully tested, it is recommended that **at least 50 images should be used for training a model to recognize any given person**.  It should also be noted that this number will increase with the number of people within the model.

With a trained model, all faces detected within an image will be compared against the collection of known faces, and assigned a probability of the face belonging to each person that the model is trained to recognize. If the probability is sufficiently high for a given person, and if the probability assigned to that person is higher than all the rest, the algorithm declares that the face belongs to that person. There is an even higher threshold at which the algorithm is so confident that the face belongs to a certain person that is takes a snapshot of the face and saves it so it can be used for future training iterations.

![example_pic_lalaland](https://raw.githubusercontent.com/m-rubik/VidSort-ML/master/examples/images/Lalaland.jpg)

![example_pic_avengers](https://raw.githubusercontent.com/m-rubik/VidSort-ML/master/examples/images/Avengers.jpg)

## Installation

Make sure that package installation order is cmake -> dlib -> face_recognition.
Should be **automatically achieved** with the following:
```bash
pip install -e .
```

## Setup
⚠️ Under construction... ⚠️

## Usage
⚠️ Under construction... ⚠️

The overall procedure is as follows:
1. OBTAIN images of people's faces
2. LEARN the faces (i.e, generate the face encodings of each person)
3. TRAIN a model (i.e, select the face encodings you want to train a model to recognize)
4. ANALYSE videos
5. SEARCH database for videos that contain the people you are looking for


### 1. Obtaining Images
**IF YOU HAVE IMAGES FOR TRAINING**
1. Create the nested directories "./images/testing". 
2. In "./images/testing" add a folder of images for each person you want the model to recognize. The name of the folder will be the name that gets assigned to that person.

**IF YOU DO NOT HAVE IMAGES**
1. Open *image_scraper.py*
2. Alter the value of *names* in the main call to contain the names of the people whose faces you want to learn:
```python
names= ["Ryan Gosling", "Emma Stone"]
```
3. Change the value of *number_images* to be the desired search amount:
```python
search_and_download(name, driver, number_images=100)
```
NOTE: Currently, this is only the number of images it will take from Google. If you request 100, it is **highly likely** that you **will not** get 100/100 images that are suitable for training. Adjust accordingly.

NOTE 2: It is suggested you manually browse through the images obtained from Google to ensure no blatant errors.

4. Run the script.

### 2. Learning faces ###
1. Ensure that "./images/training" is complete with sub-folders full of images for each person whose face encodings you want to learn
2. Run GUI.py
3. Change to the "Train" tab
4. Use the "Select Training Folder" button to navigate to "./images/training" (or wherever your top-level training directory is)
5. Press the "LEARN" button
6. Wait and monitor the console output. ⚠️ Currently this process runs in series, so it is fine if the GUI stops responding.

### 3. Training a model ###
1. Run GUI.py
2. Switch to the "Train" tab
3. In the table with all the known people (i.e, the face encodings stored in "./encodings/"), click the name of each person you want to add to the model. The name will go GREEN if it is SELECTED.
4. Enter a name for the model in the model name entry field
5. Chose the model type (MLP is suggested as default)
6. Press the "TRAIN" button
7. Wait and monitor the console output.

### 4. Analyzing a set of videos ###
1. Run GUI.py
2. Use the "Select Video Folder" button to navigate to the folder containing all videos you want to analyse
3. Use the "Select Classifier" button to select the model you want to use in the analysis (located in "./models"
4. Press "START"
5. Each video will now be analysed 1-by-1 (though in the future it may be done in parallel). For each video is processes, it will add an entry into a database (found in "./databases") with the same name as the model's name. The overall process is complete when the "START" button is no longer grayed-out.

### 5. Searching a database for people ###
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