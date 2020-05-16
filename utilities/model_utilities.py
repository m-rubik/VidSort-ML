import pickle
import face_recognition
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from utilities.file_utilities import get_unique_filename
import statistics
import cv2
import os


def load_object(path):
    try:
        with open(path, 'rb+') as f:
            obj = pickle.load(f)
    except Exception as e:
        print(e)
        return 1
    return obj

def save_object(path, obj):
    try:
        folder = os.path.split(path)[0]
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(path, 'wb+') as f:
            pickle.dump(obj, f)
    except Exception as e:
        print(e)
        return 1
    return 0

def extract_all_face_encodings(path):
    train_dir = os.listdir(path)
    # TODO: Ensure that there is nothing but folders in this path...
    for person_folder in train_dir:
        if not os.path.exists("./encodings/"+person_folder):
            extract_face_encodings(path+person_folder)

def extract_face_encodings(path):
    """!
    Iterate through a folder of images and extract the
    facial featuresets (encodings). Add all encodings
    to a list and save it under "./encodings/[name]"
    """

    encodings = []

    name = os.path.split(path)[1]
    print("Extracting facial features of", name)
    training_images = os.listdir(path)
    
    for image in training_images:
        print("Analysing:", path + "/" + image)
        face = face_recognition.load_image_file(path + "/" + image)
        face_bounding_boxes = face_recognition.face_locations(face)

        if len(face_bounding_boxes) == 1: # Picture only contains 1 face
            face_enc = face_recognition.face_encodings(face)[0]
            encodings.append(face_enc)
        elif len(face_bounding_boxes) == 0: # Picture must contain 1 face
            print("WARNING: No face could be found. It is suggested you remove this image from future trainings.")
        else: # Picture cannot contain more than 1 face
            print("WARNING: More than one face is detected, so it cannot be used for training. It is suggested you remove this image from future trainings.")

    save_object("./encodings/"+name, encodings)
    return 0

def train_model(model_name, names, model_type="mlp"):
    """
    1. For each name in the names list, open the face encodings of that person and add them to the master list
    2. Fit the model with the master lists
    """

    master_encodings = []
    master_names = []

    for name in names:
        encodings = load_object("./encodings/"+name)
        for encoding in encodings:
            master_encodings.append(encoding)
            master_names.append(name)

    if model_type == "svc":
        clf = clf = svm.SVC(gamma='scale', probability=True)
    elif model_type == "knn":
        clf = KNeighborsClassifier(n_neighbors=15)
    elif model_type == "mlp":

        # https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
        num_input_neurons = master_encodings[0].size
        num_output_neurons = len(names)
        num_samples = len(master_encodings)
        scaling_factor = 2
        # num_hidden_nodes = num_samples/(scaling_factor*(num_input_neurons+num_output_neurons))
        num_hidden_nodes = round(num_input_neurons*(2/3) + num_output_neurons)
        # num_hidden_nodes = round(statistics.mean([num_input_neurons, num_output_neurons]))

        num_hidden_layers = 3
        hidden_layer_sizes = tuple()
        for _ in range(num_hidden_layers):
            hidden_layer_sizes = hidden_layer_sizes + (round(num_hidden_nodes/num_hidden_layers),)

        clf = MLPClassifier(hidden_layer_sizes=(num_hidden_nodes, ), max_iter=1000, verbose=True)

    print("Training", model_type, "model...")
    clf.fit(master_encodings,master_names)
    save_object("./models/"+model_name, clf)
    print("Model", model_name, "has been trained.")

def test_model(model_name, test_image_name):
    """!
    For each person that the model is trained for:
    1. Load a picture of that person
    2. Ensure that the model recognizes that person with a very high confidence
    """

    clf = load_object("./models/"+model_name)

    # Load the test image with unknown faces into a numpy array
    test_image = face_recognition.load_image_file(test_image_name)

   # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(test_image)
    face_encodings = face_recognition.face_encodings(test_image, face_locations)

    index = 0
    for face_encoding in face_encodings:
        prob = clf.predict_proba([face_encoding])[0]
        # name = clf.predict([face_encoding])
        prob_per_class_dictionary = dict(zip(clf.classes_, prob))
        # print(prob_per_class_dictionary)
        for face, probability in prob_per_class_dictionary.items():
            if probability >= 0.7:
                percentage = round(probability*100,2)
                confidence = str(percentage)+"%"
                print("Found", face, "with", confidence, "confidence")

                face_location = face_locations[index]
                top = face_location[0]
                right = face_location[1]
                bottom = face_location[2]
                left = face_location[3]
                
                draw = True
                if draw:
                    # Draw a box around the face
                    cv2.rectangle(test_image, (left, top), (right, bottom), (0, 0, 255), 2)

                    # Draw a label with a name below the face
                    cv2.rectangle(test_image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(test_image, face+": "+confidence, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                # cropped_frame = test_image[top:bottom, left:right]
                folder = "./images/"+face+"/"
                if not os.path.exists(folder):
                    os.makedirs(folder)
                file_name = get_unique_filename("./images/"+face+"/")
                cv2.imwrite(folder+file_name+".jpg", cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR))
        index += 1

if __name__ == "__main__":
    # extract_all_face_encodings('./images/training/')
    names = ["Ryan Gosling", "Emma Stone"]
    # names = ["Ariana Grande", "Beyonce", "Chris Pratt", "Dwayne Johnson", "Justin Bieber", "Kim Kardashian", "Kylie Jenner", "Rihanna", "Selena Gomez", "Taylor Swift"]
    train_model(model_name="lalaland_mlp", names=names, model_type="mlp")
    # test_model("top9", "1.jpg")