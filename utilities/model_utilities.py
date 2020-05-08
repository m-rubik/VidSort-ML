import pickle
import face_recognition
from sklearn import svm
import os

def load_model(model_name):
    try:
        with open("./models/"+model_name, 'rb+') as f:
            model = pickle.load(f)
    except Exception as e:
        print(e)
        return 1
    return model

def save_model(model_name, model):
    try:
        with open("./models/"+model_name, 'wb+') as f:
            pickle.dump(model, f)
    except Exception as e:
        print(e)
        return 1
    return 0

def train_model(model_name):
    """
    Structure:
            images/
                training/
                    <person_1>/
                        1.jpg
                        2.jpg
                        .
                        .
                        n.jpg
                    <person_2>/
                        1.jpg
                        2.jpg
                        .
                        .
                        n.jpg
                    .
                    .
                    <person_n>/
                        1.jpg
                        2.jpg
                        .
                        .
                        n.jpg
    """
   
    # Training the SVC classifier

    # The training data would be all the face encodings from all the known images and the labels are their names
    encodings = []
    names = []

    # Training directory
    train_dir = os.listdir('./images/training/')

    # Loop through each person in the training directory
    for person in train_dir:
        pix = os.listdir("./images/training/" + person)
        print("Training classifier to recognize", person, "with", len(pix), 'images.')

        # Loop through each training image for the current person
        for person_img in pix:
            # Get the face encodings for the face in each image file
            face = face_recognition.load_image_file("./images/training/" + person + "/" + person_img)
            face_bounding_boxes = face_recognition.face_locations(face)

            if len(face_bounding_boxes) == 1: # If training image contains exactly one face
                face_enc = face_recognition.face_encodings(face)[0]
                # Add face encoding for current image with corresponding label (name) to the training data
                encodings.append(face_enc)
                names.append(person)
            elif len(face_bounding_boxes) == 0:
                print("WARNING: No face could be found in", person + "/" + person_img + " It is suggested you remove this image from this training folder.")
            else:
                print("WARNING: More than one face is detected in", person + "/" + person_img + ", so it cannot be used for training. It is suggested you remove this image from this training folder.")

    # Create and train the SVC classifier
    clf = svm.SVC(gamma='scale', probability=True)
    clf.fit(encodings,names)
    save_model(model_name, clf)

def test_model(model_name, test_image_name):
    """!
    For each person that the model is trained for:
    1. Load a picture of that person
    2. Ensure that the model recognizes that person with a very high confidence
    
    TODO: ALL OF THIS
    """

    clf = load_model(model_name)

    # Load the test image with unknown faces into a numpy array
    test_image = face_recognition.load_image_file(test_image_name)

    # Find all the faces in the test image using the default HOG-based model
    face_locations = face_recognition.face_locations(test_image)
    no = len(face_locations)
    print("Number of faces detected: ", no)

    # Predict all the faces in the test image using the trained classifier
    print("Found:")
    for i in range(no):
        test_image_enc = face_recognition.face_encodings(test_image)[i]
        print(clf.predict_proba([test_image_enc]))
        name = clf.predict([test_image_enc])
        print(*name)

if __name__ == "__main__":
    train_model("top9")
    # test_model("top9", "test_image.jpg")
