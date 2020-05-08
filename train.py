import pickle
from sklearn import svm
import os
import cv2
from PIL import Image
import sys

RESIZE_SIZE = 800

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
            objects/
                training/
                    <object_1>/
                        1.jpg
                        2.jpg
                        .
                        .
                        n.jpg
                    <object_2>/
                        1.jpg
                        2.jpg
                        .
                        .
                        n.jpg
                    .
                    .
                    <object_n>/
                        1.jpg
                        2.jpg
                        .
                        .
                        n.jpg
                test/
                    <object_1>.jpg
                    <object_2>.jpg
                    .
                    .
                    <object_n>.jpg
    """
    
    names = []
    encodings = []
 
    # Training directory
    train_dir = os.listdir('./objects/training/')

    # Loop through each person in the training directory
    for person in train_dir:
        pix = os.listdir("./objects/training/" + person)
        print("Training classifier to recognize", person, "with", len(pix), 'images.')

        path = "./objects/training/" + person + '/'
        dirs = os.listdir(path)
        final_size = RESIZE_SIZE
        resize(path, dirs)
        # resize_aspect_fit(path, dirs, final_size)

        for person_img in pix:
            f, e = os.path.splitext(person_img)
            if e == ".png":
                im = Image.open("./objects/training/" + person + '/' + f + ".png")
                im = im.convert("RGB")
                im.save("./objects/training/" + person + '/' + f + '.jpg', 'JPEG', quality=95)
                os.remove("./objects/training/" + person + '/' + f + ".png")

        # Loop through each training image for the current person
        for person_img in pix:
            image = cv2.imread("./objects/training/" + person + "/" + person_img, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                # image = create_features(image)
                names.append(person)
                encodings.append(image.flatten())

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

    image = cv2.imread(test_image_name, cv2.IMREAD_GRAYSCALE)
    im = Image.open(test_image_name)
    f, e = os.path.splitext(test_image_name)
    imResize = im.resize((RESIZE_SIZE,RESIZE_SIZE), Image.ANTIALIAS)
    imResize = imResize.convert("RGB")
    imResize.save(f + '.jpg', 'JPEG', quality=90)
    image = cv2.imread(test_image_name, cv2.IMREAD_GRAYSCALE)
    image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # image = create_features(image)

    prob = clf.predict_proba([image.flatten()])[0]
    prob_per_class_dictionary = dict(zip(clf.classes_, prob))
    for face, probability in prob_per_class_dictionary.items():
        print(face+": "+str(round(probability*100,2))+"%")

def resize_aspect_fit(path, dirs, final_size):
    for item in dirs:
        if item == '.DS_Store':
            continue
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            size = im.size
            ratio = float(final_size) / max(size)
            new_image_size = tuple([int(x*ratio) for x in size])
            im = im.resize(new_image_size, Image.ANTIALIAS)
            new_im = Image.new("RGB", (final_size, final_size))
            new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
            new_im.save(f + '.jpg', 'JPEG', quality=95)

def resize(path, dirs):
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((RESIZE_SIZE,RESIZE_SIZE), Image.ANTIALIAS)
            imResize = imResize.convert("RGB")
            imResize.save(f + '.jpg', 'JPEG', quality=90)

def create_features(img):
    from skimage.feature import hog
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    # hog_features, hog_image = hog(img,
    #                           visualize=True,
    #                           block_norm='L2-Hys',
    #                           pixels_per_cell=(16, 16))

    # plt.imshow(hog_image, cmap=mpl.cm.gray)
    # plt.show()

    hog_features = hog(img, block_norm='L2-Hys', pixels_per_cell=(16, 16))
    # combine color and hog features into a single array
    flat_features = np.hstack((img.flatten(), hog_features))
    return flat_features

if __name__ == "__main__":
    train_model("fruits")
    test_model("fruits", "./objects/test/apple.jpg")
    test_model("fruits", "./objects/test/orange.jpg")
