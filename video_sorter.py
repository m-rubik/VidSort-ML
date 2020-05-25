import face_recognition
import cv2
import numpy as np
import os
from utilities.file_utilities import get_unique_filename
from utilities.model_utilities import load_object
from utilities.database_utilities import save_database, load_database
from sklearn.preprocessing import StandardScaler

class videoAnalyser():

    video_name: str
    model_name: str
    video_capture: cv2.VideoCapture
    capture_interval: int
    capture_multiplier: int

    def __init__(self, video_name, model_name, capture_interval):
        self.video_name = video_name
        self.model_name = model_name
        self.capture_interval = capture_interval

        # Initialize the video_face_data
        self.video_face_data = {}

        # Load the classifier
        self.model = load_object("./models/"+self.model_name)

        # Load the associated database
        self.db = load_database(self.model_name)

    def detect_faces_in_video(self, detection_threshold=75, save_threshold=100):

        # Open video file
        self.video_capture = cv2.VideoCapture(self.video_name)

        fps = self.video_capture.get(cv2.CAP_PROP_FPS) # Gets the frames per second
        self.capture_multiplier = round(fps * self.capture_interval)

        counter = 0
        print("Running...")

        while self.video_capture.isOpened():
           
            # Read a single video frame
            ret, frame = self.video_capture.read()

            # Exit if video is over
            if not ret:
                break

            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            frameId = int(round(self.video_capture.get(1)))

            if frameId % self.capture_multiplier == 0:

                counter = counter + self.capture_multiplier
                # print("Frame:", counter, "Timestamp:", round(frameId/fps)) ###################

                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                small_frame = small_frame[:, :, ::-1]

                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(small_frame)
                face_encodings = face_recognition.face_encodings(small_frame, face_locations)
                scaler = StandardScaler()
                if face_encodings:
                    scaler = StandardScaler()
                    face_encodings = scaler.fit_transform(face_encodings)

                index = 0
                for face_encoding in face_encodings:
                    prob = self.model.predict_proba([face_encoding])[0]
                    # name = clf.predict([face_encoding])
                    prob_per_class_dictionary = dict(zip(self.model.classes_, prob))
                    # print(prob_per_class_dictionary)
                    for face, probability in prob_per_class_dictionary.items():
                        if probability*100 > detection_threshold:
                            percentage = round(probability*100,2)
                            confidence = str(percentage)+"%"
                            print("Found", face, "with", confidence, "confidence on frame", counter, "(timestamp:", str(round(frameId/fps))+").")

                            try:
                                self.video_face_data[face].append(percentage)
                            except KeyError:
                                self.video_face_data[face] = list()
                                self.video_face_data[face].append(percentage)

                            # print(self.video_face_data) #################

                            if percentage >= save_threshold:
                                face_location = face_locations[index]
                                top = face_location[0]
                                right = face_location[1]
                                bottom = face_location[2]
                                left = face_location[3]

                                ## Display the results
                                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                                top *= 4
                                right *= 4
                                bottom *= 4
                                left *= 4
                            
                                draw = False
                                if draw:
                                    # Draw a box around the face
                                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                                    # Draw a label with a name below the face
                                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                                    font = cv2.FONT_HERSHEY_DUPLEX
                                    cv2.putText(frame, face+": "+confidence, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                                cropped_frame = frame[top:bottom, left:right]
                                self.save_face_to_img(face, cropped_frame)
                    
                    index += 1
                    ## MAKE A LIST ORDERED FROM HIGHEST TO LOWEST PROBABILITY
                    # results_ordered_by_probability = map(lambda x: x[0], sorted(zip(clf.classes_, prob), key=lambda x: x[1], reverse=True))

        self.video_capture.release()
        self.print_results()
        self.save_to_database()

    def save_face_to_img(self, face, frame):
        folder = "./images/"+face+"/"
        if not os.path.exists(folder):
            os.makedirs(folder)
        file_name = get_unique_filename("./images/"+face+"/")
        cv2.imwrite(folder+file_name+".jpg", frame)

    def print_results(self):
        print("\n============ FINAL RESULTS ============")
        print("The following people were recognized in", os.path.split(self.video_name)[1], "using the", self.model_name, "classifier model:")
        for person, data in self.video_face_data.items():
            print(person+":", str(round(sum(data)/len(data),1))+"%")
        
    def save_to_database(self):
        people_found = {}
        for person, data in self.video_face_data.items():
            people_found[person] = round(sum(data)/len(data),1)
        self.db.update(os.path.split(self.video_name)[1], people_found)
        save_database(self.db)
        # self.db.display_entries()
        print("Results saved to", self.model_name, "database.")


if __name__ == "__main__":
    va = videoAnalyser("./videos/Youtube/Top 10 Most Popular Celebrities on Social Media.mp4", "top9", 2)
    va.detect_faces_in_video(detection_threshold=75, save_threshold=85)