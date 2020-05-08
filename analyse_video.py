"""!
TO BE REMOVED: OLD CODE
"""


import face_recognition
import cv2
import numpy as np

# Open video file
video_capture = cv2.VideoCapture("./videos/Youtube/Top 10 Most Popular Celebrities on Social Media.mp4")

# # Load a third sample picture and learn how to recognize it.
# mason_image = face_recognition.load_image_file("images/mason.jpg")
# mason_face_encoding = face_recognition.face_encodings(mason_image)[0]

# # Load a third sample picture and learn how to recognize it.
# marlene_image = face_recognition.load_image_file("images/marlene.jpg")
# marlene_face_encoding = face_recognition.face_encodings(marlene_image)[0]

# # Load a third sample picture and learn how to recognize it.
# alex_image = face_recognition.load_image_file("images/alex.jpg")
# alex_face_encoding = face_recognition.face_encodings(alex_image)[0]

# # Create arrays of known face encodings and their names
# known_face_encodings = [
#     mason_face_encoding,
#     marlene_face_encoding,
#     alex_face_encoding
# ]
# known_face_names = [
#     "Mason",
#     "Marlene",
#     "Alex"
# ]

lady_image = face_recognition.load_image_file("images/lady.png")
lady_face_encoding = face_recognition.face_encodings(lady_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    lady_face_encoding
]

known_face_names = [
    "Lady"
]

frames = []
frame_count = 0

seconds = 1
fps = video_capture.get(cv2.CAP_PROP_FPS) # Gets the frames per second
multiplier = round(fps * seconds)
# print(multiplier)

counter = 0
print("Running...")
while video_capture.isOpened():
    # Grab a single frame of video
    ret, frame = video_capture.read()

    frameId = int(round(video_capture.get(1)))

    if frameId % multiplier == 0:

        print("Frame:", counter, "Timestamp:", round(frameId/fps))
        counter = counter + multiplier
        # if counter > 20: # Only look at the first 20 frames for now
        #     ret = False

        # Bail out when the video file ends
        if not ret:
            break

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        frame = frame[:, :, ::-1]

        face_names = []
        face_confidences = []

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            face_confidences.append(face_distances[best_match_index])
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
            print(face_names)

video_capture.release()