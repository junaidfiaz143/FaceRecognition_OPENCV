import cv2

def detect_face(image, face_cascade):

    faces = face_cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5);
    
    if len(faces) == 0:
        return None, None
    else:
        print("Number of faces detected: ", len(faces))
        return faces

image_path = "training-data/s1/jd6.jpg"
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier("opencv-files/haarcascade_frontalface_alt.xml")
faces = detect_face(gray, face_cascade)

for i, face in enumerate(faces):
	if face is not None:
	    (x, y, w, h) = face
	    cv2.imshow("FACE: " + str(i), gray[y:y+w, x:x+h])
	    
	    cv2.waitKey(0)
	else:
		print ("[INFO] no face found!") 
		break

cv2.destroyAllWindows() 