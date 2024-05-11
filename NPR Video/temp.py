import cv2

def gen_frames():
    harcascade = "model/haarcascade_russian_plate_number.xml"
    cap = cv2.VideoCapture(0)
    cap.set(3, 640) # width
    cap.set(4, 480) # height
    min_area = 500
    count = 0
    while True:
        success, img = cap.read()
        plate_cascade = cv2.CascadeClassifier(harcascade)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)
        for (x,y,w,h) in plates:
            area = w * h
            if area > min_area:
                cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(img, "Number Plate", (x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
                img_roi = img[y: y+h, x:x+w]
                cv2.imshow("ROI", img_roi)  

        if cv2.waitKey(1) & 0xFF == ord('s'):
            file_path = "plates/scaned_img_" + str(count) + ".jpg"
            cv2.imwrite(file_path, img)  
            cv2.rectangle(img, (0,200), (640,300), (0,255,0), cv2.FILLED)
            cv2.putText(img, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
            cv2.imshow("Results",img)
            cv2.waitKey(500)
            count += 1

        # Convert the image to JPEG format
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Testing the function
for frame in gen_frames():
    cv2.imshow('frame', cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), -1))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

