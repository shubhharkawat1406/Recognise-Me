import cv2
import face_recognition as fr

me_image = fr.load_image_file('me.jpeg')
me_face_encodings = fr.face_encodings(me_image)[0]

known_face_encodings = [me_face_encodings]
known_face_names = ["Me"]

webcam_video_cam = cv2.VideoCapture(0)
# Loop through every frame in the video
while True:
    success, img = webcam_video_cam.read()
    img = cv2.flip(img, 1)
    # Resizing to 1/4th of current image size
    img_small = cv2.resize(img, (0,0), fx=0.25, fy=0.25)
    
    all_face_locations = fr.face_locations(img_small, number_of_times_to_upsample=1, model='hog')
    all_face_encodings = fr.face_encodings(img_small, all_face_locations)
    all_face_names   = []  
   
    for current_face_locations,current_face_encodings in zip(all_face_locations,all_face_encodings):
        top_pos, right_pos, bottom_pos, left_pos = current_face_locations
        all_matches = fr.compare_faces(known_face_encodings,current_face_encodings)
        top_pos, right_pos, bottom_pos, left_pos = 4*top_pos, 4*right_pos, 4*bottom_pos, 4*left_pos
        
        name_of_person = ""
    
        if True in all_matches:
            first_match_index = all_matches.index(True)
            name_of_person = known_face_names[first_match_index]

            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name_of_person, (left_pos+5,bottom_pos-10), font, 2,(255,255,0), 1)       
            cv2.rectangle(img, (left_pos,top_pos), (right_pos,bottom_pos), (255,255,0), 2)
    cv2.imshow("Face Detection", img)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
            
webcam_video_cam.release()
cv2.destroyAllWindows()