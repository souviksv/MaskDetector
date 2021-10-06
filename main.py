import cv2
from mask_detect import MaskDetector
from face_detector import FaceDetector


def main():
    face_detector = FaceDetector()
    detect_mask = MaskDetector()
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL

    while(True):
        _, frame = cap.read()
        height, _ = frame.shape[:2]
        get_cropped_face = face_detector.detect_face(frame)
        label = detect_mask.classify_mask(get_cropped_face)
        if(label == 'with_mask'):
            cv2.putText(frame,str(label),(100,height-20), font, 2,(0,255,0),1,cv2.LINE_AA)
        else:
            cv2.putText(frame,str(label),(100,height-20), font, 2,(0,0,255),1,cv2.LINE_AA)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()