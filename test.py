import cv2
import argparse
from pathlib import Path
import torch
from mtcnn.detector import FaceDetector
from face_learner import FaceLearner
from utils import load_facebank, draw_box_name, prepare_facebank


def main():
    parser = argparse.ArgumentPa  rser(description='for face verification')
    parser.add_argument("-f",x "--file_name", help="video file path",default=None, type=str)
    parser.add_argument("-s", "--save_name", help="output file name",default='recording', type=str)
    parser.add_argument('-th','--threshold',help='threshold to decide identical faces',default=1.54, type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank",action="store_true")
    parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true")
    parser.add_argument("-b", "--begin", help="from when to start detection(in seconds)", default=0, type=int)
    parser.add_argument("-d", "--duration", help="perform detection for how long(in seconds)", default=0, type=int)
    parser.add_argument("-c", "--cpu", action='store_true', help="use cuda")

    args = parser.parse_args()

    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    face_detector = FaceDetector()
    face_recogniser = FaceLearner()
    face_recogniser.threshold = args.threshold
    face_recogniser.load_state()
    print ("Face detector and recogniser loaded. Using {}".format(device))

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        isSuccess,image = cap.read()
        if isSuccess:
            bboxes, faces = face_detector.getFaces(image, face_limit, 16, 112)
            if bboxes is None or bboxes.numel() == 0:
                print('no face')
                continue

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()