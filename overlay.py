import cv2
import numpy as np

def writeOverlayText(image, textLines=[]):
    for i, textLine in enumerate(textLines):
        cv2.putText(
            img=image,
            text=textLine,
            org=(10,30*(i+1)),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=0.5,
            color=(255,255,255),
            thickness=1,
            lineType=cv2.LINE_AA)
