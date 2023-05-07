import cv2
import numpy as np
from PIL import Image,ImageEnhance
#import pybase64 as base64
import glob
import mediapipe1 as mp1
"""def converttobase64():
    path = '/usr/share/enrollment/croppedimg/'
    dest = '/usr/share/enrollment/croppedimg/'
    for filepath in glob.glob(path+'*png'):

        file_name = filepath[filepath.index(path)+len(path):] 
        with open(filepath, "rb") as image2string:
            converted_string = base64.b64encode(image2string.read())
        print(converted_string)
    
        with open(dest+file_name[:-3]+'csv', "wb") as file:
            file.write(converted_string)
"""
frame = cv2.imread('/usr/share/enrollment/images/input.jpg')
frame = cv2.flip(frame, 1)
emptyframe=frame[np.abs(100):np.abs(450), np.abs(int(150)):np.abs(450)]
#dim = (413, 413)	
#resized = cv2.resize(emptyframe, dim, interpolation = cv2.INTER_AREA)
#cv2.imwrite('/usr/share/enrollment/croppedimg/out.jpg', resized)
#rcompressed = cv2.resize(emptyframe, (192, 192) , interpolation = cv2.INTER_AREA)
#cv2.imwrite("/usr/share/enrollment/croppedimg/compressed.jpg",rcompressed)
im = Image.fromarray(cv2.cvtColor(emptyframe, cv2.COLOR_BGR2RGB)).convert("RGBA")
enhancer = ImageEnhance.Brightness(im)
factor = 1.4 #brightens the image
imb = enhancer.enhance(factor)
enhancer1 = ImageEnhance.Contrast(imb)
image=enhancer1.enhance(0.7)
image.resize((600,600), Image.ANTIALIAS).save('/usr/share/enrollment/croppedimg/out.png', "PNG", dpi=(300, 300), optimize=False,quality=100)
image = Image.fromarray(cv2.cvtColor(emptyframe, cv2.COLOR_BGR2RGB)).convert("RGBA")
image.resize((64,64), Image.ANTIALIAS).save('/usr/share/enrollment/croppedimg/compressed.png', "PNG", dpi=(300, 300), optimize=False,quality=100)
#converttobase64()
mp1.mp_exec()
#converttobase64()
#iprint("Valid image cropped along Red Box since face is not detected. Press start capture, if face not correct")
