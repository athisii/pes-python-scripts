import cv2
import mediapipe as mp
import numpy as np
import random
from PIL import Image,ImageEnhance,ImageFilter

def mp_exec():
    p_drawing = mp.solutions.drawing_utils
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    BG_COLOR = (192, 192, 192) # gray
    MASK_COLOR = (255, 255, 255) # white

    with mp_selfie_segmentation.SelfieSegmentation(
        model_selection=0) as selfie_segmentation:


        image = cv2.imread('/usr/share/enrollment/croppedimg/out.png')
        
        b,g, r = cv2.split(image)
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = selfie_segmentation.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # convert to 0 and 1
        #condition = (results.segmentation_mask > 0.25).astype(np.uint8)
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.80
        #Erosion for mask
        #kernel = np.ones((5, 5), np.uint8)
        #img_erosion = cv2.erode(condition, kernel, iterations=1)
        fg_image = np.zeros(image.shape, dtype=np.uint8)
        fg_image[:] = MASK_COLOR
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        output_image = np.where(condition, fg_image, bg_image)
        gray = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
        
        ret,thresh1 = cv2.threshold(gray,200,1,cv2.THRESH_BINARY)
        #add aplpha channel to image
        rgba = cv2.merge([r, g, b, thresh1*255], 4)
       	#Smoothen
        img=rgba.copy() 

        """alpha=img[:,:,3]
        alpha = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)[1]
        
        alphamask=Image.fromarray(alpha)
        alphamask=alphamask.filter(ImageFilter.ModeFilter(size=13))
        
        image_data = np.asarray(image)

        alphamask1=np.asarray(alphamask)

        res = cv2.bitwise_and(image_data,image_data,mask = alphamask1)
        res1=cv2.cvtColor(res, cv2.COLOR_RGBA2BGRA)
        res1[:, :, 3] = alphamask1"""

        im = Image.fromarray(rgba)                                                        
        enhancer = ImageEnhance.Brightness(im)
        factor = 1.4 #brightens the image
        imb = enhancer.enhance(factor)
        enhancer1 = ImageEnhance.Contrast(imb)
        trans_img=enhancer1.enhance(0.7)
        trans_img.resize((64,64), Image.ANTIALIAS).save('/usr/share/enrollment/croppedimg/compressedsub.png', "PNG", dpi=(300, 300), optimize=False,quality=100)
        trans_img1 =enhancer1.enhance(factor)                                                      
	
        trans_img1.resize((600,600), Image.ANTIALIAS).save('/usr/share/enrollment/croppedimg/sub.png', "PNG", dpi=(300, 300), optimize=False,quality=100)    
        
        print("Valid image")
