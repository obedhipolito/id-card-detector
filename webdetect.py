#look for label_map_util.py --- change tf.io.
# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import subprocess
import dlib
import pandas
#import tensorflow.compat.v1 as tf
from skimage import io,draw,transform,color
from PIL import Image
from PIL import ImageOps
import pytesseract
from flask import Flask, request
sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

directory_path = os.path.dirname(os.path.realpath(__file__)) 
UPLOAD_FOLDER = f"{directory_path}\\..\\ine\\testing\\"
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif'}




def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



#Name= "Arturo Getsemani Roa Borbolla"
##Domicilio="Av Francisco I Madero 31"
#Ciudad ="Orizaba"
#Colonia = "El Yute 94340"

MODEL_NAME = 'model'
output_path = 'C:\\Users\\artur\\Git\\Uv\\python\\test.jpg'
CWD_PATH = os.getcwd()

PATH_TO_CKPT = "C:\\Users\\artur\\Git\\Uv\\python\\id-card-detector\\model\\frozen_inference_graph.pb"
PATH_TO_LABELS = "C:\\Users\\artur\\Git\\Uv\\python\\id-card-detector\\data\\labelmap.pbtxt"


NUM_CLASSES = 1

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
#IMAGE_NAME = 'C:\\Users\\artur\\Git\\Uv\\python\\ine\\inenegro.jpg'

# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/', methods=['GET', 'POST'])
def form_example():
    if request.method == 'GET':
        return '''<form method="POST" enctype = "multipart/form-data">
        Nombre: <input type="text" name="name" id="name"/><br>
        Direccion: <input type="text" name="direccion" id="direccion"/><br>
        Numero: <input type="text" name="numero" id="numero"/><br>
        Colonia: <input type="text" name="colonia" id="colonia"/><br>
        Ciudad: <input type="text" name="ciudad" id="ciudad"/><br>
        Estado: <input type="text" name="estado" id="estado"/><br>
        Course: <input type="file" name="file" id="file"/><br>
        <input type="submit" value="Submit"/>
        </form>'''
    if request.method == 'POST':
        data = request.form.get('name').upper().split(" ") + request.form.get('direccion').upper().split(" ") + request.form.get('numero').upper().split(" ") + request.form.get('colonia').upper().split(" ") + request.form.get('ciudad').upper().split(" ") + request.form.get('estado').upper().split(" ")         
        print(data)
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            #flash('No selected file')
            print("Cant find name")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #return redirect(url_for('download_file', name=filename))
        IMAGE_NAME = os.path.join(app.config['UPLOAD_FOLDER'], filename) #f"{directory_path}\\id-card-detector\\{file.filename}"
        PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME) 
        image_path  = IMAGE_NAME
        im = Image.open(image_path)
        #im = im.resize((480,640))
        im = ImageOps.exif_transpose(im)
        image = np.array(im)
        #image = cv2.imread(PATH_TO_IMAGE)
        #image = cv2.resize(image,(480,640))
        image_expanded = np.expand_dims(image, axis=0)
        image3=image.copy()
        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})
        # Draw the results of the detection (aka 'visulaize the results')
        image, array_coord = vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=3,
            min_score_thresh=0.95)
        xmin,xmax,ymin,ymax =array_coord
        shape = np.shape(image)
        y,x,z=image.shape
        im_width, im_height = shape[1], shape[0]
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
        print(left, right, top, bottom)
        #im.crop((left, top, right, bottom)).save(output_path, quality=95)
        #cv2.imshow('ID-CARD-DETECTOR : ', image)
        maxscore=[0,0,0,0]

        for i in range(len(scores)):
            if scores[i][0]  > maxscore[0] and scores[i][1]  > maxscore[1] and scores[i][2]  > maxscore[2] and scores[i][3]  > maxscore[3]:
                score=i
        ymin1 = boxes[0][score][1]*im_height
        xmin1 = boxes[0][score][0]*im_width
        ymax1 = boxes[0][score][3]*im_height
        xmax1 = boxes[0][score][2]*im_width
        #ROI=image3[int(left):int(bottom),int(right):int(top)]
        #ROI = cv2.resize(ROI,(1280,960))
        #cv2.imshow("Testing",ROI)
        #cv2.waitKey(0)
        #cv2.imwrite(output_path,ROI)
        ##result = subprocess.check_output(f"python C:\\Users\\artur\\Git\\Uv\\python\\pyocr.py {output_path}", shell=True)
        ##print(result)


        #cv2.imshow("Fuck",)
        image2=image[:,:,1]       
        image2[image2<=254]=0
        
        th, im_th = cv2.threshold(image2, 254, 255, cv2.THRESH_BINARY);
        
        # Copy the thresholded image.
        im_floodfill = im_th.copy()
        
        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = im_th.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        
        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0,0), 255);
        
        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        
        # Combine the two images to get the foreground.
        im_out = im_th | im_floodfill_inv
        im_out[im_out<=254]=0
        #im_out[im_out>=254]=1
        #image3=image3*im_out
        # Display images.
        #cv2.imshow("Foreground", im_out)

        #print(int(xmin),int(xmax),int(ymin),int(ymax))
        #image_cropped = image[int(ymin):int(ymax)][int(xmin):int(xmax)]
        #image_cropped = cv2.imread(output_path)
        #print(image_cropped.shape)
        #cv2.imshow("ID-CARD-CROPPED : ", image_cropped)

        # All the results have been drawn on image. Now display the image.
        #cv2.imshow('ID CARD DETECTOR', image3)

        # Press any key to close the image
        #v2.waitKey(0)
        white_pixels = np.array(np.where(im_out == 255))
        #print(white_pixels)
        first_white_pixel = white_pixels[:,0]
        last_white_pixel = white_pixels[:,-1]
        print(xmin,xmax,ymin,ymax)
        xmin2=image.shape[1]
        xmax2=0
        ymin2=image.shape[0]
        ymax2=0
        for i in white_pixels[0]:
            if i < xmin2:
                xmin2=i
            if i > xmax2:
                xmax2=i
        for i in white_pixels[1]:
            if i > ymax2:
                ymax2=i
            if i > ymin2:
                ymin2=i
        print(xmin2,xmax2,ymin2,ymax2)

        # displaying the image after drawing contours
        print(first_white_pixel,last_white_pixel)
        xmin3=first_white_pixel[0]
        xmax3=last_white_pixel[0]
        ymin3=first_white_pixel[1]
        ymax3=last_white_pixel[1]
        print(xmin3,xmax3,ymin3,ymax3)
        xminm=int(min(left,xmin1,xmin2,xmin3))
        xmaxm=int(max(right,xmax1,xmax2,xmax3))
        ymaxm=int(max(bottom,ymax1,ymax2,ymax3))
        yminm=int(min(top,ymin1,ymin2,ymin3))
        print(xminm,xmaxm,yminm,ymaxm)
        #print(image3.shape)
        ROI=image3[xminm:xmaxm,yminm:ymaxm]
        cv2.imwrite(output_path,ROI)
        #cv2.imshow("Cropped1",ROI)
        #cv2.imshow("Testing",image3[0:1599,0:899])
        #cv2.waitKey(0)
        i=0

        faces=cv2.CascadeClassifier(f'{directory_path}\\..\\ine\\faces.xml')
        for i in range(0,18):
            image2=image3[:,:,0]
            image3=image3.copy()
            rotation_angle= i * 22
            (h, w) = image2.shape[:2]
            (cX, cY) = (w // 2, h // 2)
            # rotate our image by 45 degrees around the center of the image
            M = cv2.getRotationMatrix2D((cX, cY), rotation_angle, 1.0)
            image2 = cv2.warpAffine(image2, M, (w, h))
            image3 = cv2.warpAffine(image3, M, (w, h))
            faces_detected = faces.detectMultiScale(image2, scaleFactor = 1.1, minNeighbors = 5)
            if len(faces_detected) >=1:
                for (fx,fy,fw,fh) in faces_detected:
                    cv2.rectangle(image3,(fx,fy),(fx+fw,fy+fh),(255,0,0),5)
                    print("Detected")
                break
        gray=image3[:,:,0]
        retval, imagebin = cv2.threshold(gray, 50, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        #imagebin[0:bottom2,left2:-1] = 255
        img_bilateralFilter = cv2.bilateralFilter(imagebin, 40, 100, 100) # Filtrado bilateral gaussiano
        text=pytesseract.image_to_string(gray)
        text1=text.upper()
        print(text)
        text2=pytesseract.image_to_string(imagebin)
        print(text)
        text2=text2.upper()
        #Check = data.upper().split(" ")#Name.upper().split(" ") + Domicilio.upper().split(" ") + Ciudad.upper().split(" ") + Colonia.upper().split(" ")
        #print(Check)
        total = len(data)
        count = 0
        detected=[]
        for word  in data:
            if word in text1 or word in text2 and word != " " and word != "":
                count+=1
                detected.append(word)
        probabilidad = count / total
        print(probabilidad)
        #cv2.imshow("Face Detected", image[fy:fy+fh,fx:fx+fw])
        return f'''
        <h1>Your data detected  is : {detected} </h1>
        Probability is: {probabilidad}
                '''
    


if __name__ == '__main__':
    app.run("0.0.0.0")
