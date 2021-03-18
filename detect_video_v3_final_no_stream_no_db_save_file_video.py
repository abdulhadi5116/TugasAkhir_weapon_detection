import os
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
from datetime import datetime
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.python.ops import gen_image_ops
from timeit import default_timer as timer
#import firebase_admin
#from firebase import firebase
#from firebase_admin import credentials, firestore, storage
#from flask_opencv_streamer.streamer import Streamer
#from onesignal_sdk.client import Client
#import threading
#from statistics import mode
#from multiprocessing import Process



flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416-tiny', 'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', '0', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.3, 'score threshold')
flags.DEFINE_boolean('count', True, 'count objects within video')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'print info on detections')
flags.DEFINE_boolean('crop', False, 'crop detections from images')
flags.DEFINE_boolean('plate', False, 'perform license plate recognition')




def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video
    # get video name by using split method
    video_name = video_path.split('/')[-1]
    video_name = video_name.split('.')[0]
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0
    count_5 = 0
    start_time = time.time()
    gun_detector = []
    knife_detector = []
    total_object = 0
    totalobj_5f = []
    threat_lvl = 0
    get_data_firebase = "1"
    case_id = "-"
    weapon_type = 0
    count_push = 1
    #global firebase
    totalobj_mode_ref = 0
    buzz_trigger = "OFF"
    time_elapsed = 0
    fps = []
    time_elapsed_list = []
    now1 = datetime.now()
    dt_string1 = now1.strftime("%d%m%Y_%H%M%S")
    filename1 = "fps_" + dt_string1 + ".txt"
    fps_mean = 0
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_num += 1
            count_5 += 1
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
    
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.5,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.5,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to allow detections for only people)
        #allowed_classes = ['person']

        # if crop flag is enabled, crop each detection and save it as new image
        if FLAGS.crop:
            crop_rate = 150 # capture images every so many frames (ex. crop photos every 150 frames)
            crop_path = os.path.join(os.getcwd(), 'detections', 'crop', video_name)
            try:
                os.mkdir(crop_path)
            except FileExistsError:
                pass
            if frame_num % crop_rate == 0:
                final_path = os.path.join(crop_path, 'frame_' + str(frame_num))
                try:
                    os.mkdir(final_path)
                except FileExistsError:
                    pass          
                crop_objects(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), pred_bbox, final_path, allowed_classes)
            else:
                pass

        if FLAGS.count:
            # count objects found
            counted_classes = count_objects(pred_bbox, by_class = True, allowed_classes=allowed_classes)
            # loop through dict and print
            #now = datetime.now()
            #dt_string = now.strftime("%d/%m/%Y_%H%M%S")
            
                
            #for key, value in counted_classes.items():
                #if count_30 < 30 :
                   #gun_counter = dict.get(pistol, default = 0)
                   #knife_counter = dict.get(knife, default = 0)
                   #count_30 = 0;
                
                #print("Number of {}s: {}".format(key, value))
            image = utils.draw_bbox(frame, pred_bbox, FLAGS.info, counted_classes, allowed_classes=allowed_classes, read_plate=FLAGS.plate)
        else:
            image = utils.draw_bbox(frame, pred_bbox, FLAGS.info, allowed_classes=allowed_classes, read_plate=FLAGS.plate)
        
        #time_elapsed = time_elapsed + (time.time() - start_time)
        #time_elapsed_int = int(time_elapsed)
        #time_elapsed_list.append(time_elapsed_int)
        #for index in range(1, len(time_elapsed_list)):
            #if time_elapsed_list[index-1] == time_elapsed_list[index] :
                #fps.append(1.0/ (time.time() - start_time))
           #else :
                #fps_mean = sum(fps) / len(fps)
                #if time_elapsed_int % 30 != 0 :
                    #fps_mean = fps_mean + (sum(fps) / len(fps))
                    #print(sum(fps) / len(fps))
                    #print(time_elapsed_int)
                #else :
                    #fps_mean = fps_mean / 30
                    #print("fps_mean : ", fps_mean)
                    #with open(filename1, "a") as outfile:
                        #outfile.write("%.2f %d" % (fps_mean, time_elapsed_int))
                        #outfile.write("\n")
                        #outfile.close()
                #time_elapsed_list.clear()
                
                
        
        
        #minus12 = int(time_elapsed) % 31
        #print(minus12)
        #if minus12 == 0 :
            #fps.clear()
            #count = 1
        #if minus12 == 30 and count == 1:
            #fps1 = sum(fps) / len(fps)
            #print("FPS: %.2f Timer elapsed : %d" % (fps1, time_elapsed))
            #with open("fps.txt", "a") as outfile:
                #outfile.write("%.2f %d" % (fps1, time_elapsed))
                #outfile.write("\n")
                #outfile.close()
                #count = 0

        #else :
            #current_fps = 1.0/ (time.time() - start_time)
            #fps.append(current_fps)
            #print(current_fps)
            #print(len(fps))
        
        
        fps1 = 1.0/ (time.time() - start_time)
        gun_counter = counted_classes.get("pistol")   
        knife_counter = counted_classes.get("knife")
        gun_counter = counted_classes.setdefault("pistol", 0)   
        knife_counter = counted_classes.setdefault("knife", 0)
        total_object = gun_counter + knife_counter
        
        #Get list of detected weapon
        if knife_counter > 0 and gun_counter > 0 :
            weapon_type = 3
        elif knife_counter > 0 and gun_counter <= 0:
            weapon_type = 1
        elif knife_counter <= 0 and gun_counter > 0 :
            weapon_type = 2
        else :
            weapon_type = 0
        
        ### end of list detected weapon
        
        if knife_counter > 2 or gun_counter > 0 :
            threat_lvl = 2
        elif knife_counter > 1 :
            threat_lvl = 1
        else :
            threat_lvl = 0
            
        result = np.asarray(image)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        
        
        if (total_object > 0) :
            now = datetime.now()
            dt_string = now.strftime("%d%m%Y_%H%M%S")
            #os.chdir(os.path.abspath(os.getcwd()))
            #filename = dt_string + ".jpg"
            #cv2.imwrite(filename, result) 
            #print("Detection File Saved : ",filename)
            print ("frame count : ", frame_num)
            print ("gun_counter  : ", gun_counter)
            print ("knife _counter  : ", knife_counter)
            print ("Total Object  : ", total_object)
            print ("Level Ancaman : ", threat_lvl)
            print ("fps : %.2f" % fps1)
            print (" ")
        
            
        if not FLAGS.dont_show:
            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("result", result)
        
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
