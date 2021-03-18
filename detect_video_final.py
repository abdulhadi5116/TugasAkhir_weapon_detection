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
import firebase_admin
from firebase import firebase
from firebase_admin import credentials, firestore, storage
from flask_opencv_streamer.streamer import Streamer
from onesignal_sdk.client import Client
from statistics import mode
from multiprocessing import Process


##buzzer setup
try :
    import RPi.GPIO as GPIO
    GPIO.setwarnings(False) 
    GPIO.setmode(GPIO.BCM)
    buzzer=23
    GPIO.setup(buzzer,GPIO.OUT) 
except :
    print('not running on Raspberry OS')

###Flask opencv streame

port = 5000
require_login = False
streamer = Streamer(port, require_login)


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
flags.DEFINE_boolean('info', True, 'print info on detections')
flags.DEFINE_boolean('crop', False, 'crop detections from images')
flags.DEFINE_boolean('plate', False, 'perform license plate recognition')

cred = credentials.Certificate("firefly-201c1-firebase-adminsdk-o0fuq-7f9a73e44b.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'firefly-201c1.appspot.com'
    })

db = firestore.client()
bucket = storage.bucket()
firebase = firebase.FirebaseApplication('https://firefly-201c1.firebaseio.com', None)

## get IP for stream
import socket
ip_local = (([ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")] or [[(s.connect(("8.8.8.8", 53)), s.getsockname()[0], s.close()) for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]]) + ["no IP found"])[0]
ip_local = '"\\' + "http://" + ip_local + ":5000" + '"'
ip_data =  { 
    'ip_data' : ip_local
}
try :
    result = firebase.patch('/camera/camera1',ip_data)
    print("IP Address Updated on firebase")
except :
    print("No internet, couldn't update database")


def send_push(threat_lvl, knife_counter, gun_counter):
    str_push = "Camera 1 Alert : Status " + str(threat_lvl) + ", " + str(knife_counter) + " Knife Detected, " + str(gun_counter) + " Gun Detected"
    notification_body = {
        'headings': {'en': 'Ancaman Terdeteksi'},
        'contents': {'en': str_push},
        'included_segments': ['Active Users']
        }
    APP_ID = "f8b1faca-8836-4a99-a057-18c8216fc37b"
    REST_API_KEY = "MjYxYzkxYjktN2VhNS00ZGFlLTkzZTktOTM0MzllYzVjZmU5"
    USER_AUTH_KEY = "ODFkZDBlMjctZjQ3OC00ZDAyLTlhY2EtYTI3NjFiYjgzMDkx"
    client1 = Client(app_id=APP_ID, rest_api_key=REST_API_KEY, user_auth_key=USER_AUTH_KEY)
    response1 = client1.send_notification(notification_body)

def send_firebase(data):
    global firebase
    try :
        result = firebase.patch('/camera/camera1',data)
    except :
        print ("failed to update database")
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
    if not streamer.is_streaming:
        streamer.start_streaming()
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
        
        fps = 1.0 / (time.time() - start_time)
        #print("FPS: %.2f" % fps)
        
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
        if threat_lvl > 0 and get_data_firebase == "1" and count_push == 1 :
            case_id = "caseID01" + str(int(time.time()))
            send_push(threat_lvl, knife_counter, gun_counter)
            get_data_firebase = "0"
            #try :
                #
            #except :
                #print("couldn't connect")
            try :
                firebase.patch('/camera/camera1', {'get_data_firebase' : get_data_firebase})
                count_push = 0
            except :
                print("failed to update get_data_firebase")
                count_push = 1
        if count_5 < 6 :
            gun_detector.append(gun_counter)
            knife_detector.append(knife_counter)
            totalobj_5f.append(total_object)
            if count_5 == 5 :
                gun_mode = mode(gun_detector)
                knife_mode = mode(knife_detector)
                totalobj_mode = mode(totalobj_5f)
                #maximum_gun = max(gun_detector)
                #maximum_knife = max(knife_detector)
                #maximum_obj = max(totalobj_5f)
                #maximum_gun = max(gun_mode)
                #maximum_knife = max(knife_mode)
                #maximum_obj = max(totalobj_mode)
                #print("gun mode = ",gun_mode)
                #print("knife mode = ",knife_mode)
                #print("total object mode = ",totalobj_mode)
                data =  { 
                    'gun_detect' : gun_mode,
                    'knife_detect' : knife_mode,
                    'jml_senjata' : totalobj_mode,
                    'lvl_threat' : threat_lvl,
                    'active_caseID' : case_id,
                    'jenis_terdeteksi' : weapon_type
                }
                    
                if totalobj_mode_ref != totalobj_mode :
                    totalobj_mode_ref = totalobj_mode
                    p1 = Process(target=send_firebase(data))
                    p1.start()
                        
                        
        else :
            count_5 = 0
            gun_detector.clear()
            knife_detector.clear()
            totalobj_5f.clear()
        
        #print ("frame count : ", frame_num)
        #print ("gun_counter  : ", gun_counter)
        #print ("knife _counter  : ", knife_counter)
        #print ("Total Object  : ", total_object)
        #print ("Level Ancaman : ", threat_lvl)
        
        #print (" ")
        #print ("count : ", count_5)
        
         ###### End firebase setup
            
           
        
        result = np.asarray(image)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if (total_object > 0) :
            #print ("frame count : ", frame_num)
            #print ("gun_counter  : ", gun_counter)
            #print ("knife _counter  : ", knife_counter)
            #print ("Total Object  : ", total_object)
            #print ("Level Ancaman : ", threat_lvl)
            #print ("Case id = ", case_id)
            print(" ")
            print(" ")
            global bucket
            now = datetime.now()
            dt_string = now.strftime("%d%m%Y_%H%M%S")
            os.chdir(os.path.abspath(os.getcwd()))
            filename = dt_string + ".jpg"
            cv2.imwrite(filename, result) 
            blob_db = "recorded/camera1" + filename
            blob = bucket.blob(blob_db)
            #ticker = threading.Event()
            try :
                blob.upload_from_filename(filename)
                url_link = '"' + '\\' + blob.public_url + '"'
                db_url = now.strftime("%d%m%Y") + "/" + case_id + "/" + str(int(time.time()))
                firebase.patch('/recorded/Camera1/',{db_url : url_link})
            except :
                print("failed to upload file")
            try :
                firebase_data = firebase.get('/camera/camera1', None)
                get_data_firebase = firebase_data.get('get_data_firebase', default="0")
                buzz_trigger = firebase_data.get('buzz_trigger', default=0)
                buzz_trigger.replace('"','')
                if get_data_firebase == "1":
                    count_push = 1
            except :
                print ("failed to retrieve data")
            if buzz_trigger == "ON" :
                try :
                    GPIO.output(buzzer,GPIO.HIGH) 
                except :
                    print("Buzzer ON but not responding due to not running on raspberry OS")
            elif buzz_trigger == "OFF" :
                try :
                    GPIO.output(buzzer,GPIO.LOW) 
                except :
                    print("Buzzer OFF but not responding due to not running on raspberry OS")
        streamer.update_frame(result)
        
            
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
