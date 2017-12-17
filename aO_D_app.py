print("Library loading....")
import time
startTime = time.time()
import os
import cv2
import argparse
import multiprocessing
import numpy as np
import tensorflow as tf
from omxplayer import OMXPlayer
import serial
import re
import threading

#from utils import FPS, WebcamVideoStream
from multiprocessing import Queue, Pool
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from picamera.array import PiRGBArray
from picamera import PiCamera

from datetime import datetime

import binascii
import struct
from bluepy.btle import UUID, Peripheral
print("Library loaded....")

DEVICE = "D7:C2:4A:3B:7E:A5"

led_service_uuid = UUID("ff00") #ffo3
led_char_uuid = UUID("ff04")

print("Serial connecting....")
ser = serial.Serial('/dev/ttyACM0',
               baudrate = 9600)
print("Serial connected....")
#print("arduino signal waiting...\n")        	
#state=ser.read()

'''print("BLE connecting....")
p = Peripheral(DEVICE, "random")
LedService=p.getServiceByUUID(led_service_uuid)

print (LedService)
ch = LedService.getCharacteristics(led_char_uuid)[0]
print(ch)
print("BLE connected")'''
# 유니코드 한글 시작 : 44032, 끝 : 55199
BASE_CODE, CHOSUNG, JUNGSUNG = 44032, 588, 28

# 초성 리스트. 00 ~ 18
CHOSUNG_LIST = ['8', '32,8', '9', '10', '32,10', '16', '17', '24', '32,24', '32', '32,32', '0', '40', '32,40', '48', '11', '19', '25', '26']

# 중성 리스트. 00 ~ 20
JUNGSUNG_LIST = ['35', '23', '28', '28,23', '14', '29', '49', '12', '37', '39', '39,23', '61', '44', '13', '15', '15,23', '13,23', '41', '42', '58', '21']

# 종성 리스트. 00 ~ 27 + 1(1개 없음)
JONGSUNG_LIST = ['0', '1', '32,1', '1,4', '18', '18,5', '18,52', '20', '2', '32,2', '2,34', '2,3', '2,4', '2,38', '2,50', '2,52', '34', '3', '3,4', '4', '12', '54', '5', '6', '22', '38', '50', '52']

# 초성 리스트. 00 ~ 18_ DOT
CHOSUNG_LIST_DOT = ['0x08', '0x20,0x8', '0x9', '0xA', '0x20,0x0A', '0x10', '0x11', '0x18', '0x20,0x18', '0x20', '0x20,0x20', '0x00', '0x28', '0x20,0x28', '0x30', '0x0B', '0x13', '0x19', '0x1A']

# 중성 리스트. 00 ~ 20_DOT
JUNGSUNG_LIST_DOT = ['0x23', '0x17', '0x1C', '0x1C,0x17', '0x0E', '0x1D', '0x31', '0x0C', '0x25', '0x27', '0x27,0x17', '0x3D', '0x2C', '0x0D', '0x0F', '0x0F,0x17', '0x0D,0x17', '0x29', '0x2A', '0x3A', '0x15']

# 종성 리스트. 00 ~ 27 + 1(1개 없음)_DOT
JONGSUNG_LIST_DOT = ['0x00', '0x01', '0x20,0x01', '0x01,0x04', '0x12', '0x12,0x05', '0x12,0x34', '0x14', '0x02', '0x02,0x01', '0x02,0x22', '0x02,0x03', '0x02,0x04', '0x02,0x26', '0x02,0x32', '0x02,0x34', '0x22', '0x03', '0x03,0x04', '0x04', '0x0C ', '0x36', '0x05', '0x06', '0x16', '0x26', '0x32', '0x34']



CWD_PATH = os.getcwd()
# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
PATH_TO_CKPT_coco = os.path.join(CWD_PATH, 'object_detection', MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', 'model_louis', 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS_coco = os.path.join(CWD_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'model_louis', 'object-detection.pbtxt')

NUM_CLASSES = 90

# Loading label map
label_map_coco = label_map_util.load_labelmap(PATH_TO_LABELS_coco)
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)

categories_coco = label_map_util.convert_label_map_to_categories(label_map_coco, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)

categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)

category_index_coco = label_map_util.create_category_index(categories_coco)
category_index = label_map_util.create_category_index(categories)

endTime = time.time()
print("\nINFO, library import time: ", (endTime - startTime),'\n')

def detect_objects(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Visualization of the results of a detection.

    data = vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    

    return data, image_np

def detect_objects_coco(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Visualization of the results of a detection.

    data = vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index_coco,
        use_normalized_coordinates=True,
        line_thickness=8)
    

    return data, image_np

def worker(input_q, output_q):
    startTime = time.time()

    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)
    endTime = time.time()
    print("\nINFO, loading Louis Tesnorflow model time: ", (endTime - startTime),'\n')

 #   fps = FPS().start()
    while True:
   #     fps.update()
        #startTime = time.time()

        frame_mod = input_q.get()
        output_q.put(detect_objects(frame_mod, sess, detection_graph))
        #endTime = time.time()
        #print("\nINFO, Object Detection time: ", (endTime - startTime),'\n')

  #  fps.stop()
    sess.close()

def worker_coco(input_q, output_q):
    startTime = time.time()

    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT_coco, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)
    endTime = time.time()
    print("\nINFO, loading  coco Tesnorflow model time: ", (endTime - startTime),'\n')

 #   fps = FPS().start()
    while True:
   #     fps.update()
        #startTime = time.time()

        frame_mod = input_q.get()
        output_q.put(detect_objects_coco(frame_mod, sess, detection_graph))
        #endTime = time.time()
        #print("\nINFO, coco Object Detection time: ", (endTime - startTime),'\n')

  #  fps.stop()
    sess.close()

def voice_object_Out(voice):
    player = OMXPlayer(os.path.join(CWD_PATH,'voice','guide_voice','this.mp3'))
    time.sleep(0.8)
    player.quit()
    player = OMXPlayer(os.path.join(CWD_PATH,'voice','object_voice',voice+'.mp3'))
    time.sleep(1.0)
    player.quit()
    player = OMXPlayer(os.path.join(CWD_PATH,'voice','guide_voice','is.mp3'))
    time.sleep(1.0)
    player.quit()

def voice_guide_Out(voice):
    player = OMXPlayer(os.path.join(CWD_PATH,'voice','guide_voice',voice +'.mp3'))
    time.sleep(5)
    #player.stop()
    #player.quit()

def send_arduino(name):
    ser.write(bytes(name.encode('ascii')))
   
def signal_wait_arduino():
    while True :
        print("waiting...\n")
        state=ser.read()
        if state == b'':
            print("trash value..")
            continue
        elif state == b'f':
            print(state)
            break

def Dotwatch_Ble(name) :
    print("BLE connecting....")
    p = Peripheral(DEVICE, "random")
    LedService=p.getServiceByUUID(led_service_uuid)

    print (LedService)
    ch = LedService.getCharacteristics(led_char_uuid)[0]
    print(ch)
    print("BLE connected")
    t = threading.currentThread()
    while getattr(t,"do_run",True) :
        ch.write(struct.pack('BBB',0x25,0x01,0x00)) # all up
        time.sleep(0.5)
        for char in name :
            #ch.write(struct.pack('BBBB', 0x0A,0x02,0x01,0x00))
            
            #print(type(char[0]))
            ch.write(struct.pack('BBBBBBBB', 0x06,0x12,0x01,0x01,char[0],char[1],char[2],char[3]))
            #print ("write : ",binascii.b2a_hex(ch.read()))
            if not getattr(t,"do_run",True):
                break;
            time.sleep(8)
    #finally:
        #continue
        
    print("stop print dotwotch")
    ch.write(struct.pack('BBB',0x26,0x01,0x00)) # all down
    p.disconnect()
    print("BLE disconnected")

def KO_split_DOT(name) :
    test_keyword = name
    split_keyword_list = list(test_keyword)
    print(split_keyword_list)
    result = list()
    for keyword in split_keyword_list:
        # 한글 여부 check 후 분리
        if re.match('.*[ㄱ-ㅎㅏ-ㅣ가-힣]+.*', keyword) is not None:
            char_code = ord(keyword) - BASE_CODE #ord는 unicode로 바꿔주는것
            char1 = int(char_code / CHOSUNG)
            result.append(CHOSUNG_LIST_DOT[char1])
            result.append(",")
            print('초성 : {}'.format(CHOSUNG_LIST_DOT[char1]))
            char2 = int((char_code - (CHOSUNG * char1)) / JUNGSUNG)
            result.append(JUNGSUNG_LIST_DOT[char2])
            result.append(",")
            print('중성 : {}'.format(JUNGSUNG_LIST_DOT[char2]))
            char3 = int((char_code - (CHOSUNG * char1) - (JUNGSUNG * char2)))
            result.append(JONGSUNG_LIST_DOT[char3])
            result.append(";")
            print('종성 : {}'.format(JONGSUNG_LIST_DOT[char3]))
        else:
            result.append(keyword)
        

    result=result[0:len(result)-1] # 마지막 , 는 없앰
    
    #if(result.size()<0) :
    #print (result)
    # result
    a="".join(result)
    #print(a)
    b= a.split(';')
    #print(b) # b return maybe !!!!
    for i in range (0,len(b)):
        b[i] = b[i].split(',')
        if(len(b[i])!=4) :
            b[i].append('0x00')
        b[i][0]= tohex(b[i][0])
        b[i][1]= tohex(b[i][1])
        b[i][2]= tohex(b[i][2])
        b[i][3]= tohex(b[i][3])
        #print(type(b[i]))
    print(b)
    return b

def tohex(hex_str):
    hex_int = int(hex_str, 16)
    #print(type(hex_int))
    #new_int = hex_int + 0x20  0
    #print (hex(hex_int))
    return hex_int

def KO_spilt(name) :
    test_keyword = name
    split_keyword_list = list(test_keyword)
    print(split_keyword_list)
    result = list()
    typesize=list()
    typesize.append("0x07")
    for keyword in split_keyword_list:
        # 한글 여부 check 후 분리
        if re.match('.*[ㄱ-ㅎㅏ-ㅣ가-힣]+.*', keyword) is not None:
            char_code = ord(keyword) - BASE_CODE #ord는 unicode로 바꿔주는것
            char1 = int(char_code / CHOSUNG)
            result.append(CHOSUNG_LIST[char1])
            result.append(",")
            print('초성 : {}'.format(CHOSUNG_LIST[char1]))
            char2 = int((char_code - (CHOSUNG * char1)) / JUNGSUNG)
            result.append(JUNGSUNG_LIST[char2])
            result.append(",")
            print('중성 : {}'.format(JUNGSUNG_LIST[char2]))
            char3 = int((char_code - (CHOSUNG * char1) - (JUNGSUNG * char2)))
            result.append(JONGSUNG_LIST[char3])
            result.append(";")
            print('종성 : {}'.format(JONGSUNG_LIST[char3]))
        else:
            result.append(keyword)

    result=result[0:len(result)-1]
    result.append("!")
    typesize.append(str(len("".join(result))))
    
    # result
    print("".join(typesize+result))
    return ("".join(typesize+result))

def translate(name):
    if name == "person":
        answer = "사람"
    elif name == "bicycle":
        answer = "자전거"
    elif name == "car":
        answer = "자동차"
    elif name == "motorcycle":
        answer = "오토바이"
    elif name == "airplane":
        answer = "비행기"
    elif name == "bus":
        answer = "버스"
    elif name == "train":
        answer = "기차"
    elif name == "truck":
        answer = "트럭"
    elif name == "boat":
        answer = "배"
    elif name == "traffic light":
        answer = "신호등"
    elif name == "fire hydrant":
        answer = "소화전"
    elif name == "stop sign":
        answer = "정지표시판"
    elif name == "parking meter":
        answer = "주차료증수기"
    elif name == "bench":
        answer = "벤치"
    elif name == "bird":
        answer = "새"
    elif name == "cat":
        answer = "고양이"
    elif name == "dog":
        answer = "개"
    elif name == "horse":
        answer = "말"
    elif name == "sheep":
        answer = "양"
    elif name == "cow":
        answer = "소"
    elif name == "elephant":
        answer = "코끼리"
    elif name == "bear":
        answer = "곰"
    elif name == "zebra":
       answer = "얼룩말"
    elif name == "giraffe":
        answer = "기린"
    elif name == "backpack":
        answer = "가방"
    elif name == "umbrella":
        answer = "우산"
    elif name == "handbag":
        answer = "핸드백"
    elif name == "tie":
        answer = "넥타이"
    elif name == "suitcase":
        answer = "여행가방"
    elif name == "frisbee":
        answer = "원반"
    elif name == "skis":
        answer = "스키"
    elif name == "snowboard":
        answer = "스노우보드"
    elif name == "sports ball":
        answer = "공"
    elif name == "kite":
        answer = "연"
    elif name == "baseball bat":
        answer = "야구배트"
    elif name == "baseball glove":
        answer = "야구글러브"
    elif name == "skateboard":
        answer = "스케이드보드"
    elif name == "surfboard":
        answer = "서핑보드"
    elif name == "tennis racket":
        answer = "테니스라켓"
    elif name == "bottle":
        answer = "물병"
    elif name == "wine glass":
        answer = "와인잔"
    elif name == "cup":
        answer = "물컵"
    elif name == "fork":
        answer = "포크"
    elif name == "knife":
        answer = "나이프"
    elif name == "spoon":
        answer = "숟가락"
    elif name == "bowl":
        answer = "그릇"
    elif name == "banana":
        answer = "바나나"
    elif name == "apple":
        answer = "사과"
    elif name == "sandwich":
        answer = "샌드위치"
    elif name == "orange":
        answer = "오렌지"
    elif name == "broccoli":
        answer = "브로콜리"
    elif name == "carrot":
        answer = "당근"
    elif name == "hot dog":
        answer = "핫도그"
    elif name == "pizza":
        answer = "피자"
    elif name == "donut":
        answer = "도넛"
    elif name == "cake":
        answer = "케이크"
    elif name == "chair":
        answer = "의자"
    elif name == "couch":
        answer = "소파"
    elif name == "potted plant":
        answer = "화분"
    elif name == "bed":
        answer = "침대"
    elif name == "dining table":
        answer = "식탁"
    elif name == "toilet":
        answer = "화장실"
    elif name == "tv":
        answer = "티비"
    elif name == "laptop":
        answer = "노트북"
    elif name == "mouse":
        answer = "마우스"
    elif name == "remote":
        answer = "리모콘"
    elif name == "keyboard":
        answer = "키보드"
    elif name == "cell phone":
        answer = "핸드폰"
    elif name == "smartphone":
        answer = "핸드폰"
    elif name == "microwave":
        answer = "전자레인지"
    elif name == "oven":
        answer = "오븐"
    elif name == "toaster":
        answer = "토스트기"
    elif name == "sink":
        answer = "싱크대"
    elif name == "refrigerator":
        answer = "냉장고"
    elif name == "book":
        answer = "책"
    elif name == "clock":
        answer = "시계"
    elif name == "watch":
        answer = "시계"
    elif name == "vase":
        answer = "꽃병"
    elif name == "scissors":
        answer = "가위"
    elif name == "teddy bear":
        answer = "곰인형"
    elif name == "hair drier":
        answer = "헤어드라이기"
    elif name == "toothbrush":
        answer = "칫솔"
    elif name == "lemon":
        answer = "레몬"
    elif name == "glasses":
        answer = "안경"
    elif name == "coke":
        answer = "콜라"
    elif name == "cider":
        answer =  "사이다"
    elif name =="pen":
        answer = "펜"
    elif name == "ebi":
        answer = "에비츄"
    return answer    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    #parser.add_argument('-src', '--source', dest='video_source', type=int,
      #                  default=0, help='Device index of the camera.')
   # parser.add_argument('-wd', '--width', dest='width', type=int,
     #                   default=480, help='Width of the frames in the video stream.')
   # parser.add_argument('-ht', '--height', dest='height', type=int,
   #                     default=360, help='Height of the frames in the video stream.')
    parser.add_argument('-w', '--num-workers', dest='num_workers', type=int,
                        default=1, help='Number of workers.')
    parser.add_argument('-q', '--queue-size', dest='queue_size', type=int,
                        default=2, help='Size of the queue.')
    args = parser.parse_args()

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)
    
    input_q = Queue(maxsize=args.queue_size)
    input_q_coco = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)
    output_q_coco = Queue(maxsize=args.queue_size)
    pool = Pool(args.num_workers, worker, (input_q, output_q))
    pool_coco = Pool(args.num_workers, worker_coco, (input_q_coco, output_q_coco))


    #video_capture = WebcamVideoStream(src=args.video_source,
      #                                width=args.width,
        #                              height=args.height).start()
    #fps = FPS().start()
    camera = PiCamera()
    camera.resolution = (320, 320)
    camera.framerate = 1
    rawCapture = PiRGBArray(camera, size=(320, 320))
    time.sleep(0.1) 	
    count=0
    '''camera.capture(raw)
    input_q.put(raw)
    data, image = output_q.get()   	
    cv2.imshow("Frame", image)
    
    while True :
        print("arduino signal waiting...\n")
        time.sleep(2)
        state=ser.read()
        print(state)
        if state == b's' :     
                             
            break
        else :
            print("trash value : ", state)
            continue
    voice_guide_Out("let")
    '''
    start = False
    try:
      for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        	# grab the raw NumPy array representing the image, then initialize the timestamp
        	# and occupied/unoccupied text

            detect = False
            startTime = time.time()
            frame = frame.array
            input_q.put(frame)
            t = time.time()
            data, image = output_q.get()   	
            '''
            im2=cv2.resize(image,(480,480),interpolation=cv2.INTER_CUBIC)
            cv2.imshow("Frame", im2)
            cv2.waitKey(1) # find this bug     
            '''
        	# show the frame
            #im2 = cv2.resize(image, (600,600))
            '''
            im2=cv2.resize(image,(480,480),interpolation=cv2.INTER_CUBIC)
            cv2.imshow("Frame", im2)
            cv2.waitKey(1) # find this bug
            '''
            #if cv2.waitKey(1) & 0xFF == ord("q"):
            #        break
            
            #input_q_coco.put(frame)
            #data, image = output_q_coco.get()  
            while start == False :
                print("coco model loading...\n")
                #time.sleep(3)
                input_q_coco.put(frame)
                output_q_coco.get()
                voice_guide_Out("start")
                print("arduino signal waiting...\n")        	
                time.sleep(3)
                state=ser.read()
                print(state)
                if state == b's' :
                    voice_guide_Out("let")
                    start =True                                   
                    break
                else :
                    print("trash value : ", state)
                    continue
            
            '''      
            im2=cv2.resize(image,(480,480),interpolation=cv2.INTER_CUBIC)
            cv2.imshow("Frame", im2)
            cv2.waitKey(1) # find this bug
            '''
        	# clear the stream in preparation for the next frame
            rawCapture.truncate(0)
            
        	# if the `q` key was pressed, break from the loop
            
            if not data :
                #print("INFO, no Object...in __our model__\n")
                input_q_coco.put(frame)
                data, image = output_q_coco.get()
                '''im2=cv2.resize(image,(480,480),interpolation=cv2.INTER_CUBIC)
                cv2.imshow("Frame", im2)
                cv2.waitKey(1) # find this bug
                '''
                #for indata in data :  
                if not data :
                    print("INFO, no Object...\n")
                    count= count+1
                    if count == 5:
                        voice_guide_Out("change")
                        send_arduino("0x08") # wait signal
                        while True:
                            print("arduino signal waiting...\n")                            
                            voice_guide_Out("start")
                            time.sleep(3)
                            state=ser.read()
                            print(state)
                            if state == b'w' :
                                voice_guide_Out("let")                                   
                                break
                            else :
                                print("trash value : ", state)
                                continue
                                
                        count =0
                        # need test
                    time.sleep(2)
                
                else:
                    detect =True
            
            else:
                detect =True

            if detect ==True :
                endTime = time.time()
                print("\nINFO, Detection Succed")
                print("INFO, realtime Detection time: ", (endTime - startTime))
                

                #cv2.imshow("Frame", im2)
                #cv2.waitKey(1) # find this bug
                #print (totaldata)
                count =0
                for indata in data :
                    im2=cv2.resize(image,(480,480),interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite("./pic/"+datetime.today().strftime("%Y%m%d%H%M%S")+indata[0]+".jpg",im2)
                    
                    print("\nname: ", indata[0], " score: ",round(indata[1],2))
                    print("xmin: {0:.2f}".format(indata[2]), " xmax: {0:.2f}".format(indata[3]) )
                    print("ymin: {0:.2f}".format(indata[4]), " ymax: {0:.2f}".format(indata[5]) )
                    print("Location : x:{0:.2f}".format((indata[3]+indata[2])/2), "y:{0:.2f}".format((indata[5]+indata[4])/2) )        
                    print("")                      
     
                    voice_object_Out(indata[0])
                    send_arduino(KO_spilt(translate(indata[0])))
                    t = threading.Thread(target =Dotwatch_Ble, args = (KO_split_DOT(translate(indata[0])), ))
                    t.do_run = True
                    t.start()
                    while True :
                        print("arduino signal waiting...\n")
                        state=ser.read()
                        print(state)
                        
                   
                        if state == b'f' :
                            #cv2.destroyAllWindows()                    
                        #if cv2.waitKey(1) & 0xFF == ord("r"):
                            print("restart...\n")
                            t.do_run = False
                            #t.join()
                            print(state)
                            voice_guide_Out("restart")
                            break
                        else :
                            print("trash value : ", state)
                            continue
                    

    except KeyboardInterrupt:
        pass
              
            
        #continue
pool.terminate()
print("INFO, LOUIS SHUT DOWN")
#p.disconnect()
#print("BLE disconnected")

#video_capture.stop()
#cv2.destroyAllWindows()
