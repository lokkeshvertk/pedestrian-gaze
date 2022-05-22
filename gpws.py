# dependencies
import random
import numpy as np
import os
from time import time, sleep
import cv2
from copy import copy

from threading import Thread
from queue import Queue
from multiprocessing import Process
from multiprocessing import Queue as MQueue
from multiprocessing import Manager

import torch
from torchvision import transforms, datasets

from darknet import darknet
from monodepth2 import networks
from monodepth2.layers import disp_to_depth
from monodepth2.evaluate_depth import STEREO_SCALE_FACTOR
from tobiiglassesctrl import TobiiGlassesController

alert = 0

def init_tobii():
    #Create Participant data
    project_name = str(input("Enter project name "))
    participant_name = str(input("Enter participant name "))
    project_id = tobiiglasses.create_project(project_name) # Function call to create a project id
    participant_id = tobiiglasses.create_participant(project_id, participant_name) # Function call to create a participant id

    #Set the gaze and video frequencies
    # gaze_fps = int(input("Enter the gaze frequency "))
    # if gaze_fps == 100:
    #     tobiiglasses.set_et_freq_100()
    # else:
    tobiiglasses.set_et_freq_50()

    #Set video frequency
    #video_fps = int(input("Enter the video frequency "))
    #if video_fps == 50:
    tobiiglasses.set_video_freq_50()
    # else:
    #     tobiiglasses.set_video_freq_25()

    return project_id, participant_id

def init_darknet():
    config_file = "/home/tklokkeshver/MasterThesis/darknet/cfg/yolov4-tiny-obj.cfg"
    data_file = "/home/tklokkeshver/MasterThesis/darknet/build/darknet/x64/data/obj.data"
    weights = "/home/tklokkeshver/MasterThesis/darknet/build/darknet/x64/backup/tiny_new/yolov4-tiny-obj_best.weights"
    batch_size = 1
    obj_net, cls_names, _ = darknet.load_network(config_file, data_file, weights, batch_size)
    cls_colours = { 'Car': (234, 243, 107), 'Van': (242, 59, 230), 'Truck': (75, 246, 235), 'Pedestrian': (10, 180, 243), 'Person_sitting': (202, 42, 116), 'Cyclist': (255,43,10), 'Tram': (255,255,255), 'Misc': (255,159,233)}
    print("\n Darknet....loading....DONE")
    #return config_file, data_file, weights, batch_size
    return obj_net, cls_names, cls_colours

def init_depth():
    model_name = "mono+stereo_640x192"
    model_path = os.path.join("/home/tklokkeshver/MasterThesis/monodepth2/models",model_name)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("loading network")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=DEVICE)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(DEVICE)
    encoder.eval()

    print("Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=DEVICE)
    depth_decoder.load_state_dict(loaded_dict)
    depth_decoder.to(DEVICE)
    depth_decoder.eval()
    return feed_height, feed_width, encoder, depth_decoder


def tobii_calib(project_id, participant_id):
    calibration_id = tobiiglasses.create_calibration(project_id, participant_id) # Function call to create a calibration id for the participant
    input("Put the calibration marker in front of the user, then press enter to calibrate")
    tobiiglasses.start_calibration(calibration_id) # Starts the calibration process
    gaze_status = tobiiglasses.wait_until_calibration_is_done(calibration_id) #Check for calibration
    return gaze_status

def get_gaze(ts_video, pts_data, gaze_data):
    video_pts = ts_video*90000                    # Calculate the PTS of video
    gaze_pts = pts_data[u'pts']			  		  # Get the value of gaze PTS
    diff = (gaze_pts - video_pts)*1e-6		      # Calculate the offset between gaze PTS and Video PTS
    gx = 0
    gy = 0
    gaze = [0,0]
    if (abs(diff) < 0.05756):
        gaze = gaze_data[u'gp']
    return gaze

def video_capture(frame_queue, darknet_image_queue, depth_image_queue, current_frame_queue, gaze_queue):
    while (vid.isOpened()):
        data_gp  = tobiiglasses.get_data()['gp']
        if (data_gp['ts'] > 0):
            ret, frame = vid.read()
            if ret == True:
                video_ts = (vid.get(cv2.CAP_PROP_POS_MSEC))*1e-3
                data_pts = tobiiglasses.get_data()['pts']
                gaze = get_gaze(video_ts, data_pts, data_gp)
                gaze_queue.put(gaze)
                frame_queue.put(frame)
                cur_frame = cv2.resize(frame, (240, 135), interpolation=cv2.INTER_LINEAR)
                current_frame_queue.put(cur_frame)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                depth_resized = cv2.resize(frame_rgb, (feed_width, feed_height), interpolation= cv2.INTER_LANCZOS4)
                depth_resized = transforms.ToTensor()(depth_resized).unsqueeze(0)
                depth_image_queue.put(depth_resized)
                frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height), interpolation=cv2.INTER_LINEAR)
                darknet_image_queue.put(frame_resized)
                #cv2.imshow('frame', frame)
                #if cv2.waitKey(1) & 0xFF == ord('q'):
            else:
                break
    vid.release()
    cv2.destroyAllWindows()


def yolo_network(input_image):
    img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
    darknet.copy_image_from_bytes(img_for_detect, input_image.tobytes())
    detections = darknet.detect_image(dark_network, class_names, img_for_detect, thresh=0.65)
    #darknet.print_detections(detections, ext_output)
    darknet.free_image(img_for_detect)
    return detections

def depth_network(input_image):
    with torch.no_grad():
    # PREDICTION
        input_image = input_image.to(DEVICE)
        features = depth_encoder(input_image)
        outputs = depth_decoder(features)

        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(disp, (135, 240), mode="bilinear", align_corners=False)
        scaled_disp, depth = disp_to_depth(disp_resized, 0.1, 100)
        pred_depth = depth[0,0,:,:].cpu().numpy()
        metric_depth = STEREO_SCALE_FACTOR * pred_depth
        return metric_depth

def inference(darknet_image_queue, depth_image_queue, current_frame_queue, print_detection_queue, print_moving_object_queue, print_fps_queue, gaze_queue, print_gaze_queue, print_vehicle_status_queue, print_alert_status_queue): #changed
    i = 0
    while True:
        darknet_image = darknet_image_queue.get()
        depth_image = depth_image_queue.get()
        current_frame = current_frame_queue.get()
        gaze = gaze_queue.get()
        prev_time = time()
        depth = depth_network(depth_image)
        detections = yolo_network(darknet_image)
        vehicle_status = "No Vehicle found"
        attention = "Not checked"
        alert_value = 0
        moving_detections = []
        if i != 0:
            moving_detections, vehicle_status, attention, alert_value = moving_objects(previous_frame, current_frame, previous_detections, detections, depth, gaze,i)
        previous_frame = copy(current_frame)
        previous_detections = copy(detections)
        print_detection_queue.put(detections)
        print_moving_object_queue.put(moving_detections) #changed
        print_vehicle_status_queue.put(vehicle_status)
        print_alert_status_queue.put([attention, alert_value])
        print_gaze_queue.put(gaze)
        fps = int(1/(time() - prev_time))
        print_fps_queue.put(fps)
        i +=1
        #print("FPS: {}".format(fps))
    #vid.release()

def convert2original(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_x       = int(x * image_w)
    orig_y       = int(y * image_h)
    orig_width   = int(w * image_w)
    orig_height  = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted

def convertdetections(bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w = [540,960]

    orig_x       = int(x * image_w)
    orig_y       = int(y * image_h)
    orig_width   = int(w * image_w)
    orig_height  = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted

def convert2relative(bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h  = bbox
    _height     = darknet_height
    _width      = darknet_width
    return x/_width, y/_height, w/_width, h/_height

def convert4cropping(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_left    = int((x - w / 2.) * image_w)
    orig_right   = int((x + w / 2.) * image_w)
    orig_top     = int((y - h / 2.) * image_h)
    orig_bottom  = int((y + h / 2.) * image_h)

    if (orig_left < 0): orig_left = 0
    if (orig_right > image_w - 1): orig_right = image_w - 1
    if (orig_top < 0): orig_top = 0
    if (orig_bottom > image_h - 1): orig_bottom = image_h - 1

    bbox_cropping = (orig_left, orig_top, orig_right, orig_bottom)

    return bbox_cropping

def bbox2points(bbox):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """
    x, y, w, h = bbox
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def print_detections(detections, coordinates=False):
    detections_resized = convertdetections(detections)
    print("\nObjects:")
    for label, confidence, bbox in detections_resized:
        x, y, w, h = bbox
        if coordinates:
            print("{}: {}%    (left_x: {:.0f}   top_y:  {:.0f}   width:   {:.0f}   height:  {:.0f})".format(label, confidence, x, y, w, h))
        else:
            print("{}: {}%".format(label, confidence))

def match_pts(img1, img2):
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=50)
    kp1, des1 = orb.detectAndCompute(g1, None)
    kp2, des2 = orb.detectAndCompute(g2, None)

    if des1 is None and des2 is None:
        new_p1 = np.zeros((8,2))
        new_p2 = np.zeros((8,2))
        status = "Fail"

    else:
        #Find Matches in two frames
        bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=False)
        try:
            matches = bf.match(des1,des2)
            matches = sorted(matches, key=lambda x: x.distance)
            mat = np.array(matches[:8])
            #match_img = cv2.drawMatches(g1, kp1, g2, kp2, matches[:8], None)

            #plt.figure(figsize=(20,20))
            #plt.imshow(match_img)

            #Point matches
            list_kp1 = []
            list_kp2 = []

            # For each match...
            for mat in matches:

            # Get the matching keypoints for each of the images
                img1_idx = mat.queryIdx
                img2_idx = mat.trainIdx

            # x - columns
            # y - rows
            # Get the coordinates
                (x1, y1) = kp1[img1_idx].pt
                (x2, y2) = kp2[img2_idx].pt

            # Append to each list
                list_kp1.append((x1, y1))
                list_kp2.append((x2, y2))

            p1 = np.array(list_kp1)
            p2 = np.array(list_kp2)

            new_p1 = []
            new_p2 = []
            for i in range(len(p1)):
                if p2[i][1]-p1[i][1] < 3 and p2[i][1]-p1[i][1] > -3:
                    new_p1.append(p1[i])
                    new_p2.append(p2[i])

            new_p1 = np.asarray(new_p1)
            new_p2 = np.asarray(new_p2)
            status = "Pass"
        except cv2.error:
            new_p1 = np.zeros((8,2))
            new_p2 = np.zeros((8,2))
            status = "Fail"      
    return new_p1, new_p2, status

def rt_matrix(p1, p2):
    # camera parameters
    fx = 1137.67
    fy = 1136.93
    cx = 936.158
    cy = 559.406
    camera = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    if sum(sum(p1)) != 0 and sum(sum(p2)) != 0:

        # Calculate Essential Matrix
        E, mask = cv2.findEssentialMat(p1,p2,camera,cv2.RANSAC, prob=0.99, threshold=2)
        if E is None:
            R = np.identity(3)
            T = np.zeros(3)
            return R, T
        # Calculate Rotation and Translation matrix
        try:
            U, S, Vt = np.linalg.svd(E)
            W = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]) #changed
            try:
                R = U.dot(W).dot(Vt) # check
                T = U[:, 2]
            except ValueError:
                R = np.identity(3)
                T = np.zeros(3)
        except np.linalg.linalg.LinAlgError:
            R = np.identity(3)
            T = np.zeros(3)
    else:
        R = np.identity(3)
        T = np.zeros(3)
    return R, T  

def masks(img, detections):
    mask = np.zeros((img.shape[0], img.shape[1]))
    masked_img = img.copy()
    label = detections[0]
    bbox = detections[2]
    if label == 'Car' or label == 'Van' or label == 'Truck':
        l, t, r, b = convert4cropping(img, bbox)
        mask[t:b, l:r] = 1
    masked_img[mask != 1] = 0
    return masked_img

def check_detections(d1,d2):
    f1 = d1.copy()
    f2 = d2.copy()
    if (len(f1) > len(f2)):
        conf = []
        for k in range(len(f1)):
            conf.append(f1[k][1])
        sort_idx = np.argsort(conf)
        idx = sort_idx[0]
        f1.pop(idx)
    else:
        conf = []
        for l in range(len(f2)):
            conf.append(f2[l][1])
        sort_idx = np.argsort(conf)
        idx = sort_idx[0]
        f2.pop(idx)
    return f1, f2

def match_detections(d1,d2):
    od2 = d2.copy()
    sort_d2 = []
    if len(d1) == len(d2):
        nd1 = d1.copy()
        for i in range(len(d1)):
            bbox1 = d1[i][2]
            p1 = np.array((bbox1[0],bbox1[1]))
            diff = []
            for j in range(len(od2)):
                bbox2 = od2[j][2]
                p2 = np.array((bbox2[0],bbox2[1]))
                diff.append(np.linalg.norm(p2-p1))
            sort_idx = np.argsort(diff)
            idx = sort_idx[0]
            sort_d2.append(od2.pop(idx))
        nd2 = sort_d2.copy()
    else:
        rdetect1, rdetect2 = check_detections(d1, d2)
        nd1, nd2 = match_detections(rdetect1,rdetect2)
    return nd1, nd2

def approaching(p_det, c_det):
    for i in range(len(c_det)):
        bbox1 = p_det[i][2]
        bbox2 = c_det[i][2]
        x1,y1,_,_ = bbox1
        x2,y2,_,_ = bbox2
        approaching_vehicle_previousframe = []
        approaching_vehicle_currentframe = []
        # if (x1 < 208):
        if (x1-x2) < 0:
            approaching_vehicle_previousframe.append(p_det[i])
            approaching_vehicle_currentframe.append(c_det[i])
        elif (x1-x2) > 0:
            approaching_vehicle_previousframe.append(p_det[i])
            approaching_vehicle_currentframe.append(c_det[i])

    return approaching_vehicle_previousframe ,approaching_vehicle_currentframe

def priority(detections):
    distances = []
    for i in range(len(detections)):
        bbox = detections[i][2]
        x,y,w,h = bbox
        c = np.array([240,135])
        p = np.array([x,y])
        distances.append(np.linalg.norm(c-p))
    sort_idx = np.argsort(distances)
    idx = sort_idx[0]
    priority = detections[0]
    return priority

def beep():
    print("\a")

def alert_check(gaze, movingobj_detections, frame_no):
    duration = 0.01  # seconds
    freq = 2000  # Hz
    gx = gaze[0] * darknet_width
    gy = gaze[1] * darknet_height
    alert_status = "No alert"
    global alert
    bbox = movingobj_detections[2]
    left,top,right,bottom = bbox
    if (gx > left and gx < right):
        if (gy < top and gy > bottom):
            alert_status = "No Alert"
    else:
        alert += 1
    # if (frame_no % 5 == 0):
    if (alert % 15 == 0):
        alert_status = "Risk"
        beep()
        #os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
        #alert = 0
    return alert_status, alert


def moving_objects(previous_frame, current_frame, prev_detections, current_detections, depth, gaze, frame_no): #previous_frame_queue, current_frame_queue, previous_detections_queue, current_detections_queue, depth_queue, gaze_queue, print_detection_queue, print_gaze_queue):

    img1 = previous_frame
    img2 = current_frame
    od1 = prev_detections
    od2 = current_detections

    moving_object_detections = []
    vehicle_status = []
    vehicle_status.append("No Vehicle found")
    alert_st = "Not Checked"
    alert_value = 0
    
    if (len(prev_detections) != 0) and (len(current_detections) != 0):
        d1,d2 = match_detections(od1,od2)
        pd,cd = approaching(d1,d2)
        if len(cd) == 0:
            vehicle_status[0] = "Non-Moving Vehicle"
            return moving_object_detections, vehicle_status, alert_st, alert_value
        px,py, status_check = match_pts(img1, img2)
        if status_check == "fail": return moving_object_detections, vehicle_status, alert_st, alert_value
        R,T = rt_matrix(px, py)


        # camera parameters for the perspective matrix
        fx = 1137.67
        fy = 1136.93
        cx = 936.158
        cy = 559.406
        perspective_matrix = np.array([[fx, 0, cx,0], [0, fy, cy,0], [0, 0, 1,0]])

        # check for moving obejcts based on detections in a frame
        
        for i in range(len(cd)):
            masked_img1 = masks(img1, pd[i])
            masked_img2 = masks(img2, cd[i])
            p0, p1, _ = match_pts(masked_img1, masked_img2)
            difference = []
        #calculate estimated p0
            for j in range(len(p1)):
                u = int(p1[j][1])
                v = int(p1[j][0])
                zcam = depth[u,v] * 1000
                xcam = (zcam/fx) * (u-cx)
                ycam = (zcam/fy) * (v-cy)
                p_3d = np.array([xcam, ycam, zcam])
                pdash_3d = np.matmul(R,p_3d - T)
                pdash_3d = np.insert(pdash_3d,3,1)
                pdash_3d = pdash_3d.reshape(4,1)
                pdash_2d = np.matmul(perspective_matrix, pdash_3d)
                pdash_2d = pdash_2d/(pdash_2d[2])
                difference.append(np.linalg.norm(p0[j]-pdash_2d[:2]))
                #print(difference)
            mean = float(0)
            if len(difference) !=0: mean = float(np.mean(difference))
            #print(mean)
            threshold = 1000
            max_threshold = 3000
            if threshold < mean < max_threshold:
                moving_object_detections.append(cd[i])
                if i == 0: vehicle_status[i] = "Moving Vehicle" + " " + "--" +" " + str(mean)
                else: vehicle_status.append("Moving Vehicle" + " " + "--" +" " + str(mean))
            else:
                if i == 0: vehicle_status[i] = "Non-Moving Vehicle"+ " "  + "--" +" " + str(mean)
                else: vehicle_status.append("Non-Moving Vehicle"+ " "  + "--" +" " + str(mean))
    else:
        vehicle_status[0] = "No data"

    if len(moving_object_detections) != 0:
        priority_vehicle = priority(moving_object_detections)
        alert_st, alert_value = alert_check(gaze, priority_vehicle, frame_no)
    return moving_object_detections, vehicle_status, alert_st, alert_value

def draw_boxes(frame, detections, image, colors):
    for label, confidence, bbox in detections:
        label = str(label)
        bbox_adjusted = convert2original(frame, bbox)
        left, top, right, bottom = bbox2points(bbox_adjusted)
        cv2.rectangle(image, (left, top), (right, bottom), colors[label], 3)
        cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                    (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colors[label], 2)
    return image

def draw_risk_boxes(frame, detections, image):
    for label, confidence, bbox in detections:
        label = str(label)
        bbox_adjusted = convert2original(frame, bbox)
        left, top, right, bottom = bbox2points(bbox_adjusted)
        cv2.rectangle(image, (left, top), (right, bottom), (0,0,255), 5)
    return image

def show_video(frame_queue, print_detection_queue, print_moving_object_queue, print_fps_queue, print_gaze_queue, print_vehicle_status_queue, print_alert_status_queue): #changed
    random.seed(3)
    while True:
        frame = frame_queue.get()
        detections = print_detection_queue.get()
        moving_vehicles = print_moving_object_queue.get() #changed
        fps = print_fps_queue.get()
        gaze = print_gaze_queue.get()
        v_status = print_vehicle_status_queue.get()
        a_status = print_alert_status_queue.get()
        darknet.print_detections(detections, ext_output)
        print("------------------------------------------------------------------------")
        print("Gaze:  ", gaze)
        print("------------------------------------------------------------------------")
        print("Vehicle status:  ", v_status)
        print("------------------------------------------------------------------------")
        print("Attention status:  ", a_status)
        print("------------------------------------------------------------------------")
        gx = gaze[0]
        gy = gaze[1]
        image_frame = draw_boxes(frame, detections, frame, class_color)
        if len(moving_vehicles) != 0: image_frame = draw_risk_boxes(frame, moving_vehicles, frame) #changed
        if a_status[0] == "Risk":
            image_frame = cv2.line(image_frame, (0,0),(960,0), (0,0,255),10) #changed
            image_frame = cv2.line(image_frame, (0,540),(960,540), (0,0,255),10) #changed
            image_frame = cv2.line(image_frame, (0,0),(0,540), (0,0,255),10) #changed
            image_frame = cv2.line(image_frame, (960,0),(960,540), (0,0,255),10) #changed
        fr_height, fr_width = frame.shape[:2]
        if sum(gaze) != 0: cv2.circle(image_frame, (int(gx*fr_width), int(gy*fr_height)), 20, (0,0,255), 5)
        #result.write(image_frame)
        cv2.imshow('Result', image_frame)
        #wk = int(round((1/fps)*1000))
        if cv2.waitKey(1) & 0xFF == ord('q'): #cv2.waitKey(fps) == 27:
            break
    #cap.release()


if __name__ == '__main__':
    ipv4_address = "192.168.71.50"
    tobiiglasses = TobiiGlassesController(ipv4_address, video_scene=True)
    DEVICE = torch.device("cuda")
    ext_output = True
    m = Manager()

    frame_queue = m.Queue()
    darknet_image_queue = Queue(maxsize=1)
    depth_image_queue = Queue(maxsize=1)
    current_frame_queue = Queue(maxsize=1)
    gaze_queue= Queue(maxsize=1)

    print_detection_queue = m.Queue(maxsize=1)
    print_moving_object_queue = m.Queue(maxsize=1) #changed
    print_fps_queue = m.Queue(maxsize=1)
    print_gaze_queue = m.Queue(maxsize=1)
    print_vehicle_status_queue = m.Queue(maxsize=1)
    print_alert_status_queue = m.Queue(maxsize=1) 

    dark_network, class_names, class_color = init_darknet()
    darknet_width = darknet.network_width(dark_network)
    darknet_height = darknet.network_height(dark_network)
    feed_height, feed_width, depth_encoder, depth_decoder = init_depth()
    proj, participant = init_tobii()
    status = tobii_calib(proj, participant)
    if status == True: os.system('play -nq -t alsa synth {} sine {}'.format(1, 500))
    else:
        for i in range(3):
            recalib = str(input("Do you want to recalibrate 'yes/no?'"))
            if recalib == "yes":
                recalib_status = tobii_calib(proj, participant)
                if recalib_status == True:
                    os.system('play -nq -t alsa synth {} sine {}'.format(1, 500))
                    break
            else:
                exit()

    start = str(input("Do you want to start the streaming? (yes/no)"))
    if start == "yes":
        #Start live capture
        vid = cv2.VideoCapture("rtsp://%s:8554/live/scene" % ipv4_address)  #Create a object to capture the livestream video data.
    else:
        sleep(100)
        vid = cv2.VideoCapture("rtsp://%s:8554/live/scene" % ipv4_address)
    # output_file = "./videotrialaug21.mp4"
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # frame_width = int(vid.get(3))
    # frame_height = int(vid.get(4))
    # size = (frame_width, frame_height)
    # result = cv2.VideoWriter(output_file, fourcc, 50, size)

    #Start video stream
    if (vid.isOpened()== True): # Check if camera opened successfully
        tobiiglasses.start_streaming() # Read until video is completed
    else:
        print("Error opening video stream or file")

    vidcap = Thread(target=video_capture, args=(frame_queue, darknet_image_queue, depth_image_queue, current_frame_queue, gaze_queue))
    detects = Thread(target=inference, args=(darknet_image_queue,depth_image_queue, current_frame_queue, print_detection_queue,print_moving_object_queue, print_fps_queue, gaze_queue, print_gaze_queue, print_vehicle_status_queue, print_alert_status_queue))
    video = Process(target=show_video, args=(frame_queue, print_detection_queue, print_moving_object_queue, print_fps_queue, print_gaze_queue, print_vehicle_status_queue, print_alert_status_queue))
    #detects.setDaemon(True)
    #video.setDaemon(True)
    vidcap.start()
    detects.start()
    video.start()

    cv2.destroyAllWindows()

