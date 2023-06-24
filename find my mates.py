#!/usr/bin/env python3.8
import rospy
from sensor_msgs.msg import Image

from cv_bridge import CvBridge
import cv2
import numpy as np

from std_msgs.msg import String

from mr_voice.msg import Voice

from pcms.openvino_models import HumanPoseEstimation, FaceDetection

from RobotChassis import RobotChassis
import mediapipe as mp
from geometry_msgs.msg import Twist


# from tf.transformations import euler_from_quaternion

# from pcms.pytorch_models import *

def callback_character(msg):
    global character
    character = msg.data


def callBack_image(msg):
    global _image
    _image = CvBridge().imgmsg_to_cv2(msg, "bgr8")


def say(text):
    global _pub_speaker
    if text is None: return
    if len(text) == 0: return
    rospy.loginfo("ROBOT: %s" % text)
    _pub_speaker.publish(text)
    rospy.sleep(1)


'''
def imu_callback(msg):
    global imu
    imu = msg
'''


def callback_depth(msg):
    global depth
    depth = CvBridge().imgmsg_to_cv2(msg, "passthrough")


def callback_voice(msg):
    global _voice
    _voice = msg


def get_real_xyz(x, y):
    global depth
    if depth is None:
        return -1, -1, -1
    h, w = depth.shape[:2]
    d = depth[y][x]
    a = 49.5 * np.pi / 180
    b = 60.0 * np.pi / 180
    real_y = (h / 2 - y) * 2 * d * np.tan(a / 2) / h
    real_x = (w / 2 - x) * 2 * d * np.tan(b / 2) / w
    return real_x, real_y, d


def get_target_d(frame):
    poses = dnn_human_pose.forward(frame)
    frame = dnn_human_pose.draw_poses(frame, poses, 0.1)
    global _image
    image = _image.copy()
    x1, y1, x2, y2 = 1000, 1000, 0, 0
    nlist = []
    dlist = []
    targetd = 10000
    if len(poses) != 0:
        for i in range(len(poses)):
            x, y, c = map(int, poses[i][0])
            nlist.append([x, y, i])
        for i in range(len(nlist)):
            d = getDepth(nlist[i][0], nlist[i][1])
            if d != 0 and d < 2900:
                dlist.append([d, nlist[i][2]])
        if len(dlist) == 0: return None, -1, -1
        for i in range(len(dlist)):
            if dlist[i][0] < targetd:
                targetd = dlist[i][0]
                targeti = dlist[i][1]
        dlist = sorted(dlist)
        print(f"dlist:{dlist}")
        pose = poses[dlist[0][1]]
        for i, p in enumerate(pose):
            x, y, c = map(int, p)
            if x < x1 and x != 0: x1 = x
            if x > x2: x2 = x
            if y < y1 and y != 0: y1 = y + 5
            if y > y2: y2 = y

        # cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        dnn_human_pose.forward(frame)
        # rospy.loginfo(appearance)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        # rospy.loginfo(cx,cy)
        cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
        # return cx, image
        d = depth[cy][cx]
        a = 0
        if d != 0:
            a = max(int(50 - (abs(d - 1150) * 0.0065)), 20)
        rospy.loginfo(a)
        print("d : " + str(d))
        cv2.rectangle(image, (x1, y1 - a), (x2, y2), (255, 0, 0), 2)
        if d == -1:
            return -1, -1, -1
        return d, image, [x1, y1, x2, y2]
    return -1, -1, -1


def getDepth(cx, cy):
    d = depth[cy][cx]
    if d == 0:
        for i in range(1, 2, 1):
            for x in range(0 - i, 0 + i, i):
                for y in range(0 - i, 0 + i, i):
                    d = depth[y][x]
                    if d != 0:
                        return d
    return d


def angular_PID(cx, tx):
    e = tx - cx
    p = 0.0015
    z = p * e
    if z > 0:
        z = min(z, 0.3)
        z = max(z, 0.01)
    if z < 0:
        z = max(z, -0.3)
        z = min(z, -0.01)
    return z


def linear_PID(cd, td):
    e = cd - td
    p = 0.00025
    x = p * e
    if x > 0:
        x = min(x, 0.2)
        x = max(x, 0.1)
    if x < 0:
        x = max(x, -0.2)
        x = min(x, -0.1)
    return x

def move_status():
    while not rospy.is_shutdown():
        code = chassis.status_code
        text = chassis.status_text

        if code == 0:  # No plan.
            pass
        elif code == 1:  # Processing.
            pass
        elif code == 3:  # Reach point.
            say("I am arrived.")
            #_status = 2
            break
        elif code == 4:  # No solution.
            say("I am trying to move again.")
            break
        else:
            rospy.loginfo("%d, %s" % (code, text))
            break

def count_color(frame):
    h, w, c = frame.shape
    c = 0
    for x in range(w):
        for y in range(h):
            if frame[y, x, 0] != 0 and frame[y, x, 1] != 0 and frame[y, x, 1] != 0:
                c += 1
    return c

def detect_color(frame):
    _frame = cv2.resize(frame, (30, 40))
    hsv_frame = cv2.cvtColor(_frame, cv2.COLOR_BGR2HSV)
    clist = []

    low_red = np.array([156, 43, 46])
    high_red = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv_frame, low_red, high_red)
    red = cv2.bitwise_and(_frame, _frame, mask=red_mask)
    clist.append([count_color(red), "red"])

    low_orange = np.array([5, 75, 0])
    high_orange = np.array([21, 255, 255])
    orange_mask = cv2.inRange(hsv_frame, low_orange, high_orange)
    orange = cv2.bitwise_and(_frame, _frame, mask=orange_mask)
    clist.append([count_color(orange), "orange"])

    low_yellow = np.array([22, 93, 0])
    high_yellow = np.array([33, 255, 255])
    yellow_mask = cv2.inRange(hsv_frame, low_yellow, high_yellow)
    yellow = cv2.bitwise_and(_frame, _frame, mask=yellow_mask)
    clist.append([count_color(yellow), "yellow"])
    # Green color
    low_green = np.array([34, 0, 0])
    high_green = np.array([94, 255, 255])
    green_mask = cv2.inRange(hsv_frame, low_green, high_green)
    green = cv2.bitwise_and(_frame, _frame, mask=green_mask)
    clist.append([count_color(green), "green"])

    low_blue = np.array([94, 10, 2])
    high_blue = np.array([126, 255, 255])
    blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
    blue = cv2.bitwise_and(_frame, _frame, mask=blue_mask)
    clist.append([count_color(blue), "blue"])

    low_purple = np.array([130, 43, 46])
    high_purple = np.array([145, 255, 255])
    purple_mask = cv2.inRange(hsv_frame, low_purple, high_purple)
    purple = cv2.bitwise_and(_frame, _frame, mask=purple_mask)
    clist.append([count_color(purple), "purple"])

    low_pink = np.array([143, 43, 46])
    high_pink = np.array([175, 255, 255])
    pink_mask = cv2.inRange(hsv_frame, low_pink, high_pink)
    pink = cv2.bitwise_and(_frame, _frame, mask=pink_mask)
    clist.append([count_color(pink), "pink"])

    low = np.array([0, 0, 0])
    high = np.array([255, 255, 130])
    black_mask = cv2.inRange(hsv_frame, low, high)
    black = cv2.bitwise_and(_frame, _frame, mask=black_mask)
    clist.append([count_color(black), "black"])

    lower_white = np.array([0, 0, 168])
    upper_white = np.array([172, 111, 255])
    mask = cv2.inRange(hsv_frame, lower_white, upper_white)
    white = cv2.bitwise_and(_frame, _frame, mask=mask)
    clist.append([count_color(white), "white"])

    lower_gray = np.array([0, 0, 168])
    upper_gray = np.array([172, 111, 255])
    mask = cv2.inRange(hsv_frame, lower_gray, upper_gray)
    gray = cv2.bitwise_and(_frame, _frame, mask=mask)
    clist.append([count_color(gray), "gray"])

    print(sorted(clist, reverse=True))
    return sorted(clist, reverse=True)[0][1]


def detect_cloth_color(image):
    poses = dnn_human_pose.forward(image)
    if len(poses) > 0:
        x1 = int(poses[0][6][0])
        y1 = int(poses[0][6][1])
        x2 = int(poses[0][11][0])
        y2 = int(poses[0][11][1])
        cv2.circle(image, (x1, y1), 5, (255, 0, 0), -1)
        cv2.circle(image, (x2, y2), 5, (255, 0, 0), -1)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        # print(cx,cy)
        d = getDepth(cx, cy)
        # print("d : ",str(d))
        # print(f"x1:{x1},y1:{y1},x2:{x2},y2:{y2}")
        if (x1 != 0 and x2 != 0 and y1 != 0 and y2 != 0) and (x1 < x2 and y1 < y2) and (
                x1 > 0 and x2 > 0 and y1 > 0 and y2 > 0):
            # print(x1,x2,y1,y2)
            x1 -= int(d * 0.01)
            x2 += int(d * 0.01)
            cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # print(f"x1:{x1},y1:{y1},x2:{x2},y2:{y2}")
            frame = image[y1:y2, x1:x2, :]
            # cv2.imshow("up",frame)
            Upcolor = detect_color(frame)
            print("upC :", str(Upcolor))
            x1 = int(poses[0][12][0])
            y1 = int(poses[0][12][1])
            x2 = int(poses[0][13][0])
            y2 = int(poses[0][13][1])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            d = getDepth(cx, cy)
            if (x1 != 0 and x2 != 0 and y1 != 0 and y2 != 0) and (x1 < x2 and y1 < y2) and (
                    x1 > 0 and x2 > 0 and y1 > 0 and y2 > 0):
                # print(x1,x2,y1,y2)
                cv2.rectangle(image, (x1 - int(d * 0.015), y1), (x2 + int(d * 0.015), y2), (0, 255, 0), 2)
                # print(f"x1:{x1},y1:{y1},x2:{x2},y2:{y2}")
                frame = image[y1:y2, x1:x2, :]
                cv2.imshow("down", frame)
                dncolor = detect_color(frame)
                print("dpwnC :", str(dncolor))
                # cv2.imwrite(image,"/home/pcms/Desktop/detectColor.png")
                return Upcolor, dncolor
    return -1, -1


def getDepth(cx, cy):
    d = depth[cy][cx]
    if d == 0:
        for i in range(1, 2, 1):
            for x in range(0 - i, 0 + i, i):
                for y in range(0 - i, 0 + i, i):
                    d = depth[y][x]
                    if d != 0:
                        return d
    return d


def getMask(img):
    if image is None: return -1
    h, w, c = image.shape
    if h == 0 or w == 0 or c == 0:
        return None, None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    upperb = np.array([54, 255, 255])
    lowerb = np.array([0, 20, 0])
    mask = cv2.inRange(img, lowerb=lowerb, upperb=upperb)
    masked = cv2.bitwise_and(img, img, mask=mask)
    return (mask, masked)


def check_mask(face_img) -> bool:
    face_img = face_img[int(face_img.shape[0] / 2):].copy()
    # check if it wearing mask
    face_img = cv2.resize(face_img, (224, 112))
    mask, masked = getMask(face_img)
    # get total mask pixel
    print(f"mask:{mask}")
    if mask is None:
        return None
    tot_pixel = np.sum(mask == 255)
    return tot_pixel < 9500

def get_face_img(frame):
    faces = dnn_face.forward(frame)
    if len(faces) > 0:
        x1, y1, x2, y2 = faces[0]
        if x1 > x2: x1,x2 = x2,x1
        if y1 > y2: y1,y2 = y2,y1
        if (x1 != 0 and x2 != 0 and y1 != 0 and y2 != 0) and (x1 < x2 and y1 < y2) and (
                x1 > 0 and x2 > 0 and y1 > 0 and y2 > 0):
            h, w, c = frame.shape
            if h != 0 and w != 0 and c != 0:
                face_img = frame[y1:y2, x1:x2, :]
                return face_img
    return None


def detectGlasses(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w, c = hsv_frame.shape
    cnt = 0
    if h != 0 and w != 0 and c != 0:
        for x in range(w):
            for y in range(h):
                if hsv_frame[y, x, 2] <= 100:
                    cnt += 1
    if cnt >= 5:
        return True
    return False


def getGlasses(frame):
    if frame is None: return False
    img = frame.copy()
    
    # imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(img)
    x1, y1, x2, y2 = 0, 0, 0, 0
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(image=img, landmark_list=faceLms, connections=mpFaceMesh.FACEMESH_TESSELATION,
                                  connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
        for id, lm in enumerate(faceLms.landmark):
            # print(lm)
            ih, iw, ic = img.shape
            if id == 55:
                x1, y1 = int(lm.x * iw), int(lm.y * ih)
            elif id == 412:
                x2, y2 = int(lm.x * iw), int(lm.y * ih)
    if x1 != 0 and y1 != 0 and x2 != 0 and y2 != 0:
        glasses_img = frame[y1:y2, x1:x2, :]
        # glasses_img = cv2.resize(glasses_img, (320, 240))
        if detectGlasses(glasses_img):
            # cv2.imshow("glasses_img", glasses_img)
            print("yessssssssssssssss")
            return True
        return False

def checkFaces(frame,cnt):
    faces = dnn_face.forward(frame)
    if cnt == 0:
        x1 = 1000
    elif cnt == 1:
        x1 =0
    elif cnt == 2:
        x1=1000
    if len(faces) == 0:
        return -1,-1,-1,-1
    for xi1, yi1, xi2, yi2 in faces:
        cx,cy = int((xi1+xi2)//2),int((yi1+yi2)//2)
        d = depth[cy][cx]
        if cnt == 0 and d < 2700:
            if xi1 < x1:
                x1,y1,x2,y2 = xi1, yi1, xi2, yi2
        elif cnt == 1 and d < 2700:
            if xi1> x1:
                x1,y1,x2,y2 = xi1, yi1, xi2, yi2
        elif cnt ==2 and d < 2700:
            if abs(xi1-320) < abs(x1-320):
                x1,y1,x2,y2 = xi1, yi1, xi2, yi2
                
    if x1 == 1000:
        return -1,-1,-1,-1
    else:
        return x1,y1,x2,y2
        



if __name__ == "__main__":
    rospy.init_node("task")
    rospy.loginfo("started task")
    _voice = None
    rospy.Subscriber("/voice/text", Voice, callback_voice)
    dnn_human_pose = HumanPoseEstimation()
    print("finish dnn")
    _image = None
    rospy.Subscriber("/camera/rgb/image_raw", Image, callBack_image)
    rospy.wait_for_message("/camera/rgb/image_raw", Image)
    print("finish rgb")
    imagePath = rospy.Publisher("/pcms/imageP", String, queue_size=10)

    character = None
    rospy.Subscriber("/pcms/appearance", String, callback_character)

    depth = None
    rospy.Subscriber("/camera/depth/image_raw", Image, callback_depth)
    print("finish camera")
    path_openvino = "/home/pcms/models/openvino/"
    dnn_face = FaceDetection(path_openvino)
    #dnn_appearance = PersonAttributesRecognition(path_openvino)
    print("wait for opening yolo")
    _pub_speaker = rospy.Publisher("/speaker/say", String, queue_size=10)
    msg_cmd = Twist()
    pub_cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    print("readys")
    vec_names = []
    vec_dicts = {}
    tlist = []  # appearance
    alist = []  # again appearance
    Llist = []  # location
    status = -1
    now_person = ""
    have_glasses = False
    t0 = None
    chassis = RobotChassis()
    cnt = 0
    guest_color = None
    pos = {"roomL": (3.95, -0.934, -1.7), "roomR": (4.64, -0.623, -3.5),"roomM": (4.64, -0.623, -1.95), "master": (5.05, 0.937, 0.85)}
    pos_Fablab = {"roomL" : (-6.67,-6.2,-3.2),"roomR" : (-6.67,-6.2,0.2),"roomM": (-6.67,-6.2,-1.75),"master" : (-6.67,-3.76,1.7)}
    save = False  # status =1
    c = 0
    publish = False
    angular = 0
    i=0
    mpDraw = mp.solutions.drawing_utils
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
    mp_drawing_styles = mp.solutions.drawing_styles

    while not rospy.is_shutdown():
        rospy.Rate(30).sleep()
        msg_cmd.linear.x = 0.0
        msg_cmd.angular.z = 0.0
        tlist = []
        # cv2.imshow("_image",_image)
        frame = _image.copy()
        print(status)
        if status == -1:  # go to the room with guests in it
            print(cnt)
            if _voice is None: continue
            if "dining" in _voice.text and "room" in _voice.text:
                rospy.loginfo(_voice.text)
                #chassis.move_to(*pos["door"])
                #chassis.move_to(*pos["waitpoint1_come"])
                if cnt == 1:
                    chassis.move_to(*pos_Fablab["roomL"])
                elif cnt == 0:
                    chassis.move_to(*pos_Fablab["roomR"])
                else:
                    chassis.move_to(*pos_Fablab["roomM"])
                #move_status()
                say('I arrived dining room')
                _voice = None
                status +=1
                print("I arrived outside the room")
                angular = 0
        elif status == 0:
            x1,y1,x2,y2 = checkFaces(frame,cnt)
            if x1 != -1:
                save_frame = frame.copy()
                cv2.rectangle(save_frame,(x1,y1),(x2,y2),(0, 255, 0), 2)
                cv2.imwrite(f"/home/pcms/Desktop/detect_guest_{str(cnt)}.png",save_frame)
                cx = ((x1 + x2) // 2)
                if max(cx, 315) == cx and min(cx, 325) == cx:
                    status += 1
                    print("finish")
                    msg_cmd.angular.z = 0
                    save = False
                    yolo = False
                    publish = False
                    rospy.sleep(0.05)
                    hasMask = -1
                    upcolor = -1
                    dncolor = -1
                    i = 0
                    have_glasses = False
                    say("please wait")
                else:
                    v = angular_PID(cx, 320)
                    msg_cmd.angular.z = v
                    angular += v *15*-1
            else:
                if cnt == 0:
                    msg_cmd.angular.z = -0.2
                    angular += 1.5
                elif cnt == 1:
                    msg_cmd.angular.z = 0.2
                    angular -= 1.5
            print(f"angular:{angular}")
            print(msg_cmd.angular.z)
            pub_cmd.publish(msg_cmd)

        elif status == 1:
            if frame is None: continue
            d, image, clist = get_target_d(frame)
            face_img = get_face_img(frame)
            if clist == -1: continue
            _frame = _image.copy()
            # print(appearance)
            cv2.imshow("test", image)
            bodyPart = frame[clist[1] - 15:clist[3] + 15, clist[0]:clist[2], :]  # y1:y2,x1:x2
            # cv2.imshow("body",bodyPart)
            upcolor, dncolor = detect_cloth_color(frame)
            if face_img is not None:
                hasMask = check_mask(face_img)
            print(f"hasMask:{hasMask}")
            if publish == False:
                # guest = appearance
                p = "/home/pcms/Desktop/test2.png"
                cv2.imwrite(p, _image)
                imagePath.publish("/home/pcms/Desktop/test2.png")
                print("publish")
                publish = True

            if i < 10:
                have_glasses = have_glasses or getGlasses(face_img)
                print(f"have_glass:{have_glasses}")
                i += 1
            if publish == True and upcolor != -1 and dncolor != -1 and hasMask is not None and i == 10:
                status += 1
                i = 0
    
        elif status == 2:
            if frame is None: continue
            print("frame is exist")
            d, image, clist = get_target_d(frame)
            # bodyPart = frame[clist[1]:clist[3], clist[0]:clist[2], :]  # y1:y2,x1:x2 
            if d != -1 and d != None:
                print("d is not None")
                if d == 0:
                    print("d == 0")
                    if clist != -1:
                        cx = int((clist[0] + clist[2])) // 2
                        cy = int((clist[1] + clist[3])) // 2
                        d = getDepth(cx, cy)
                rospy.loginfo(d)
                if d < 900 or d > 1000:
                    print("executing PID Function!")
                    v = linear_PID(d, 950)
                    
                    msg_cmd.linear.x = v
                    print(f"speed : {v}")
                else:
                    #rospy.sleep(0.05)
                    msg_cmd.linear.x = 0.0
                    say('I found a guest')
                    status += 1
                    print("done")
                    #rospy.sleep(0.05)
                    say("What is your name")
                    print("say your name")
                frame = image
            pub_cmd.publish(msg_cmd)

        elif status == 3:
            if _voice is None: continue
            Amelialist = ["Minion", "Emilla","Amelia","million","Maria"]
            Angellist = ["angel","Angel"]
            Avalist = ["ever","Ava","AVA"]
            Charlielist = ["Charlie","Cherry","Poly"]
            Charlottelist = ["Charlotte","Eric","solid"]
            Hunterlist = ["Sonta","Santa","Hunter"]
            Maxlist = ["Max"]
            Mialist = ["smile","Mia"]
            Olivialist = ["Olivia"]
            Parkerlist = ["Parker"]
            Samlist = ["Sam"]
            Jacklist = ["Jeff","Jack"]
            Noahlist = ["Wrong","Noah","lower","lord"]
            Oliverlist = ["Oliver"]
            Thomaslist = ["Thomas"]
            Williamlist = ["William", "Lily"]
            rospy.loginfo(_voice.text)
            v = _voice.text.split(" ")[-1]
            if v == "is" or v == "name" or (v not in Amelialist and v not in Angellist and v not in Avalist and v not in Charlielist and v not in Charlottelist and v not in Hunterlist and v not in Maxlist and v not in Mialist and v not in Olivialist and v not in Parkerlist and v not in Samlist and v not in Jacklist and v not in Noahlist and v not in Oliverlist and v not in Thomaslist and v not in Williamlist):
                say("Could you repeat your name")
                _voice = None
                continue
            elif v in Amelialist:
                now_person = "Amelia"
            elif v in Angellist:
                now_person = "Angel"
            elif v in Avalist:
                now_person = "William"
            elif v in Charlielist:
                now_person = "Charlie"
            elif v in Charlottelist:
                now_person = "Charlotte"
            elif v in Hunterlist:
                now_person = "Hunter"
            elif v in Maxlist:
                now_person = "Max"
            elif v in Mialist:
                now_person = "Mia"
            elif v in Olivialist:
                now_person = "Olivia"
            elif v in Parkerlist:
                now_person = "Parker"
            elif v in Samlist:
                now_person = "Sam"
            elif v in Jacklist:
                now_person = "Jack"
            elif v in Noahlist:
                now_person = "Noah"
            elif v in Oliverlist:
                now_person = "Oliver"
            elif v in Thomaslist:
                now_person = "Thomas"
            elif v in Williamlist:
                now_person = "William"
            rospy.loginfo(now_person)
            # np.savetxt("/home/pcms/catkin_ws/src/beginner_tutorials/src/" + v, vec)
            # vec_names.append(v)
            # rospy.loginfo("saved")
            _voice = None
            status += 1
            
        elif status == 4:
            #chassis.move_to(*pos["door"])
            #chassis.move_to(*pos["waitpoint1_back"])
            chassis.move_to(*pos_Fablab["roomM"])
            chassis.move_to(*pos_Fablab["master"])
            #move_status()
            print(character)
            while character is None:
                rospy.Rate(20).sleep()
                continue
            tem = []
            print("character : " + character)
            # imagePath.publish(None)
            # tlist.append(character)
            gender = character.split()[1]
            tlist = []
            say(f"I found {now_person} in the room")
            if hasMask and "mask" not in alist and len(tlist) < 2:
                tlist.append(f"{now_person} is wearing a mask")
                alist.append("mask")
            if have_glasses and "glass" not in alist and len(tlist) < 2:
                tlist.append(f"{now_person} is wearing a glasses")
                alist.append("glass")
            if "upcolor" not in alist and len(tlist) < 2:
                tlist.append(f"{now_person} is wearing a {upcolor} cloth")
                alist.append("upcolor")
            if "dncolor" not in alist and len(tlist) < 2:
                tlist.append(f"{now_person} is wearing a lower cloth in {dncolor}")
                alist.append("dncolor")
            if "gender" not in alist and len(tlist) < 2 and cnt == 2:
                tlist.append(f"{now_person} is a {gender}")
                alist.append("gender")
            if "hair" not in alist and len(tlist) < 2 and cnt == 2:
                if gender == "man" or gender == "boy":
                    tlist.append(f"{now_person} has short hair")
                else:
                    tlist.append(f"{now_person} has long hair")
                alist.append("hair")
            if cnt == 0:
                '''
                if angular <= 45:
                    say(f"{now_person} is next to a book")
                elif angular > 45 and angular <= 90:
                    say(f"{now_person} is next to a box")
                elif angular > 90 and angular <= 135:
                    say(f"{now_person} is next to a box")
                else:'''  
                if angular < 42:
                    say(f"{now_person} is next to a book")
                elif angular > 42 and angular < 47:
                    say(f"{now_person} is next to a box")
                elif angular >47:
                    say(f"{now_person} is next to a bottle")  
            elif cnt == 1:
                if angular >= -45:
                    say(f"{now_person} is next to a bottle")
                elif angular < -45 and angular >= -90:
                    say(f"{now_person} is next to a box")
                elif angular < -90 and angular >= -135:
                    say(f"{now_person} is next to a box")
                else:
                    say(f"{now_person} is next to book")
            elif cnt ==2:
                if angular <= 30 or angular >= -30:
                    say(f"{now_person} is next to a box")
                elif angular < -30:
                    say(f"{now_person} is next to a book")
                elif angular > 30:
                    say(f"{now_person} is next to a bottle")


            for t in tlist:
                say(t)
                #rint(f"t:{t}")
                rospy.sleep(0.05)
            cnt += 1
            print(f"angular: {angular}")
            status = -1
            angular = 0
            print("status :", status)
            if cnt == 3:
                break
            break
        cv2.imshow("image", frame)
        cv2.waitKey(1)
