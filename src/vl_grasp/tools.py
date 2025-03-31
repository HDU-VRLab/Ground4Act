import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import threading
import rospy
from gjt_ur_moveit_gazebo.srv import grasp_pose, grasp_poseRequest
from saveimg import ImageSaver
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

def get_instance_angle(mask):
    binary_mask = np.uint8(mask > 0)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 1:
        return None
    points = contours[0]
    [vx, vy, x, y] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    angle = np.arctan2(vy, vx) * 180 / np.pi
    start_point = (int(x - vx * 100), int(y - vy * 100))
    end_point = (int(x + vx * 100), int(y + vy * 100))
    return angle, start_point,end_point

def show_angle(suction_pts_image,start_points, end_points, contour_angles):
    canvas = np.zeros_like(suction_pts_image)
    for start_point, end_point, angle in zip(start_points, end_points, contour_angles):
        line_length = math.sqrt((end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2)
        shortened_length = line_length / 1.5
        angle_rad = math.atan2(end_point[1] - start_point[1], end_point[0] - start_point[0])
        new_start_x = int(start_point[0] + shortened_length * math.cos(angle_rad))
        new_start_y = int(start_point[1] + shortened_length * math.sin(angle_rad))
        new_start_point = (new_start_x, new_start_y)
        new_end_x = int(end_point[0] - shortened_length * math.cos(angle_rad))
        new_end_y = int(end_point[1] - shortened_length * math.sin(angle_rad))
        new_end_point = (new_end_x, new_end_y)
        cv2.arrowedLine(suction_pts_image, new_end_point,new_start_point,  (255,0,0), 3)
        line_center = ((new_start_point[0] + new_end_point[0]) // 2, (new_start_point[1] + new_end_point[1]) // 2)
        line_length = 30 
        line_end = (line_center[0] + line_length, line_center[1])
        cv2.line(suction_pts_image, line_center, line_end, (0, 255, 0), 2)
        text_position = (int((start_point[0] + end_point[0]) / 2), int((start_point[1] + end_point[1]) / 2))
        angle=np.round(angle, 1)
        cv2.putText(suction_pts_image, "{}".format(angle), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return suction_pts_image

def show_img(name,image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

class MyThread(threading.Thread):
    def __init__(self, prompt):
        threading.Thread.__init__(self)
        self.prompt = prompt
        self.text = None
    
    def run(self):
        self.text = input(self.prompt)

def get_camera_points(pix_x,pix_y,cam_intrinsics):
    u = int(float(pix_x)/224*640)
    v = int(float(pix_y)/224*480)
    camera_points=[]
    ppx_up =  cam_intrinsics[0][2]
    ppy_up =  cam_intrinsics[1][2]
    pos_z = 0.769
    pos_x = np.multiply(u  - ppx_up,pos_z / cam_intrinsics[0][0])
    pos_y = np.multiply(v  - ppy_up,pos_z / cam_intrinsics[1][1])
    camera_point = np.asarray([pos_x,pos_y,pos_z])
    camera_points.append(camera_point)
    return camera_points

def get_grasp_angle(angle):
    if (abs(angle)>90):
        angle = -(180-abs(angle))
    angle =abs(90-abs(angle))
    return angle

def make_object_info_dict_oneimg(camera_points, contour_angles):
    camera_points_nums = [float(num) for num in camera_points[0]]
    contour_angles_nums = [float(num) for num in contour_angles[0]]
    object_info_dict = {
        'camera_point': camera_points_nums,
        'contour_angle': contour_angles_nums,
    }

    return object_info_dict

def make_object_info_dict_one_angle(camera_points, angle):
    camera_points_nums = [float(num) for num in camera_points[0]]
    object_info_dict = {
        'camera_point': camera_points_nums,
        'contour_angle': angle
    }

    return object_info_dict


def send_pose_to_robot(object_info_dict,action_id):
    move_client = rospy.ServiceProxy("moveit_grasp", grasp_pose)
    rospy.wait_for_service("moveit_grasp")
    camera_point = object_info_dict['camera_point']
    angles =  object_info_dict['contour_angle']
    start_x, start_y, start_z = camera_point

    if action_id == 0:
        print("--------------Pushing--------------")
        angle = angles[0]
    elif action_id == 1:
        print("--------------Grasping--------------")
        angle = angles[0]
    elif action_id == 2:
        print("--------------Releaseing_gripper-------------")
        angle = 0
    move_req = grasp_poseRequest()
    pos_x,pos_y,pos_z=start_x,start_y,start_z
    rpy = [angle, action_id, 0.08230054773210334]
    move_req.grasppose_x, move_req.grasppose_y, move_req.grasppose_z = pos_x,pos_y,pos_z
    move_req.grasppose_R, move_req.grasppose_P, move_req.grasppose_Y =rpy[0], rpy[1], rpy[2]
    result = move_client.call(move_req)
    return  result

def save_image():
    image_saver = ImageSaver()
    image_saver.save_images(color_filename="saved_picture/color{}.png",
                            depth_filename="saved_picture/depth{}.png".format(image_saver.counter))
    print("{} images have been saved".format(image_saver.counter))

def get_camera_data():
    rospy.sleep(1)
    raw_img = rospy.wait_for_message('/camera/rgb/image_raw', Image)
    color_img = CvBridge().imgmsg_to_cv2(raw_img, "bgr8")
    depth_img = rospy.wait_for_message('/camera/depth/image_raw', Image)
    rospy.sleep(0.01)
    depth_image = CvBridge().imgmsg_to_cv2(depth_img,desired_encoding='passthrough') 
    # print(depth_image)
    depth_array = np.array(depth_image, dtype=np.float32)
    depth_img= cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return color_img, depth_img