import json
import os
import cv2

def read_sample_info(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data

def read_image_frames(root):
    '''
    :param root: 相机帧所在的目录
    :return: {cam_id: [image0, image1,...]}
    '''
    images = os.listdir(os.path.join(root, "images"))

    camera_images = {}
    cur_cam_idx = -1
    for image in images:
        # one_camera_images = []
        cam_idx, filename = image.split("_")

        if cam_idx not in camera_images:
            camera_images[cam_idx] = []

        img = cv2.imread(os.path.join(root, "images", image), cv2.IMREAD_UNCHANGED)
        camera_images[cam_idx].append(img)

    return camera_images

