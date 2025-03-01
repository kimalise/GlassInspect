import os
from hj_common import *
from common import *
import shutil

root = "E:/shijue"
target_dir = "data/for_labelme"
ref_dir = "data/for_ref"

def prepare_for_labelme():
    date_dirs = os.listdir(root)
    for date in date_dirs:
        if not os.path.isdir(os.path.join(root, date)):
            continue
        product_dirs = os.listdir(os.path.join(root, date))
        for product in product_dirs:
            print("Processing ", os.path.join(root, date, product))
            ref_product = os.path.join(ref_dir, product)
            target_product = os.path.join(target_dir, product)

            if not os.path.exists(ref_product):
                os.makedirs(ref_product)

            if not os.path.exists(target_product):
                os.makedirs(target_product)

            ref_image1 = os.path.join(root, date, product, "defects_on_original.jpg")
            if not os.path.exists(ref_image1):
                # print(ref_image1)
                ref_image2 = os.path.join(root, date, product, "image_with_defects.jpg")
                if not os.path.exists(ref_image2):
                    print("[no defect image] ", ref_image1, ref_image2)
                else:
                    shutil.copy(ref_image2, os.path.join(ref_product, "ref_image.jpg"))
            else:
                shutil.copy(ref_image1, os.path.join(ref_product, "ref_image.jpg"))

            product_root = os.path.join(root, date, product)
            sample_info = read_sample_info(os.path.join(product_root, 'sample_info.json'))
            if 'device' in sample_info:
                pixel_range = sample_info["device"]["pixel_range"]
            else:
                with open(os.path.join(product_root, "config", "device.json"), "r", encoding="utf-8") as f:
                    device = json.load(f)
                    pixel_range = device["pixel_range"]

            camera_images = read_image_frames(product_root)

            full_image = merge_images(camera_images, pixel_range)
            glass_image = extract_glass_area(full_image)

            cv2.imwrite(os.path.join(target_product, "image.jpg"), glass_image)

if __name__ == '__main__':
    prepare_for_labelme()