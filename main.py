from common import *
from hj_common import *

def inspect1(root_path):
    sample_info = read_sample_info(os.path.join(root_path,'sample_info.json'))
    pixel_range = sample_info["device"]["pixel_range"]

    camera_images = read_image_frames(root_path)

    full_image = merge_images(camera_images, pixel_range)
    glass_image = extract_glass_area(full_image)
    inspect_result = inspect_glass_defect1(glass_image)

    return inspect_result

if __name__ == '__main__':
    root_path = "data/241016_110832_917_13"
    inspect1(root_path)
    print('finish')