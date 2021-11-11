import cv2 as cv2
import os, shutil
import numpy as np
import argparse
from skimage import measure

class ShadowRemoverMain:
    @staticmethod
    def remove_shadows(orignal_img,lab_adjustment=False,region_adjustment_kernel_size=10,shadow_dilation_iteration=5,shadow_dilation_kernel_size=3,verbose=False):
        if orignal_img.shape[2] == 4:
            orignal_img = cv2.cvtColor(orignal_img, cv2.COLOR_BGRA2BGR)
        converted_img = cv2.cvtColor(orignal_img, cv2.COLOR_BGR2LAB)
        ouput_image = np.copy(orignal_img) 

        means = [np.mean(converted_img[:, :, i]) for i in range(3)]
        thresholds = [means[i] - (np.std(converted_img[:, :, i]) / 3) for i in range(3)]

        if sum(means[1:]) <= 256:
            mask = cv2.inRange(converted_img, (0, 0, 0), (thresholds[0], 256, 256))
        else: 
            mask = cv2.inRange(converted_img, (0, 0, 0), (thresholds[0], 256, thresholds[2]))

        kernel_size = (region_adjustment_kernel_size, region_adjustment_kernel_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, mask)
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, mask)
        
        #further implemenetation need to be done
        return ouput_image

    @staticmethod
    def process_image_file(img_name,save,verbose,region_adjustment_kernel_size,shadow_dilation_kernel_size,shadow_dilation_iteration,lab_adjustment):
        orignal_img = cv2.imread(img_name)
        print("Image Path:= {}".format(img_name))
        folder = 'images/Contours/'     #delete all the old temp files earlier
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

        shadow_clear_image = ShadowRemoverMain.remove_shadows(orignal_img,lab_adjustment=lab_adjustment,region_adjustment_kernel_size=region_adjustment_kernel_size,shadow_dilation_iteration=shadow_dilation_iteration,shadow_dilation_kernel_size=shadow_dilation_kernel_size,verbose=verbose)
        if save:
            f_name = img_name[:img_name.index(".")] + "_result" + img_name[img_name.index("."):]
            cv2.imwrite(f_name, shadow_clear_image)
            print("Final Image Path:=" + f_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove shadows from given image",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--image', required=True, help="Image of interest")
    parser.add_argument('-s', '--save', help= "Save the result",default=True)
    parser.add_argument('-v', '--verbose', help="Verbose",default=False)
    parser.add_argument('--rk', help="Region Adjustment Kernel Size", default=10)
    parser.add_argument('--sdk', help="Shadow Dilation Kernel Size", default=3)
    parser.add_argument('--sdi', help="Shadow Dilation Iteration", default=5)
    parser.add_argument('--lab', help="Adjust the pixel values according to LAB",default=False)
    args = parser.parse_args()
    ShadowRemoverMain.process_image_file(*vars(args).values())
