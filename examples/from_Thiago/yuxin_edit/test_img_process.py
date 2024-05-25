from fx19.dr_probe import DrProbe
import os
import cv2
#initiate
dr_probe = DrProbe()
parent_folder = os.getcwd()
parent_folder =  '/home/share/g-chan/yuxin_fantastx'
#the input folder and .cel file should be prepared in advance
input_dir_path = f'{parent_folder}/input/'
output_dir_path = f'{parent_folder}/output/'

dr_probe.inputs(input_dir_path= input_dir_path, 
                output_dir_path=output_dir_path, 
                defoci_vals=[5, 8, 10, 12, 14, 16, 20, 22, 24])

# dr_probe.run_dr_probe(input_dir_path=input_dir_path, 
#                 output_dir_path=output_dir_path)

# dr_probe.figure_generation()

# Assuming you have already defined dr_probe instance and required functions


# Load the images
crp_exp_img = cv2.imread(f'{parent_folder}/denoised_13_rotated_cropped.jpg', cv2.IMREAD_GRAYSCALE)
dr_probe_img = cv2.imread(f'{parent_folder}/POSCAR_Pt_6layer_yuxin_edit_mirrrored_100slc_850x850_12nmDefocus.png', cv2.IMREAD_GRAYSCALE)

# Call the evaluate_model function
dr_probe.evaluate_model(crp_exp_img, dr_probe_img)