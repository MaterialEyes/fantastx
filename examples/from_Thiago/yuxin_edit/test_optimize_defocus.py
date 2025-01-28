from fx19.dr_probe import DrProbe
import os
import cv2
#initiate
dr_probe = DrProbe()
parent_folder = os.getcwd()
parent_folder =  '/home/share/g-chan/yuxin_all/fantastx_examples/test_buildcell'
#the input folder and .cel file should be prepared in advance
input_dir_path = f'{parent_folder}/input/'
output_dir_path = f'{parent_folder}/output/'

dr_probe.inputs(input_dir_path= input_dir_path, 
                output_dir_path=output_dir_path, 
                defoci_vals=[5, 8, 10, 12, 14, 16, 20, 22, 24])

dr_probe.run_dr_probe(input_dir_path=input_dir_path, 
                output_dir_path=output_dir_path)

dr_probe.figure_generation()

exp_img = cv2.imread(f'{parent_folder}/denoised_13_rotated_cropped.jpg', cv2.IMREAD_GRAYSCALE)
dr_probe.optimize_all(exp_img)
dr_probe.save_optimized_img()
dr_probe.find_best_paras()