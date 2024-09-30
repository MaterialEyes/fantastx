#Import important modules
import drprobe as drp
import numpy as np
import os
import shutil
import sys 
#sys.path.append(r'')
from random import randint
import time
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import cv2
from skimage.feature import blob_dog, blob_log
from skimage.transform import rotate
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution


class DrProbe():

    def __init__(self):


        # Initialize constant parameters
        self.ht = 300;                   # High tension is 300 kV
        self.nx = 180; self.ny = 135;         #yuxin-edit:same pixel as exp img # All images, wavefunctions, etc. will be 512 x 512 pixels
        self.nz = 100;                 # The structures will be cut into 100 slices
        self.dwf = True; self.buni = 0.005;   # Debye-Wallar factor on and set B = 0.5 Ang^2
        #dwf = True; buni = 0.000;   # Debye-Wallar factor on and set B = 0.5 Ang^2
        self.absorb = True;              # Apply built-in absorptive form factors
        self.output = True;              # Give chatty output during simulations
        ##Cs = -13000
        #Cs = -13000                 # Cs = -9 um, enter in nm
        #C5 = 5000000                # C5 = 5 mm, enter in nm
        self.Cs = -9000                 # Cs = -9 um, enter in nm
        self.C5 = 5000000                # C5 = 5 mm, enter in nm
        self.msa_prm_gen = drp.msaprm.MsaPrm()
        self.wav_prm_gen = drp.wavimgprm.WavimgPrm()


    def inputs(self,input_dir_path,output_dir_path, defoci_vals):

        #%% 1.2) Specify input and output directories

        # Input directory containing .cel files to be simulated
        #input_dir = r'/home/thiago/Desktop/Argonne/ingrained_test/Cif_model_examples/Cif_model_examples/test/test_new_script/input'
        self.input_dir = r'{}'.format(input_dir_path)

        # Output directory where simulations should be saved
        #output_dir = r'/home/thiago/Desktop/Argonne/ingrained_test/Cif_model_examples/Cif_model_examples/test/test_new_script/output'
        self.output_dir = r'{}'.format(output_dir_path)
        # initialize output dir, incase run in one dir repeatly
        shutil.rmtree(self.output_dir,ignore_errors=True)
        os.makedirs(self.output_dir, exist_ok=True)
        # Create the full path to the SimulationCheck directory
        output_check_dir = os.path.join(self.output_dir, 'SimulationCheck')

        # Create the directory and any intermediate directories if they don't exist
        os.makedirs(output_check_dir, exist_ok=True)


        # output_check_dir = os.path.join(output_dir, 'SimulationCheck')
        # if os.path.isdir(output_check_dir):
        #     pass # Do nothing if exists
        # else:
        #     os.mkdir(output_check_dir) # Make directory otherwise

        # Range of defocus in nm. Positive values indicate overfocus
        #defoci = [-5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15]
        #defoci = [-12, -10, -8, -6, -4, 0, 4, 6, 8, 10, 12]
        #defoci = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1, 0]
        #defoci = [5]
        #defoci = [600]

        self.defoci = defoci_vals
        self.dict_paras = {val: {} for val in defoci_vals}
        # ex: defoci = [5, 8, 10, 12, 14, 16, 20, 22, 24],
        # where defoci_vals = [5, 8, 10, 12, 14, 16, 20, 22, 24]

        #defoci = [6]
        #defoci = [1,2,3,4,0]

        #input_file = str(input(r'/Users/ramon/Dropbox/Users/Ramon_Manzorro/Big_Data_HDR/Data/Image_simulations/input_test/pt-ceo2-thickness.cif'))

        #drp.commands.cellmuncher(cif_file=input, output=file='pt-ceo2-thickness.cel')

        # Read lattice parameters from cel filed

        self.cel_files = [f for f in os.listdir(input_dir_path) if f.endswith(".cel")]

        #if len(cel_files) != 1:
        #    raise ValueError("Expected one .cel file in input_dir_path, found {}".format(len(cel_files)))

        cel_file_path = os.path.join(input_dir_path, self.cel_files[0])

        self.a, self.b, self.c = np.genfromtxt(cel_file_path, skip_header=1, skip_footer=1, usecols=(1, 2, 3))[0]
        #self.a, self.b, self.c= np.genfromtxt("{}".format(self.input_dir), skip_header=1, skip_footer=1, usecols=(1, 2, 3))[0]
        #nz = int(round(c*6)) # Determine number of slices given that we would like 40 slices every nm


        #%% 1.3) Initialize Parameter Files

        # Initialize general MSA Parameter File
        #self.msa_prm_gen = drp.msaprm.MsaPrm()
        self.msa_prm_gen.wavelength = 0.0019687482               # Electron wavelength in nm
        self.msa_prm_gen.focus_spread = 4                        # Focus half-spread in nm
        self.msa_prm_gen.tilt_x = 0
        self.msa_prm_gen.tilt_y = 0
        self.msa_prm_gen.h_scan_frame_size = self.a                   # This is the size of the cel in nm
        self.msa_prm_gen.v_scan_frame_size = self.b 
        self.msa_prm_gen.scan_columns = self.nx                       # Unsure if matters but consistent with image
        self.msa_prm_gen.scan_rows = self.ny  
        self.msa_prm_gen.temp_coherence_flag = 0                 # Turn off temporal coherence calculation (STEM only)
        self.msa_prm_gen.spat_coherence_flag = 0                 # Turn off spatial coherence calculation (STEM only)
        self.msa_prm_gen.slice_files = ''                        # String of slice file, will be set iteratively later
        self.msa_prm_gen.number_of_slices = self.nz                   # Load one slice at a time
        self.msa_prm_gen.det_readout_period = 0                  # No detector effects included.
        self.msa_prm_gen.tot_number_of_slices = self.nz               # Each structure contains 155 slices
        self.msa_prm_gen.aberrations_dict = {1: (0, 0),          # Defocus of 0 nm standard, will be adjusted iteratively later
                                        5: (self.Cs, 0),         # Cs = -13 um
                                        11: (self.C5, 0)}        # C5 = 5 mm         
        #self.msa_prm_gen.save_msa_prm(r'home/thiago/Desktop/Argonne/ingrained_test/Cif_model_examples/Cif_model_examples/test/test_new_script/output/init_prm/MsaPrm_parallel_Initialized.prm') # Save the initailized MSA prm file
        self.msa_prm_gen.save_msa_prm(os.path.join(output_dir_path, 'init_prm', 'MsaPrm_parallel_Initialized.prm'))

        # Initizliae general WavImg Parameter File
        #self.wav_prm_gen = drp.wavimgprm.WavimgPrm()
        self.wav_prm_gen.high_tension = self.ht                       # Set HT to 300 kV
        self.wav_prm_gen.wave_dim = (self.nx,self.ny)                      # Pixel dimensions of wave are nx by ny
        self.wav_prm_gen.wave_sampling = (self.a/self.nx, self.b/self.ny)            # Pixel size is width of cell divided by pixel dimensions
        self.wav_prm_gen.output_format = 0                       # Output TEM image
        self.wav_prm_gen.output_dim = (self.nx,self.ny)                    # Pixel dimensions of image are nx by ny
        self.wav_prm_gen.coherence_model = 2                     # Explicit TCC calculation
        self.wav_prm_gen.temp_coherence = (1,4)                  # Turn on temporal coherence. Focal spread half-width of 4 nm
        self.wav_prm_gen.spat_coherence = (1,0.2)                # Turn on spatial coherence. 2nd number = Beam convergence half angle of 0.2 mrad
        self.wav_prm_gen.mtf = (0, 1, r'E:\MTF-US2k-300.mtf')    # Turn off detector MTF effect. Calculation scale of the mtf = (sampling rate experiment)/(sampling rate simulation)
        self.wav_prm_gen.vibration = (1, 0.05, 0.05, 0)          # 50 pm isotropic vibration applied.
        self.wav_prm_gen.oa_radius = 250                          # Objective aperture essentially out, set to 250 mrad
        self.wav_prm_gen.aberrations_dict = {1: (0, 0),          # Defocus of 0 nm standard, will be adjusted iteratively later
                                        5: (self.Cs, 0),         # Cs = -13 um
                                        11: (self.C5, 0)}        # C5 = 5 mm    
        #self.wav_prm_gen.save_wavimg_prm(r'home/thiago/Desktop/Argonne/ingrained_test/Cif_model_examples/Cif_model_examples/test/test_new_script/output/init_prm/WavPrm_Initialized.prm') # Save the initailized WavImg prm file)
        self.wav_prm_gen.save_wavimg_prm(os.path.join(output_dir_path, 'init_prm', 'MsaPrm_parallel_Initialized.prm'))
    
    def run_dr_probe(self,input_dir_path,output_dir_path):
        start = time.time()
        for self.cel_file in os.listdir(self.input_dir):   
            #self.cel_file=self.input_dir
        #for cel_file in os.listdir(input_dir):                          # For every structure in the input directory
        
            # Set-up parent directory for this structure
            self.structure_name = self.cel_file.strip('.cel')                                     # Remove the file extension to isolate the structure name
            self.structure_dir = os.path.join(output_dir_path, self.structure_name)                    # Name of directory for this structure, to be within the output directory
            os.makedirs(self.structure_dir,exist_ok=True)                                                     # Create structure's directory with name specified above


            ## 2.1) Create back-up of cel file in output directory
            # Set-up cel sub-directory
            self.structure_cel_dir = os.path.join(self.structure_dir, 'cel')                      # Set up name of cel sub-directory for this structure
            os.makedirs(self.structure_dir,exist_ok=True)                                                # Create cel directory with name specified above
            self.cel_original = r'"{}"'.format(os.path.join(input_dir_path, self.cel_file))            # Name of full directory to original cel file
            self.cel_copy = r'"{}"'.format(os.path.join(self.structure_cel_dir, self.cel_file))        # The r'"{}"' formatting is necessary to enclose the path in double quotes.
            
            #cel_file= r'/Users/ramon/Dropbox/Users/Ramon Manzorro/Big Data HDR/Data/Simulations/pt-1a-ceo2-111/input/pt-1a-ceo2-111.cel'
            
        #    # Create back-up cel
        #    drp.commands.cellmuncher(cel_original, cel_copy, 
        #                             output=True)                                      # Save the cel file in the copy directory
        #    
            
            
            ## 2.2) Slice cel and save slices in \slc directory
            # Set-up slice sub-directory
            self.structure_slc_dir = os.path.join(self.structure_dir, 'slc')                      # Name of directory for this structure's slices
            os.makedirs(self.structure_slc_dir,exist_ok=True)                                                   # Create the structure's slice directory
            self.slice_name = self.structure_name + '_slc'                                        # The r'"{}"' formatting is unnecessary since the path doesn't contain spaces.
            self.slice_path_and_name = os.path.join(self.structure_slc_dir,self.slice_name)
            print(self.slice_name)
            print(self.slice_path_and_name)
            ## slice cel
            drp.commands.celslc(self.cel_original, self.slice_path_and_name,                      # Create the slices and save them in the slice directory
                            self.ht, self.nx, self.ny, self.nz, 
                            absorb=self.absorb, dwf=self.dwf, buni=self.buni, pot=True,
                            output=True)
            
            ## 2.3) Perform multislice simulation
            # Parameter initialization
            self.structure_prm_dir = os.path.join(self.structure_dir, 'prm')                      # Set up name of parameter (or 'prm') sub-directory for this structure
            os.makedirs(self.structure_prm_dir,exist_ok=True)                                                # Create parameter directory with name specified above
            self.msa_prm = self.msa_prm_gen                                                       # Load general parameter file to be edited
            self.msa_prm.slice_files = self.slice_path_and_name                                   # Specify location of phase gratings generated in section 2.2
            self.msa_prm_name = 'MsaPrm_'+self.structure_name+'.prm'                              # Name the MSA parameter file to be saved
            self.msa_prm_path_and_name = os.path.join(self.structure_prm_dir, self.msa_prm_name)       # Path location to where the MSA parameter file should be saved 
            self.msa_prm.save_msa_prm(self.msa_prm_path_and_name)                                 # Save the parameter file with the slice locations
            
            # Set-up wave function sub-directory
            self.structure_wav_dir = os.path.join(self.structure_dir, 'wav')                      # Set up name of wave function (or 'wav') sub-directory for this structure
            os.makedirs(self.structure_wav_dir,exist_ok=True)                                                  # Create wave function directory with name specified above
            self.wav_path = os.path.join(self.structure_wav_dir,self.structure_name)                   # Specify name (and path) of output wavefunction
            print(self.msa_prm_path_and_name)
            print(self.wav_path)
            # Calculate exit surface wavefunction
            drp.commands.msa(self.msa_prm_path_and_name, self.wav_path,                           # Calculates the exit wavefunction and saves it in wav_path
                            ctem = True, output = True, 
                            silent = False)
            
            
            ## 2.4) Generate simulated images
            # Parameter initialization
            self.wav_prm = self.wav_prm_gen                                                       # Load general wav parameter file for image simulation
            self.wav_prm.wave_files = self.wav_path + '_sl' + str(self.nz) + '.wav'                    # Name of wave file and location after full multislice simulation (if nz bigger than or equal to 100) 
            #wav_prm.wave_files = wav_path + '_sl0' + str(nz) + '.wav'         #Use this if nz lower than 100           # Name of wave file and location after full multislice simulation (if nz smaller than 100)
            self.wav_prm_name = 'WavPrm_'+self.structure_name+'.prm'                              # Set name of wave paramter file
            self.wav_prm_path_and_name = os.path.join(self.structure_prm_dir, self.wav_prm_name)       # Path location to where the wav parameter file should be saved
            self.wav_prm.save_wavimg_prm(self.wav_prm_path_and_name)                              # Save the parameter file with the slice locations
            # Set-up image sub-directory
            self.structure_img_dir = os.path.join(self.structure_dir, 'img')                      # Set up name of image (or 'img') sub-directory for this structure
            os.makedirs(self.structure_img_dir,exist_ok=True)                                                   # Create image function directory with name specified above
            
            # Simulate images
            for defocus in self.defoci:
                    # Specify name (img_name) of path (output_img) of output image
                    if self.nz >= 100:
                        self.img_name = self.structure_name+'_'+ str(self.nz)+'slc_' +str(self.nx)+'x'+str(self.ny)+'_'+str(defocus)+'nmDefocus'+'.dat' # if nz bigger than or equal to 100 
                    else:
                        
                        self.img_name = self.structure_name+'_'+ str(self.nz)+'slc_0' +str(self.nx)+'x'+str(self.ny)+'_'+str(defocus)+'nmDefocus'+'.dat' # if nz smaller than 100
                    
                    self.output_img = os.path.join(self.structure_img_dir,self.img_name)
                    self.dict_paras[defocus]['img_name'] = self.img_name.strip('.dat')
                    self.dict_paras[defocus]['defocus'] = defocus
                    # Calculate image
                    drp.commands.wavimg(self.wav_prm_path_and_name, self.output_img, 
                                        foc = defocus,
                                        sil = False, output=True)
                    
                    
                    # Save every 1 in 10 images randomly for diagnostics
                    #if randint(0,100) > 90:
                        #shutil.copyfile(output_img, os.path.join(output_check_dir,img_name))
            
            ## 2.5) Clean up slice directory
            # Delete slice sub-directory to save space
            try:
                shutil.rmtree(self.structure_slc_dir)
            except OSError as e:
                print ("Error: %s - %s." % (e.filename, e.strerror))


                
            end = time.time()
            print(end - start)
        
    def figure_generation(self,exp_img):
        #self.input_dir = input_dir_img
        self.exp_img = exp_img
        self.output_folder = os.path.join(self.structure_dir, 'defocus_images')    # Name of the folder to save the images
        #/home/thiago/Desktop/Argonne/ingrained_test/Cif_model_examples/Cif_model_examples/test/test_new_script/output/Pt_6layer_modified/img'
        # Create the output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)

        self.list_img_path = []
        for dat_file in os.listdir(self.structure_img_dir):
            if dat_file.endswith(".dat") and not dat_file.startswith("._"):
                self.img_file_name = dat_file.strip('.dat')
                self.dat_file_path = os.path.join(self.structure_img_dir, dat_file)
                self.dat_file_data = np.fromfile(self.dat_file_path, dtype=np.float32)
                
                self.dat_file_data_r = np.reshape(self.dat_file_data, self.exp_img.shape) #was (850, 850), but we need generalized shape
                self.dat_file_data_r = cv2.flip(self.dat_file_data_r, 0)
                self.scaled = self.dat_file_data_r * 255  # Scale the data to the range of 0-255


                self.save_path = os.path.join(self.output_folder, f"{self.img_file_name}.png")
                plt.imsave(self.save_path,self.scaled)

                # plt.figure(figsize=(4,3), dpi=45)
                # plt.axis('off')  # Turn off the axis; optional. Depends on your preference
                # plt.imshow(self.scaled, cmap='gray')
                # # dont set the title to ensure a pure figure
                # # plt.title(self.img_file_name)  # Set the title of the figure
                # self.save_path = os.path.join(self.output_folder, f"{self.img_file_name}.png")
                
                # plt.savefig(self.save_path,bbox_inches='tight', pad_inches=0)  # Save the figure as a PNG 

                for defocus, value in self.dict_paras.items():
                    if value.get('img_name') == self.img_file_name:
                        self.dict_paras[defocus]['img_file_path']=self.save_path

                self.list_img_path.append(self.save_path)
                plt.show() # Remove this line if you do not want the output figures to be shown 
                plt.close()  # Close the current figure
        # print(self.list_figure)

    def scale_pixels(self,img, mode=None):
            """
            Select a pixel scaling technique

            Args:
                mode: (string) pixel scaling technique
                rescale    :  stretch the pixel intensities so to fill the 
                                range from 0 to 1 (float64)
                center     :  enforce zero mean, unit variance (float64)
                grayscale  :  stretch the pixel intensities so to fill the 
                                range from 0 to 255 (uint8)

            Returns:
                A numpy array (either float64 or uint8) of the scaled image.
            """
            img = img.astype(np.float64)
            if mode == None:
                return img
            elif mode == "rescale":
                return ((img - img.min()) / (img.max() - img.min()) + 1e-16).astype(np.float64)
            elif mode == "center":
                return ((img - img.mean()) / (img.std())).astype(np.float64)
            elif mode == "grayscale":
                return (255 * (img - img.min()) / (img.max() - img.min()) + 1e-16).astype(
                    np.uint8
            )

    def score_ssim(self,img1, img2):
            """
            Compute the mean structural similarity index between two images

            Args:
                    img1, img2: (ndarray) images

            Return:
                    1 - the mean structural similarity index (i.e. ΔSSIM)
            """

            #data_range = 255  # Dynamic range of pixel values in typical images

            img1 = self.scale_pixels(img1, mode="rescale")  # You need to define the scale_pixels function
            img2 = self.scale_pixels(img2, mode="rescale")
            im_max, im_min = max(img1.max(), img2.max()), min(img1.min(), img2.min())
            return 1 - ssim(img1, img2, data_range=im_max - im_min)  # You need to import the ssim function

    def crop_img(self,dr_probe_img):
        '''
        only need to crop when ratio(exp img) != ratio(cell, or simulated img)
        '''
        simulated_blobs = blob_dog(dr_probe_img, max_sigma=30, threshold=0.1)
        simulated_central_point = np.mean(simulated_blobs, axis=0)[:2].astype(int)
        # Specify the central point (201, 191)
        center_x, center_y = simulated_central_point[1],simulated_central_point[0]
        # Specify the desired width and height of the cropped region
        desired_width = self.exp_img.shape[1]  # must be same as exp image pixels
        desired_height = self.exp_img.shape[0]  # Corresponding to 4:3 aspect ratio
        # Calculate the top-left corner coordinates of the cropped region
        start_x = max(0, center_x - desired_width // 2)
        start_y = max(0, center_y - desired_height // 2)
        # Calculate the width and height of the actual cropped region
        actual_width = min(desired_width, dr_probe_img.shape[1] - start_x)
        actual_height = min(desired_height, dr_probe_img.shape[0] - start_y)
        # Perform the cropping
        cropped_image = dr_probe_img[start_y:start_y + actual_height, start_x:start_x + actual_width]
        # cv2.imwrite("cropped_image.png", cropped_image)
        plt.imshow(cropped_image,cmap='gray')
        plt.axis('off')
        plt.show()

        cropped_image = dr_probe_img
        start_x,start_y, actual_width, actual_height = 0,0,180,135
        return cropped_image, [start_x,start_y, actual_width, actual_height]

    def zoom_img(self,img,zoom_factor):
        zoomed_img = img[round(0.5*img.shape[0]*(zoom_factor-1)/zoom_factor):round(0.5*img.shape[0]*(zoom_factor+1)/zoom_factor),round(0.5*img.shape[1]*(zoom_factor-1)/zoom_factor):round(0.5*img.shape[1]*(zoom_factor+1)/zoom_factor)]
        zoomed_img = cv2.resize(zoomed_img,(img.shape[1],img.shape[0]))
        return zoomed_img
    
    def translate_img(self,img,delta_x,delta_y):
        height, width = img.shape[:2]

        # Move the first 5 pixels along the x-axis to the end of the image
        shifted_image_x = np.concatenate((img[:, delta_x:], img[:, :delta_x]), axis=1)

        # Move the first 7 pixels along the y-axis to the end of the image
        shifted_image_xy = np.concatenate((shifted_image_x[delta_y:], shifted_image_x[:delta_y]), axis=0)
        return shifted_image_xy
    
    def rotate_img(self,img,angle):
        height, width = img.shape[:2]
        # Calculate the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
        # Perform the rotation
        rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height),borderValue=int(img[0][0]))
        return rotated_img

    def evaluate_mismatch(self,x0,*args):
        zoom_factor = x0[0]
        delta_x = int(x0[1])
        delta_y = int(x0[2])
        angle = x0[3]
        img,exp_img = args
        # modify img
        # img, start_x,start_y, actual_width, actual_height = self.crop_img(img)
        zoomed_img = self.zoom_img(img,zoom_factor=zoom_factor)
        translated_img = self.translate_img(zoomed_img,delta_x,delta_y)
        rotated_img = self.rotate_img(translated_img,angle)
        # calc mismatch = 1-ssim
        (ssim_value, diff) = ssim(rotated_img, exp_img, full=True)
        # print(x0)
        # print(f'zoom_factor:{zoom_factor}, delta_x:{delta_x},delta_y:{delta_y},angle:{angle},mismatch:{1-ssim_value} ')
        return 1-ssim_value
    
    def optimize_postprocess(self,img):
        #find optimized crop+zoom+rotate+translation paras for a certain defocus img

        # Initialize an empty list to store the iterations
        initial_x0 = [1.01,1,1,1. ]
        bounds_x0 = ( # should be variable in input.yaml, cuz all values based on initial guess results. e.g. initial guess gives optimized zoom factor=1.46, then give a boundary like 1.2~1.7
            (0.8,1.8),
            (-10,15),
            (-10,15),
            (-6.0,6.0)
        )
        res = differential_evolution(self.evaluate_mismatch, args=(self.img,self.exp_img, ), bounds=bounds_x0,integrality=[0,1,1,0],disp=True)
        return res

    def optimize_all(self,exp_img):
        # find optimized defocus para, in which we also find optimized crop+zoom+rotate+translation paras for each defocus para
        self.exp_img = exp_img
        for defocus, value in self.dict_paras.items():
            img_file_path = self.dict_paras[defocus]['img_file_path']
            self.img = cv2.imread(img_file_path, cv2.IMREAD_GRAYSCALE)
            # crop doesnt require optimization, so put it before zoom/translation optimization
            print(self.img.shape, self.exp_img.shape)
            self.img, crop_factor = self.crop_img(self.img)
            print(self.img.shape, self.exp_img.shape)
            res = self.optimize_postprocess(self.img)
            self.dict_paras[defocus]['crop_factor'] = crop_factor
            self.dict_paras[defocus]['zoom_factor'] = res.x[0]
            self.dict_paras[defocus]['delta_x'] = int(res.x[1])
            self.dict_paras[defocus]['delta_y'] = int(res.x[2])
            self.dict_paras[defocus]['angle'] = res.x[3]
            self.dict_paras[defocus]['mismatch'] = res.fun
        # print(self.dict_paras)

    
    def save_optimized_img(self):
        for defocus, value in self.dict_paras.items():
            img_file_path = self.dict_paras[defocus]['img_file_path']
            img = cv2.imread(img_file_path, cv2.IMREAD_GRAYSCALE)
            [start_x,start_y, actual_width, actual_height] = self.dict_paras[defocus]['crop_factor']
            zoom_factor = self.dict_paras[defocus]['zoom_factor']
            delta_x = self.dict_paras[defocus]['delta_x']
            delta_y = self.dict_paras[defocus]['delta_y']
            angle = self.dict_paras[defocus]['angle']

            cropped_img = img[start_y:start_y + actual_height, start_x:start_x + actual_width]
            zoomed_img = self.zoom_img(cropped_img,zoom_factor=zoom_factor)
            translated_img = self.translate_img(zoomed_img,delta_x=delta_x,delta_y=delta_y)
            rotated_img = self.rotate_img(translated_img,angle=angle)

            # overlap exp img and simulated img together and see how their blobs overlap
            experimental_blobs = blob_dog(self.exp_img, max_sigma=30, threshold=0.04)
            # Detect blobs in the simulated image using Difference of Gaussian (DoG)
            simulated_blobs = blob_dog(rotated_img, max_sigma=30, threshold=0.04)

            plt.figure(figsize=(18, 6))
            plt.title(f'after alignment,mismatch={round(1-ssim(rotated_img, self.exp_img, full=True)[0],2)},zoom_factor:{round(zoom_factor,2)}, delta_x:{delta_x},delta_y:{delta_y},angle:{round(angle,2)}')
            plt.axis('off')
            plt.subplot(1, 3, 1)
            plt.title('exp img')
            plt.imshow(self.exp_img, cmap='gray')
            for blob in experimental_blobs:
                y, x, _ = blob
                plt.plot(x, y,'ro', markersize=5)  # Red points for experimental atoms

            # Plot simulated image
            plt.subplot(1, 3, 2)
            plt.title('processed simulated img')
            plt.imshow(rotated_img, cmap='gray')
            for blob in simulated_blobs:
                y, x, _ = blob
                plt.plot(x, y, 'go', markersize=5)  # Green points for simulated atoms

            # Plot rotated simulated image
            plt.subplot(1, 3, 3)
            plt.title('two img overlapping')
            plt.imshow(self.exp_img, cmap='gray',alpha=0.5)
            for blob in experimental_blobs:
                y, x, _ = blob
                plt.plot(x, y,'ro', markersize=5)  # Red points for experimental atoms

            plt.imshow(rotated_img, cmap='gray',alpha=0.5)
            for blob in simulated_blobs:
                y, x, _ = blob
                plt.plot(x, y, 'go', markersize=5)  # Green points for simulated atoms
            plt.savefig(f'{self.output_dir}defocus{defocus}.png',bbox_inches='tight', pad_inches=0)

    def find_best_paras(self):
        min_mismatch = float('inf')
        for defocus,value in self.dict_paras.items():
            float('inf')
            if value['mismatch'] < min_mismatch:
                min_mismatch = value['mismatch']
                min_key = defocus
        print('minimized mismatch:', self.dict_paras[min_key])
        return self.dict_paras[min_key]
