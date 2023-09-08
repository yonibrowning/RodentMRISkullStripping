import numpy as np
import os
from pathlib import Path
import SimpleITK as sitk
from keras.models import load_model
from aind_mri_utils.file_io.image_files import read_image

from RodentMRISkullStripping3D.rbm3.core.paras import PreParas, KerasParas
from RodentMRISkullStripping3D.rbm3.core.dice import dice_coef,dice_coef_loss
from RodentMRISkullStripping3D.rbm3.scripts.rbm3 import out_LabelHot_map_3D
from RodentMRISkullStripping3D.rbm3.core.utils import min_max_normalization, resample_img


def load_data_folder(folder,resolution =.1,use_outline = False):
    """
    Load data from folder. Folder must contain one and only one of each of:
    (1) A .nii file with the with unprocessed data that includes the keyword "raw"
    (2) A .nii file with the brain mask, which includes the keyword "brain"
    Optionally, there can also be a 
    (3) A .nii file that contains the keyword "outline." This will allow seperate outline labeling,
    BUT it is really an antiquated thing I tried and isn't needed. It will only be used
    if the variable "use_outline" is true
    
    Additional input resolution specifies the image resolution (some .nii files do not include this).
    Default is .1(mm)
    
    Outputs:
    Img = sitk Image with raw nii information
    Lbl = sitk Image with training labels for Img
    """
    fls = os.listdir(folder)
    for ii,flname in enumerate(fls):
        if 'raw' in flname:
            Img = read_image(os.path.join(folder,flname))
            #Img.SetSpacing([resolution]*3)
        if ('brain' in flname) or ('skull_strip' in flname):
            print(os.path.join(folder,flname))
            Brn = read_image(os.path.join(folder,flname))
            #Brn.SetSpacing([resolution]*3)
            brn = sitk.GetArrayFromImage(Brn)
            brn = (brn-np.min(brn))/(np.max(brn)-np.min(brn))
            brn = (brn>.5).astype(np.float32)
        if 'outline' in flname and use_outline:
            Out = read_image(os.path.join(folder,flname))
            #Out.SetSpacing([resolution]*3)
            out = sitk.GetArrayFromImage(Out)
            out = (out-np.min(out))/(np.max(out)-np.min(out))
            out = (out>.5).astype(np.float32)
        
    lbl = np.zeros(brn.shape,dtype = np.float32)
    lbl[brn>0]=1
    if use_outline:
        lbl[out>0] = 2
    Lbl = sitk.GetImageFromArray(lbl)
    Lbl.SetSpacing([resolution]*3)
    return Img,Lbl

def get_dataset_partitions(X,Y, train_split=0.8, val_split=0.2, test_split=0, shuffle=True):
    """
    Partitions generic data X and Y into training, testing, and validation sets
    Inputs:
    X: list of data
    Y: list of labels corresponding to X
    Inputs (optional):
    train_split: (default .8) fraction of X/Y for training data
    val_split: (default .2) fraction of X/Y for validation data
    test_split: (default 0) fraction of X/Y for testing data
    shuffle: (default True) randomize order? (false keeps input order)
    
    Output:
    train_x, train_y, val_x,val_y,test_x,test_y: Pairs of X/Y subsets for each of training/validation/test sets
    """
    assert (train_split + test_split + val_split) == 1
    assert(len(X)==len(Y))
    X = np.array(X)
    Y = np.array(Y)
    
    if shuffle:
        # Specify seed to always have the same split distribution between runs
        order = np.random.permutation(len(X))
    else:
        order = np.arange(0,len(X))
            
    train_size = int(train_split * len(X))
    val_size = int(val_split * len(X))
    test_size = len(X)-train_size-val_size
    
    # Train
    train_x = X[order[:train_size]]
    train_y = Y[order[:train_size]]

    # validate
    val_x = X[order[train_size:train_size+val_size]]
    val_y = Y[order[train_size:train_size+val_size]]
    
    # test
    test_x = X[order[-test_size:]]
    test_y = Y[order[-test_size:]]
    
    return train_x, train_y, val_x,val_y,test_x,test_y

def resample_spacing(Img,Lbl,newSpacing,image_interpolator = sitk.sitkLinear,label_interpolator = sitk.sitkNearestNeighbor):
    """
    Resample sitk Image and labels to a new spacing. Options allow image and labels to use different interpolators
    """
    re_Img = resample_img(Img,
                            new_spacing=newSpacing,
                            interpolator=image_interpolator)
    re_img = sitk.GetArrayFromImage(re_Img)
    re_img = min_max_normalization(re_img)

    re_Lbl = resample_img(Lbl,
                            new_spacing=newSpacing,
                            interpolator=label_interpolator)
    re_lbl = sitk.GetArrayFromImage(re_Lbl)
    re_lbl = (re_lbl-np.min(re_lbl))/(np.max(re_lbl)-np.min(re_lbl))
    re_lbl = (re_lbl>.5).astype(np.float32)
    
    return re_img,re_lbl

def get_3d_paras(model_path =  Path(r'C:/Users/yoni.browning/OneDrive - Allen Institute/Documents/GitHub/RodentMRISkullStripping/RodentMRISkullStripping3D/rbm3/scripts')/Path('rat_brain-3d_unet.hdf5')):
    pre_paras = PreParas()
    pre_paras.patch_dims = [64, 64, 64]
    pre_paras.patch_label_dims = [64, 64, 64]
    pre_paras.patch_strides = [16, 16, 16]
    pre_paras.n_class = 2
    pre_paras.issubtract = 0
    pre_paras.organids = [1]

    keras_paras = KerasParas()
    keras_paras.outID = 0
    keras_paras.thd = 0.5
    keras_paras.loss = 'dice_coef_loss'
    keras_paras.img_format = 'channels_last'
    keras_paras.model_path = model_path

    return pre_paras, keras_paras

def keep_only_largest_island(Image):
    # Find connected regions of image
    CC = sitk.ConnectedComponent(Image)
    cc = sitk.GetArrayFromImage(CC)
    labels,counts = np.unique(cc,return_counts=True)

    # Eliminate zeros.
    counts = counts[labels>0]
    labels = labels[labels>0]
    mx = labels[np.argmax(counts)]

    # Update labels, keeping only the largest.
    cc[cc!=mx] = 0
    cc[cc==mx] = 1

    # create a new sitk object
    Mask = sitk.GetImageFromArray(cc)
    Mask.CopyInformation(Image)
    
    # Cast as 
    cif = sitk.CastImageFilter()
    cif.SetOutputPixelType(sitk.sitkInt8)
    Mask = cif.Execute(Mask)
    
    # and return
    return Mask


def skull_strip_file(input_path,output_path,voxsize = 0.1,model_path =r'C:\Users\yoni.browning\OneDrive - Allen Institute\Documents\GitHub\RodentMRISkullStripping\yoniModelRetrain\long_train_unet_3d-ROUND2_6Brain.h5' ):
    assert voxsize==.1, "Voxel size must equal .1; code to change voxel size is not implemented in this function"
    
    # load model
    seg_net = load_model(model_path,
                         custom_objects={'dice_coef_loss': dice_coef_loss,
                                         'dice_coef': dice_coef})
    
    # Read image
    imgobj = read_image(input_path)
    img_array = sitk.GetArrayFromImage(imgobj)

    # Normalize image to match training data
    normed_array = min_max_normalization(img_array)

    # Do Labeling
    pre_paras,keras_paras = get_3d_paras(model_path = model_path)
    out_label_map, out_likelihood_map = out_LabelHot_map_3D(normed_array,
                                                            seg_net,
                                                            pre_paras,
                                                            keras_paras)
    # Get labeling outputs as sitk image
    out_label_img = sitk.GetImageFromArray(out_label_map.astype(np.uint8))
    out_label_img.CopyInformation(imgobj)

    # Get the largest labeled object
    mask_img = keep_only_largest_island(out_label_img)
    #mask_img = out_label_img
    
    # Save the results
    sitk.WriteImage(mask_img, output_path)
    
    return mask_img,imgobj