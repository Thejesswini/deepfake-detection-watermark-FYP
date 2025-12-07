# deepfake-detection-watermark-FYP
use reversible neural networks to embed and extract watermarks which can later be used for deepfake detection and localization


so in this branch - localisation_heatmap
extrcat_and_coompare_1.py - takes 2 images in directory format directly (to etract watermarks)
local.py - takes 2 images in directory format directory ( for visualisation of points of modification)


under unwatermarked / deepfake directory  : 
unwatermarked/deepfake/uw2_deep.png - this image is the deepfake modified image so use this if u want to test for unwatermarked image producing deepfakes 

under unwatermarked directory
unwatermarked/unw_2.jpg - this is the orignal unwatermarked image taken directly from the dataset
unwatermarked/unw_1.png - this is the original source unwatermarked image taken directly from the dataset

these are the output directories of the images being produced when you run extract_and_compare1.py script 
1.unwatermarked/deepfake_unwatermarked_wm.png
2.unwatermarked/orig_unwatermarked_wm.png

this branch has extract_and_compare.py and visualize_localisation.py modified.

things to do if u want more clarity on results between unwatermarked:
try for more images -  I tried only for 1 image unwatermarked all 4 cases :
prolly change the config files for striking balace between inpercibiity and extrcation also changing epochs to see if things changes(embedding)

