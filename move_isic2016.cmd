@echo off
REM Ensure target folders exist
mkdir dataset_raw\ISIC_2016\masks
mkdir dataset_raw\ISIC_2016\superpixels

REM Move segmentation masks
move images\*_Segmentation.png dataset_raw\ISIC_2016\masks

REM Move superpixels
move images\*_superpixels.png dataset_raw\ISIC_2016\superpixels

echo Files moved successfully!
pause
