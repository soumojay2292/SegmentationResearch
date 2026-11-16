@echo off

REM === ISIC2017 ===
mkdir dataset_raw\ISIC_2017\images
mkdir dataset_raw\ISIC_017\masks
mkdir dataset_raw\ISIC_2017\superpixels
move dataset_raw\ISIC_2017\*.jpg dataset_raw\ISIC_2017\images
move dataset_raw\ISIC_2017\*_Segmentation.png dataset_raw\ISIC_2017\masks
move dataset_raw\ISIC_2017\*_superpixels.png dataset_raw\ISIC_2017\superpixels

REM === ISIC2018 ===
mkdir dataset_raw\ISIC2018\images
mkdir dataset_raw\ISIC2018\masks
mkdir dataset_raw\ISIC2018\superpixels
move dataset_raw\ISIC2018\*.jpg dataset_raw\ISIC2018\images
move dataset_raw\ISIC2018\*_Segmentation.png dataset_raw\ISIC2018\masks
move dataset_raw\ISIC2018\*_superpixels.png dataset_raw\ISIC2018\superpixels

echo Sorting complete for ISIC2016–2018!
pause
