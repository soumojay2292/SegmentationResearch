@echo off
setlocal enabledelayedexpansion

:: ============================================================
::  run_all.bat — Sequential training of all segmentation models
::  Dataset  : ISIC_2018
::  Env      : segmentation (conda)
::  Run from : project root (where train.py lives)
:: ============================================================

:: ---------- activate conda environment ----------
call "%USERPROFILE%\miniconda3\condabin\conda.bat" activate segmentation
if errorlevel 1 (
    echo [ERROR] Failed to activate conda environment "segmentation". Aborting.
    exit /b 1
)

:: ---------- create logs folder ----------
if not exist logs mkdir logs

:: ---------- shared training args ----------
set DATASET=ISIC_2018
set EPOCHS=20
set IMG_SIZE=384
set BATCH=2
set LR=0.0005
set WORKERS=0
set BASE_ARGS=--dataset %DATASET% --epochs %EPOCHS% --img_size %IMG_SIZE% --batch_size %BATCH% --lr %LR% --workers %WORKERS%

set FAILED=

:: ============================================================
:: 1. U-Net
:: ============================================================
echo.
echo ============================================================
echo  [1/5] Training U-Net
echo ============================================================
python train.py --model unet %BASE_ARGS% --exp unet_run > logs\unet_log.txt 2>&1
if errorlevel 1 (
    echo [WARNING] U-Net training FAILED. Check logs\unet_log.txt
    set FAILED=%FAILED% unet
) else (
    echo [OK] U-Net finished. Log: logs\unet_log.txt
)

:: ============================================================
:: 2. Attention U-Net
:: ============================================================
echo.
echo ============================================================
echo  [2/5] Training Attention U-Net
echo ============================================================
python train.py --model attention_unet %BASE_ARGS% --exp attn_unet_run > logs\attn_log.txt 2>&1
if errorlevel 1 (
    echo [WARNING] Attention U-Net training FAILED. Check logs\attn_log.txt
    set FAILED=%FAILED% attention_unet
) else (
    echo [OK] Attention U-Net finished. Log: logs\attn_log.txt
)

:: ============================================================
:: 3. UNet++
:: ============================================================
echo.
echo ============================================================
echo  [3/5] Training UNet++
echo ============================================================
python train.py --model unetpp %BASE_ARGS% --exp unetpp_run > logs\unetpp_log.txt 2>&1
if errorlevel 1 (
    echo [WARNING] UNet++ training FAILED. Check logs\unetpp_log.txt
    set FAILED=%FAILED% unetpp
) else (
    echo [OK] UNet++ finished. Log: logs\unetpp_log.txt
)

:: ============================================================
:: 4. TransUNet
:: ============================================================
echo.
echo ============================================================
echo  [4/5] Training TransUNet
echo ============================================================
python train.py --model transunet %BASE_ARGS% --exp transunet_run > logs\transunet_log.txt 2>&1
if errorlevel 1 (
    echo [WARNING] TransUNet training FAILED. Check logs\transunet_log.txt
    set FAILED=%FAILED% transunet
) else (
    echo [OK] TransUNet finished. Log: logs\transunet_log.txt
)

:: ============================================================
:: 5. MAFFNet  (requires SAM2 checkpoint)
:: ============================================================
echo.
echo ============================================================
echo  [5/5] Training MAFFNet
echo ============================================================
python train.py --model maffnet %BASE_ARGS% --checkpoint checkpoints/sam2_hiera_large.pt --exp maffnet_run > logs\maffnet_log.txt 2>&1
if errorlevel 1 (
    echo [WARNING] MAFFNet training FAILED. Check logs\maffnet_log.txt
    set FAILED=%FAILED% maffnet
) else (
    echo [OK] MAFFNet finished. Log: logs\maffnet_log.txt
)

:: ============================================================
:: Summary
:: ============================================================
echo.
echo ============================================================
echo  All runs complete.
if defined FAILED (
    echo  FAILED models:%FAILED%
    echo  Check the corresponding log files in logs\ for details.
) else (
    echo  All 5 models trained successfully.
)
echo ============================================================

:: Uncomment the line below to shut down the PC after completion:
:: shutdown /s /t 60 /c "Training complete — shutting down in 60 seconds."

endlocal
