@echo off
setlocal enabledelayedexpansion

REM Run only the Oort baseline and place it into an existing main experiment tag.
REM This avoids rerunning Ours/FedAvg/FedProx/PoC when only the Oort row is missing.

set "PROJECT_ROOT=%~dp0..\.."
cd /d "%PROJECT_ROOT%"

set CONDA_ENV=

if not "%CONDA_ENV%"=="" (
  call conda activate %CONDA_ENV%
  if errorlevel 1 exit /b 1
)

python -c "import sys; print(sys.executable)"
if errorlevel 1 (
  echo Python is not available. Please activate your conda environment first.
  exit /b 1
)

if not exist logs mkdir logs
if not exist save mkdir save

REM Change this tag if you want to attach Oort to another main run.
set TARGET_MAIN_TAG=20260510_105527

set DATASET=cifar
set MODEL=cnn
set EPOCHS=100
set NUM_USERS=100
set NUM_SELECTED=5
set LOCAL_EP=2
set LOCAL_BS=32
set LR=0.01
set ALPHA=0.1
set TEST_SIZE=10000
set GPU_ID=0
set DP_CLIP_NORM=1.0
set CHANNEL_SIGMA=0.1
set DP_COMMON=--dp_advanced --dp_noise_schedule constant --dp_adaptive_clip --dp_clip_scope layer --dp_clip_percentile 80 --dp_clip_ema 0.8 --dp_clip_growth 1.2 --dp_min_clip_norm 0.05 --dp_max_clip_norm 1.0 --dp_channel_assisted --dp_channel_mode channel_only --dp_channel_gain_cap 2.0
set DP_ARGS=--privacy_mode central --dp_clip_norm %DP_CLIP_NORM% %DP_COMMON% --dp_noise_multiplier 0.0 --dp_channel_noise_multiplier %CHANNEL_SIGMA%

echo ========================================
echo Oort-only main baseline run
echo Target: save\main\%TARGET_MAIN_TAG%
echo Seeds: 42, 123, 2024
echo ========================================

cd /d "%PROJECT_ROOT%\src"

for %%S in (42 123 2024) do (
  set OUT=main\%TARGET_MAIN_TAG%\seed%%S

  echo.
  echo [Oort] Seed %%S, output: !OUT!
  python federated_main.py ^
    --dataset %DATASET% --model %MODEL% --epochs %EPOCHS% ^
    --num_users %NUM_USERS% --num_selected %NUM_SELECTED% ^
    --local_ep %LOCAL_EP% --local_bs %LOCAL_BS% --lr %LR% ^
    --dirichlet_alpha %ALPHA% --seed %%S --test_size %TEST_SIZE% ^
    --gpu %GPU_ID% ^
    --no_shapley --selection_method oort ^
    --use_energy --sigma_squared 1.0 --initial_energy 500.0 --energy_threshold 50.0 ^
    %DP_ARGS% ^
    --output_folder !OUT!
  if errorlevel 1 goto failed
)

cd /d "%PROJECT_ROOT%"
echo.
echo Oort baseline finished. Now aggregate with:
echo python src\aggregate_multiseed.py --tag %TARGET_MAIN_TAG%
goto end

:failed
echo.
echo Oort baseline failed. Check the latest output above.
exit /b 1

:end
endlocal
