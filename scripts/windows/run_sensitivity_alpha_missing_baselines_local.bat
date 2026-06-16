@echo off
setlocal enabledelayedexpansion

REM Supplement an existing non-IID alpha sensitivity run with the missing
REM FedProx and Oort baselines. Existing Ours/FedAvg/PoC results are untouched.

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

set TARGET_TAG=20260510_153941

set DATASET=cifar
set MODEL=cnn
set EPOCHS=100
set NUM_USERS=100
set NUM_SELECTED=5
set LOCAL_EP=2
set LOCAL_BS=32
set LR=0.01
set TEST_SIZE=10000
set SEED=42
set GPU_ID=0
set DP_CLIP_NORM=1.0
set CHANNEL_SIGMA=0.1
set DP_COMMON=--dp_advanced --dp_noise_schedule constant --dp_adaptive_clip --dp_clip_scope layer --dp_clip_percentile 80 --dp_clip_ema 0.8 --dp_clip_growth 1.2 --dp_min_clip_norm 0.05 --dp_max_clip_norm 1.0 --dp_channel_assisted --dp_channel_mode channel_only --dp_channel_gain_cap 2.0
set DP_ARGS=--privacy_mode central --dp_clip_norm %DP_CLIP_NORM% %DP_COMMON% --dp_noise_multiplier 0.0 --dp_channel_noise_multiplier %CHANNEL_SIGMA%

echo ========================================
echo Supplement alpha sensitivity baselines
echo Target: save\sensitivity_alpha\%TARGET_TAG%
echo Missing methods: FedProx, Oort
echo Alphas: 0.1, 0.25, 0.5, 1.0
echo ========================================

cd /d "%PROJECT_ROOT%\src"

for %%A in (0.1 0.25 0.5 1.0) do (
  set OUT=sensitivity_alpha\%TARGET_TAG%\alpha%%A

  echo.
  echo ========================================
  echo Alpha %%A
  echo Output folder: !OUT!
  echo ========================================

  echo [1/2] FedProx
  python federated_main.py ^
    --dataset %DATASET% --model %MODEL% --epochs %EPOCHS% ^
    --num_users %NUM_USERS% --num_selected %NUM_SELECTED% ^
    --local_ep %LOCAL_EP% --local_bs %LOCAL_BS% --lr %LR% ^
    --dirichlet_alpha %%A --seed %SEED% --test_size %TEST_SIZE% ^
    --gpu %GPU_ID% ^
    --no_shapley --selection_method random --use_fedprox --fedprox_mu 0.01 ^
    %DP_ARGS% ^
    --output_folder !OUT!
  if errorlevel 1 goto failed

  echo [2/2] Oort
  python federated_main.py ^
    --dataset %DATASET% --model %MODEL% --epochs %EPOCHS% ^
    --num_users %NUM_USERS% --num_selected %NUM_SELECTED% ^
    --local_ep %LOCAL_EP% --local_bs %LOCAL_BS% --lr %LR% ^
    --dirichlet_alpha %%A --seed %SEED% --test_size %TEST_SIZE% ^
    --gpu %GPU_ID% ^
    --no_shapley --selection_method oort ^
    --use_energy --sigma_squared 1.0 --initial_energy 500.0 --energy_threshold 50.0 ^
    %DP_ARGS% ^
    --output_folder !OUT!
  if errorlevel 1 goto failed
)

cd /d "%PROJECT_ROOT%"
echo.
echo Supplement finished. Regenerate the figure/table with:
echo python src\plot_sensitivity_alpha.py --tag %TARGET_TAG%
goto end

:failed
echo.
echo Supplement alpha sensitivity run failed. Check the latest output above.
exit /b 1

:end
endlocal
