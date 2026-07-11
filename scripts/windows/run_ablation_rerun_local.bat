@echo off
setlocal enabledelayedexpansion

REM Rerun corrected ablation experiments on local Windows.
REM This script uses the same core settings as the main table, but runs only ablations.
REM Optional overrides before running, for example:
REM   set EPOCHS=20
REM   set SEEDS=42
REM   set RUN_TAG=local_ablation_test

set "PROJECT_ROOT=%~dp0..\.."
cd /d "%PROJECT_ROOT%"

if not defined CONDA_ENV set "CONDA_ENV=flsv"
if not "%CONDA_ENV%"=="" (
  call conda activate %CONDA_ENV%
  if errorlevel 1 (
    echo Failed to activate conda environment "%CONDA_ENV%".
    echo Activate it manually or set CONDA_ENV= before running this script.
    exit /b 1
  )
)

python -c "import sys; print(sys.executable)"
if errorlevel 1 (
  echo Python is not available. Please check your conda environment.
  exit /b 1
)

if not exist logs mkdir logs
if not exist save mkdir save

if not defined DATASET set "DATASET=cifar"
if not defined MODEL set "MODEL=cnn"
if not defined EPOCHS set "EPOCHS=100"
if not defined NUM_USERS set "NUM_USERS=100"
if not defined NUM_SELECTED set "NUM_SELECTED=5"
if not defined LOCAL_EP set "LOCAL_EP=2"
if not defined LOCAL_BS set "LOCAL_BS=32"
if not defined LR set "LR=0.01"
if not defined DIRICHLET_ALPHA set "DIRICHLET_ALPHA=0.1"
if not defined TEST_SIZE set "TEST_SIZE=10000"
if not defined GPU_ID set "GPU_ID=0"
if not defined SEEDS set "SEEDS=42 123 2024"
if not defined RUN_TAG (
  for /f %%T in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "RUN_TAG=%%T"
)

if not defined DP_CLIP_NORM set "DP_CLIP_NORM=1.0"
if not defined CHANNEL_SIGMA set "CHANNEL_SIGMA=0.1"
if not defined SELECTION_BETA set "SELECTION_BETA=1.0"

set "SCHED_WEIGHTS=--sv_weight 0.7 --battery_weight 0.15 --channel_weight 0.15"
set "COMMON_DP_ARGS=--dp_advanced --dp_noise_schedule constant --dp_adaptive_clip --dp_clip_scope layer --dp_clip_percentile 80 --dp_clip_ema 0.8 --dp_clip_growth 1.2 --dp_min_clip_norm 0.05 --dp_max_clip_norm 1.0 --dp_channel_assisted --dp_channel_mode channel_only --dp_channel_gain_cap 2.0"
set "DP_ARGS=--privacy_mode central --dp_clip_norm %DP_CLIP_NORM% %COMMON_DP_ARGS% --dp_noise_multiplier 0.0 --dp_channel_noise_multiplier %CHANNEL_SIGMA%"
set "BASE_ARGS=--dataset %DATASET% --model %MODEL% --epochs %EPOCHS% --num_users %NUM_USERS% --num_selected %NUM_SELECTED% --local_ep %LOCAL_EP% --local_bs %LOCAL_BS% --lr %LR% --dirichlet_alpha %DIRICHLET_ALPHA% --test_size %TEST_SIZE% --gpu %GPU_ID%"
set "ENERGY_ARGS=--use_energy --sigma_squared 1.0 --initial_energy 500.0 --energy_threshold 50.0"
set "LYAP_ARGS=--use_lyapunov --lyapunov_V 10.0 --energy_budget 5.0 --selection_beta %SELECTION_BETA%"
set "SV_UPDATE_ARGS=--shapley_update_method mean --shapley_alpha 0.5"
set "SV_CC_ARGS=--shapley_estimator complementary --shapley_allocation neyman --shapley_pilot_samples 1 --shapley_max_iter 20"

echo.
echo ========================================
echo Corrected local ablation rerun
echo RUN_TAG=%RUN_TAG%
echo SEEDS=%SEEDS%
echo EPOCHS=%EPOCHS%, N=%NUM_USERS%, K=%NUM_SELECTED%, alpha=%DIRICHLET_ALPHA%
echo Output root: save\sv_supp\%RUN_TAG%\ablation
echo.
echo Ablations:
echo   Full      = SV + Energy + Lyapunov queue
echo   w/o SV    = Random + Energy + Lyapunov queue
echo   w/o Queue = SV + Energy + Lyapunov utility, no virtual-queue penalty
echo   w/o Energy= SV only
echo ========================================

cd /d "%PROJECT_ROOT%\src"

for %%S in (%SEEDS%) do (
  set "OUT=sv_supp\%RUN_TAG%\ablation\seed%%S"

  echo.
  echo ----------------------------------------
  echo Full: SV + Energy + Lyapunov, seed=%%S
  echo Output folder: !OUT!
  echo Start: %DATE% %TIME%
  echo ----------------------------------------
  python federated_main.py ^
    %BASE_ARGS% --seed %%S ^
    %SV_CC_ARGS% %SV_UPDATE_ARGS% ^
    %ENERGY_ARGS% %LYAP_ARGS% ^
    %SCHED_WEIGHTS% %DP_ARGS% ^
    --output_folder "!OUT!"
  if errorlevel 1 goto failed

  echo.
  echo ----------------------------------------
  echo w/o SV: random + Energy + Lyapunov, seed=%%S
  echo Output folder: !OUT!
  echo Start: %DATE% %TIME%
  echo ----------------------------------------
  python federated_main.py ^
    %BASE_ARGS% --seed %%S ^
    --no_shapley --selection_method random ^
    %ENERGY_ARGS% %LYAP_ARGS% ^
    %SCHED_WEIGHTS% %DP_ARGS% ^
    --output_folder "!OUT!"
  if errorlevel 1 goto failed

  echo.
  echo ----------------------------------------
  echo w/o Queue: SV + Energy + Lyapunov utility without queue penalty, seed=%%S
  echo Output folder: !OUT!
  echo Start: %DATE% %TIME%
  echo ----------------------------------------
  python federated_main.py ^
    %BASE_ARGS% --seed %%S ^
    %SV_CC_ARGS% %SV_UPDATE_ARGS% ^
    %ENERGY_ARGS% %LYAP_ARGS% --disable_queue_penalty ^
    %SCHED_WEIGHTS% %DP_ARGS% ^
    --output_folder "!OUT!"
  if errorlevel 1 goto failed

  echo.
  echo ----------------------------------------
  echo w/o Energy: SV only, seed=%%S
  echo Output folder: !OUT!
  echo Start: %DATE% %TIME%
  echo ----------------------------------------
  python federated_main.py ^
    %BASE_ARGS% --seed %%S ^
    %SV_CC_ARGS% %SV_UPDATE_ARGS% ^
    %DP_ARGS% ^
    --output_folder "!OUT!"
  if errorlevel 1 goto failed
)

cd /d "%PROJECT_ROOT%"
echo.
echo ========================================
echo Corrected local ablation rerun finished!
echo Results root: save\sv_supp\%RUN_TAG%\ablation
echo Aggregate with:
echo   python src\summarize_sv_supp_results.py --tag %RUN_TAG%
echo Output table:
echo   save\sv_supp\%RUN_TAG%\summary_tables\ablation_summary.csv
echo ========================================
goto end

:failed
cd /d "%PROJECT_ROOT%"
echo.
echo Corrected local ablation rerun failed. Check the latest output above.
exit /b 1

:end
endlocal
