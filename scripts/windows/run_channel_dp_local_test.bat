@echo off
setlocal enabledelayedexpansion

REM Local Windows test for the channel-noise privacy-utility route.
REM It trains full-model updates, clips them to bound sensitivity,
REM then applies equivalent channel noise to the aggregate update.

set "PROJECT_ROOT=%~dp0..\.."
cd /d "%PROJECT_ROOT%"

REM If you already activated conda before running this bat, leave CONDA_ENV empty.
REM Example: set CONDA_ENV=flsv
set CONDA_ENV=

if not "%CONDA_ENV%"=="" (
  call conda activate %CONDA_ENV%
  if errorlevel 1 (
    echo Failed to activate conda env "%CONDA_ENV%".
    exit /b 1
  )
)

echo Python executable:
python -c "import sys; print(sys.executable)"
if errorlevel 1 (
  echo Python is not available. Please activate your conda environment first.
  exit /b 1
)

if not exist logs mkdir logs
if not exist save mkdir save

set DATASET=cifar
set MODEL=cnn
set EPOCHS=100
set NUM_USERS=100
set LOCAL_EP=1
set LOCAL_BS=64
set LR=0.01
set DIRICHLET_ALPHA=0.5
set TEST_SIZE=3000
set SEED=42
set DP_CLIP_NORM=1.0
set GPU_ID=0

set SCHED_WEIGHTS=--sv_weight 0.7 --battery_weight 0.15 --channel_weight 0.15
set DP_COMMON=--dp_advanced --dp_noise_schedule linear_increase --dp_noise_start_multiplier 0.7 --dp_adaptive_clip --dp_clip_scope layer --dp_clip_percentile 80 --dp_clip_ema 0.8 --dp_clip_growth 1.2 --dp_min_clip_norm 0.05 --dp_max_clip_norm 1.0 --dp_channel_assisted --dp_channel_mode channel_only --dp_channel_gain_cap 2.0

set RUN_TAG=%DATE:~0,4%%DATE:~5,2%%DATE:~8,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%
set RUN_TAG=%RUN_TAG: =0%
set EXP_ROOT=privacy_utility\%RUN_TAG%

echo ========================================
echo CIFAR-10 channel-only privacy-utility local test
echo Dataset: %DATASET%, epochs: %EPOCHS%, seed: %SEED%
echo GPU ID: %GPU_ID%
echo Full model is trained. This route uses only equivalent channel noise.
echo Output root: save\%EXP_ROOT%
echo ========================================

cd /d "%PROJECT_ROOT%\src"

echo.
echo [0/3] CIFAR raw baseline: K=20, alpha=0.5, no clipping, no privacy noise
python federated_main.py ^
  --dataset %DATASET% --model %MODEL% --epochs %EPOCHS% ^
  --num_users %NUM_USERS% --num_selected 20 ^
  --local_ep %LOCAL_EP% --local_bs %LOCAL_BS% --lr %LR% ^
  --dirichlet_alpha %DIRICHLET_ALPHA% --seed %SEED% --test_size %TEST_SIZE% ^
  --gpu %GPU_ID% ^
  --shapley_update_method mean --shapley_alpha 0.5 --shapley_max_iter 5 ^
  --use_energy --initial_energy 500.0 --energy_threshold 50.0 ^
  --use_lyapunov --lyapunov_V 10.0 --energy_budget 5.0 ^
  %SCHED_WEIGHTS% ^
  --privacy_mode none ^
  --output_folder %EXP_ROOT%\raw
if errorlevel 1 goto failed

echo.
echo [1/3] CIFAR clipping baseline: K=20, alpha=0.5, clipping only, no aggregate privacy noise
python federated_main.py ^
  --dataset %DATASET% --model %MODEL% --epochs %EPOCHS% ^
  --num_users %NUM_USERS% --num_selected 20 ^
  --local_ep %LOCAL_EP% --local_bs %LOCAL_BS% --lr %LR% ^
  --dirichlet_alpha %DIRICHLET_ALPHA% --seed %SEED% --test_size %TEST_SIZE% ^
  --gpu %GPU_ID% ^
  --shapley_update_method mean --shapley_alpha 0.5 --shapley_max_iter 5 ^
  --use_energy --initial_energy 500.0 --energy_threshold 50.0 ^
  --use_lyapunov --lyapunov_V 10.0 --energy_budget 5.0 ^
  %SCHED_WEIGHTS% ^
  --privacy_mode central --dp_clip_norm %DP_CLIP_NORM% %DP_COMMON% ^
  --dp_noise_multiplier 0.0 --dp_channel_noise_multiplier 0.0 ^
  --output_folder %EXP_ROOT%\clipping_only
if errorlevel 1 goto failed

echo.
echo [2/3] CIFAR K=20, alpha=0.5, channel-only sigma_ch=0.25
python federated_main.py ^
  --dataset %DATASET% --model %MODEL% --epochs %EPOCHS% ^
  --num_users %NUM_USERS% --num_selected 20 ^
  --local_ep %LOCAL_EP% --local_bs %LOCAL_BS% --lr %LR% ^
  --dirichlet_alpha %DIRICHLET_ALPHA% --seed %SEED% --test_size %TEST_SIZE% ^
  --gpu %GPU_ID% ^
  --shapley_update_method mean --shapley_alpha 0.5 --shapley_max_iter 5 ^
  --use_energy --initial_energy 500.0 --energy_threshold 50.0 ^
  --use_lyapunov --lyapunov_V 10.0 --energy_budget 5.0 ^
  %SCHED_WEIGHTS% ^
  --privacy_mode central --dp_clip_norm %DP_CLIP_NORM% %DP_COMMON% ^
  --dp_noise_multiplier 0.0 --dp_channel_noise_multiplier 0.25 ^
  --output_folder %EXP_ROOT%\channel_ch0.25
if errorlevel 1 goto failed

echo.
echo [3/3] CIFAR K=20, alpha=0.5, channel-only sigma_ch=0.5
python federated_main.py ^
  --dataset %DATASET% --model %MODEL% --epochs %EPOCHS% ^
  --num_users %NUM_USERS% --num_selected 20 ^
  --local_ep %LOCAL_EP% --local_bs %LOCAL_BS% --lr %LR% ^
  --dirichlet_alpha %DIRICHLET_ALPHA% --seed %SEED% --test_size %TEST_SIZE% ^
  --gpu %GPU_ID% ^
  --shapley_update_method mean --shapley_alpha 0.5 --shapley_max_iter 5 ^
  --use_energy --initial_energy 500.0 --energy_threshold 50.0 ^
  --use_lyapunov --lyapunov_V 10.0 --energy_budget 5.0 ^
  %SCHED_WEIGHTS% ^
  --privacy_mode central --dp_clip_norm %DP_CLIP_NORM% %DP_COMMON% ^
  --dp_noise_multiplier 0.0 --dp_channel_noise_multiplier 0.5 ^
  --output_folder %EXP_ROOT%\channel_ch0.5
if errorlevel 1 goto failed

cd /d "%PROJECT_ROOT%"
echo.
echo ========================================
echo Summary
echo ========================================
python src\summarize_dp_results.py --pattern "privacy_utility/%RUN_TAG%/*"
if errorlevel 1 goto failed

echo.
echo Done. Results are under save\%EXP_ROOT%
goto end

:failed
echo.
echo Channel-assisted DP local test failed.
exit /b 1

:end
endlocal
