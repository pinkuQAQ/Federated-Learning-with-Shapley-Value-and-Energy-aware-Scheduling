@echo off
setlocal enabledelayedexpansion

REM Full local Windows run for MC-Shapley iteration sensitivity.
REM Runs Ours with M in {5, 10, 20, 50}.

set "PROJECT_ROOT=%~dp0..\.."
cd /d "%PROJECT_ROOT%"

set CONDA_ENV=

if not "%CONDA_ENV%"=="" (
  call conda activate %CONDA_ENV%
  if errorlevel 1 (
    echo Failed to activate conda env "%CONDA_ENV%".
    exit /b 1
  )
)

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
set NUM_SELECTED=5
set LOCAL_EP=2
set LOCAL_BS=32
set LR=0.01
set ALPHA=0.1
set TEST_SIZE=10000
set SEED=42
set GPU_ID=0
set DP_CLIP_NORM=1.0
set CHANNEL_SIGMA=0.1
set SCHED_WEIGHTS=--sv_weight 0.7 --battery_weight 0.15 --channel_weight 0.15
set DP_COMMON=--dp_advanced --dp_noise_schedule constant --dp_adaptive_clip --dp_clip_scope layer --dp_clip_percentile 80 --dp_clip_ema 0.8 --dp_clip_growth 1.2 --dp_min_clip_norm 0.05 --dp_max_clip_norm 1.0 --dp_channel_assisted --dp_channel_mode channel_only --dp_channel_gain_cap 2.0
set DP_ARGS=--privacy_mode central --dp_clip_norm %DP_CLIP_NORM% %DP_COMMON% --dp_noise_multiplier 0.0 --dp_channel_noise_multiplier %CHANNEL_SIGMA%

set RUN_TAG=%DATE:~0,4%%DATE:~5,2%%DATE:~8,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%
set RUN_TAG=%RUN_TAG: =0%
set EXP_ROOT=sensitivity_M\%RUN_TAG%

echo ========================================
echo CIFAR-10 MC-Shapley M sensitivity local run
echo M values: 5, 10, 20, 50
echo Seed: %SEED%, alpha: %ALPHA%, epochs: %EPOCHS%, K: %NUM_SELECTED%
echo Channel-only privacy sigma_ch: %CHANNEL_SIGMA%
echo Output root: save\%EXP_ROOT%
echo ========================================

cd /d "%PROJECT_ROOT%\src"

for %%M in (5 10 20 50) do (
  set OUT=%EXP_ROOT%\M%%M

  echo.
  echo ========================================
  echo MC-Shapley M %%M
  echo Output folder: !OUT!
  echo ========================================

  python federated_main.py ^
    --dataset %DATASET% --model %MODEL% --epochs %EPOCHS% ^
    --num_users %NUM_USERS% --num_selected %NUM_SELECTED% ^
    --local_ep %LOCAL_EP% --local_bs %LOCAL_BS% --lr %LR% ^
    --dirichlet_alpha %ALPHA% --seed %SEED% --test_size %TEST_SIZE% ^
    --gpu %GPU_ID% ^
    --shapley_update_method mean --shapley_alpha 0.5 --shapley_max_iter %%M ^
    --use_energy --sigma_squared 1.0 --initial_energy 500.0 --energy_threshold 50.0 ^
    --use_lyapunov --lyapunov_V 10.0 --energy_budget 5.0 ^
    %SCHED_WEIGHTS% ^
    %DP_ARGS% ^
    --output_folder !OUT!
  if errorlevel 1 goto failed
)

cd /d "%PROJECT_ROOT%"
echo.
echo ========================================
echo MC-Shapley M sensitivity local run finished.
echo Results are under save\%EXP_ROOT%
echo ========================================
goto end

:failed
echo.
echo MC-Shapley M sensitivity local run failed. Check the latest output above.
exit /b 1

:end
endlocal
