@echo off
setlocal enabledelayedexpansion

REM Rerun corrected Oort and normalized FedProx baselines on local Windows.
REM Defaults mirror scripts/slurm/run_oort_fedprox_rerun.sh.
REM Optional overrides before running, for example:
REM   set RUN_ALPHA=0
REM   set EPOCHS=20
REM   set RUN_TAG=local_oort_fedprox_test

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
if not defined ALPHAS set "ALPHAS=0.1 0.25 0.5 1.0"
if not defined FEDPROX_MU set "FEDPROX_MU=0.01"
if not defined RUN_MAIN set "RUN_MAIN=1"
if not defined RUN_ALPHA set "RUN_ALPHA=1"
if not defined RUN_TAG (
  for /f %%T in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "RUN_TAG=%%T"
)

if not defined DP_CLIP_NORM set "DP_CLIP_NORM=1.0"
if not defined CHANNEL_SIGMA set "CHANNEL_SIGMA=0.1"

set "DP_COMMON=--dp_advanced --dp_noise_schedule constant --dp_adaptive_clip --dp_clip_scope layer --dp_clip_percentile 80 --dp_clip_ema 0.8 --dp_clip_growth 1.2 --dp_min_clip_norm 0.05 --dp_max_clip_norm 1.0 --dp_channel_assisted --dp_channel_mode channel_only --dp_channel_gain_cap 2.0"
set "DP_ARGS=--privacy_mode central --dp_clip_norm %DP_CLIP_NORM% %DP_COMMON% --dp_noise_multiplier 0.0 --dp_channel_noise_multiplier %CHANNEL_SIGMA%"
set "BASE_ARGS=--dataset %DATASET% --model %MODEL% --epochs %EPOCHS% --num_users %NUM_USERS% --num_selected %NUM_SELECTED% --local_ep %LOCAL_EP% --local_bs %LOCAL_BS% --lr %LR% --test_size %TEST_SIZE% --gpu %GPU_ID%"
set "MAIN_BASE_ARGS=%BASE_ARGS% --dirichlet_alpha %DIRICHLET_ALPHA%"
set "ENERGY_ARGS=--use_energy --sigma_squared 1.0 --initial_energy 500.0 --energy_threshold 50.0"
set "FEDPROX_ARGS=--no_shapley --selection_method random --use_fedprox --fedprox_mu %FEDPROX_MU%"
set "OORT_ARGS=--no_shapley --selection_method oort"

echo.
echo ========================================
echo Local Oort/FedProx rerun
echo RUN_TAG=%RUN_TAG%
echo SEEDS=%SEEDS%
echo ALPHAS=%ALPHAS%
echo EPOCHS=%EPOCHS%, N=%NUM_USERS%, K=%NUM_SELECTED%, main alpha=%DIRICHLET_ALPHA%
echo RUN_MAIN=%RUN_MAIN%, RUN_ALPHA=%RUN_ALPHA%
echo FedProx: random selection + proximal local objective, mu=%FEDPROX_MU%
echo Oort: corrected Algorithm-1-style selector with energy availability
echo Main output root: save\sv_supp\%RUN_TAG%\main
echo Alpha output root: save\sensitivity_multiseed\%RUN_TAG%\alpha
echo ========================================

cd /d "%PROJECT_ROOT%\src"

if "%RUN_MAIN%"=="1" (
  echo.
  echo ========================================
  echo [Group 1] Main comparison rerun: Oort + FedProx
  echo ========================================
  for %%S in (%SEEDS%) do (
    set "OUT=sv_supp\%RUN_TAG%\main\seed%%S"

    echo.
    echo ----------------------------------------
    echo FedProx main alpha=%DIRICHLET_ALPHA%, seed=%%S
    echo Output folder: !OUT!
    echo Start: %DATE% %TIME%
    echo ----------------------------------------
    python federated_main.py ^
      %MAIN_BASE_ARGS% --seed %%S ^
      %FEDPROX_ARGS% ^
      %DP_ARGS% ^
      --output_folder "!OUT!"
    if errorlevel 1 goto failed

    echo.
    echo ----------------------------------------
    echo Oort main alpha=%DIRICHLET_ALPHA%, seed=%%S
    echo Output folder: !OUT!
    echo Start: %DATE% %TIME%
    echo ----------------------------------------
    python federated_main.py ^
      %MAIN_BASE_ARGS% --seed %%S ^
      %OORT_ARGS% ^
      %ENERGY_ARGS% %DP_ARGS% ^
      --output_folder "!OUT!"
    if errorlevel 1 goto failed
  )
)

if "%RUN_ALPHA%"=="1" (
  echo.
  echo ========================================
  echo [Group 2] Dirichlet-alpha sensitivity rerun: Oort + FedProx
  echo ========================================
  for %%A in (%ALPHAS%) do (
    for %%S in (%SEEDS%) do (
      set "OUT_BASE=sensitivity_multiseed\%RUN_TAG%\alpha\alpha%%A\seed%%S"

      echo.
      echo ----------------------------------------
      echo FedProx alpha=%%A, seed=%%S
      echo Output folder: !OUT_BASE!\fedprox
      echo Start: %DATE% %TIME%
      echo ----------------------------------------
      python federated_main.py ^
        %BASE_ARGS% --dirichlet_alpha %%A --seed %%S ^
        %FEDPROX_ARGS% ^
        %DP_ARGS% ^
        --output_folder "!OUT_BASE!\fedprox"
      if errorlevel 1 goto failed

      echo.
      echo ----------------------------------------
      echo Oort alpha=%%A, seed=%%S
      echo Output folder: !OUT_BASE!\oort
      echo Start: %DATE% %TIME%
      echo ----------------------------------------
      python federated_main.py ^
        %BASE_ARGS% --dirichlet_alpha %%A --seed %%S ^
        %OORT_ARGS% ^
        %ENERGY_ARGS% %DP_ARGS% ^
        --output_folder "!OUT_BASE!\oort"
      if errorlevel 1 goto failed
    )
  )
)

cd /d "%PROJECT_ROOT%"
echo.
echo ========================================
echo Local Oort/FedProx rerun finished!
echo Main results root: save\sv_supp\%RUN_TAG%\main
echo Alpha results root: save\sensitivity_multiseed\%RUN_TAG%\alpha
echo Aggregate main with:
echo   python src\summarize_sv_supp_results.py --tag %RUN_TAG%
echo Aggregate alpha with:
echo   python src\summarize_sensitivity_multiseed.py --tag %RUN_TAG%
echo ========================================
goto end

:failed
cd /d "%PROJECT_ROOT%"
echo.
echo Local Oort/FedProx rerun failed. Check the latest output above.
exit /b 1

:end
endlocal
