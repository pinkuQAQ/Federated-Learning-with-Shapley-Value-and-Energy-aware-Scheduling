@echo off
chcp 65001 >nul
set KMP_DUPLICATE_LIB_OK=TRUE
REM ============================================
REM 信道鲁棒性实验：不同噪声方差 σ²=0.5/1.0/3.0
REM Ours (SV + Energy + Lyapunov) vs FedAvg (Random)
REM ============================================

echo ========================================
echo Channel Robustness: sigma^2 = 0.5 / 1.0 / 3.0
echo ========================================

REM === Shared Parameters ===
set DATASET=cifar
set MODEL=cnn
set EPOCHS=100
set NUM_USERS=100
set NUM_SELECTED=10
set LOCAL_EP=2
set LOCAL_BS=32
set LR=0.01
set DIRICHLET_ALPHA=0.1
set SEED=42

echo.
echo Base config: %DATASET% / %MODEL% / %EPOCHS% rounds
echo Clients: %NUM_USERS% total, %NUM_SELECTED% per round
echo Non-IID: Dirichlet alpha=%DIRICHLET_ALPHA%  Seed=%SEED%
echo.
echo [1/6] Ours    : sigma^2=0.5
echo [2/6] FedAvg  : sigma^2=0.5
echo [3/6] Ours    : sigma^2=1.0
echo [4/6] FedAvg  : sigma^2=1.0
echo [5/6] Ours    : sigma^2=3.0
echo [6/6] FedAvg  : sigma^2=3.0
echo.
pause

cd /d "%~dp0"
cd src

for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c%%a%%b)
for /f "tokens=1-2 delims=/:" %%a in ('time /t') do (set mytime=%%a%%b)
set OUTPUT_FOLDER=channel_robustness_%mydate%_%mytime%

REM ============================================
REM [1/6] Ours: sigma^2 = 0.5
REM ============================================
echo.
echo [1/6] Running Ours (sigma^2=0.5)...
python federated_main.py ^
    --dataset %DATASET% --model %MODEL% --epochs %EPOCHS% ^
    --num_users %NUM_USERS% --num_selected %NUM_SELECTED% ^
    --local_ep %LOCAL_EP% --local_bs %LOCAL_BS% --lr %LR% ^
    --dirichlet_alpha %DIRICHLET_ALPHA% --seed %SEED% ^
    --shapley_update_method mean ^
    --shapley_alpha 0.5 ^
    --use_energy ^
    --sigma_squared 0.5 ^
    --initial_energy 500.0 ^
    --energy_threshold 50.0 ^
    --use_lyapunov ^
    --lyapunov_V 10.0 ^
    --energy_budget 5.0 ^
    --output_folder %OUTPUT_FOLDER%
echo [1/6] Done!
timeout /t 5

REM ============================================
REM [2/6] FedAvg: sigma^2 = 0.5
REM ============================================
echo.
echo [2/6] Running FedAvg (sigma^2=0.5)...
python federated_main.py ^
    --dataset %DATASET% --model %MODEL% --epochs %EPOCHS% ^
    --num_users %NUM_USERS% --num_selected %NUM_SELECTED% ^
    --local_ep %LOCAL_EP% --local_bs %LOCAL_BS% --lr %LR% ^
    --dirichlet_alpha %DIRICHLET_ALPHA% --seed %SEED% ^
    --no_shapley ^
    --selection_method random ^
    --sigma_squared 0.5 ^
    --output_folder %OUTPUT_FOLDER%
echo [2/6] Done!
timeout /t 5

REM ============================================
REM [3/6] Ours: sigma^2 = 1.0
REM ============================================
echo.
echo [3/6] Running Ours (sigma^2=1.0)...
python federated_main.py ^
    --dataset %DATASET% --model %MODEL% --epochs %EPOCHS% ^
    --num_users %NUM_USERS% --num_selected %NUM_SELECTED% ^
    --local_ep %LOCAL_EP% --local_bs %LOCAL_BS% --lr %LR% ^
    --dirichlet_alpha %DIRICHLET_ALPHA% --seed %SEED% ^
    --shapley_update_method mean ^
    --shapley_alpha 0.5 ^
    --use_energy ^
    --sigma_squared 1.0 ^
    --initial_energy 500.0 ^
    --energy_threshold 50.0 ^
    --use_lyapunov ^
    --lyapunov_V 10.0 ^
    --energy_budget 5.0 ^
    --output_folder %OUTPUT_FOLDER%
echo [3/6] Done!
timeout /t 5

REM ============================================
REM [4/6] FedAvg: sigma^2 = 1.0
REM ============================================
echo.
echo [4/6] Running FedAvg (sigma^2=1.0)...
python federated_main.py ^
    --dataset %DATASET% --model %MODEL% --epochs %EPOCHS% ^
    --num_users %NUM_USERS% --num_selected %NUM_SELECTED% ^
    --local_ep %LOCAL_EP% --local_bs %LOCAL_BS% --lr %LR% ^
    --dirichlet_alpha %DIRICHLET_ALPHA% --seed %SEED% ^
    --no_shapley ^
    --selection_method random ^
    --sigma_squared 1.0 ^
    --output_folder %OUTPUT_FOLDER%
echo [4/6] Done!
timeout /t 5

REM ============================================
REM [5/6] Ours: sigma^2 = 3.0
REM ============================================
echo.
echo [5/6] Running Ours (sigma^2=3.0)...
python federated_main.py ^
    --dataset %DATASET% --model %MODEL% --epochs %EPOCHS% ^
    --num_users %NUM_USERS% --num_selected %NUM_SELECTED% ^
    --local_ep %LOCAL_EP% --local_bs %LOCAL_BS% --lr %LR% ^
    --dirichlet_alpha %DIRICHLET_ALPHA% --seed %SEED% ^
    --shapley_update_method mean ^
    --shapley_alpha 0.5 ^
    --use_energy ^
    --sigma_squared 3.0 ^
    --initial_energy 500.0 ^
    --energy_threshold 50.0 ^
    --use_lyapunov ^
    --lyapunov_V 10.0 ^
    --energy_budget 5.0 ^
    --output_folder %OUTPUT_FOLDER%
echo [5/6] Done!
timeout /t 5

REM ============================================
REM [6/6] FedAvg: sigma^2 = 3.0
REM ============================================
echo.
echo [6/6] Running FedAvg (sigma^2=3.0)...
python federated_main.py ^
    --dataset %DATASET% --model %MODEL% --epochs %EPOCHS% ^
    --num_users %NUM_USERS% --num_selected %NUM_SELECTED% ^
    --local_ep %LOCAL_EP% --local_bs %LOCAL_BS% --lr %LR% ^
    --dirichlet_alpha %DIRICHLET_ALPHA% --seed %SEED% ^
    --no_shapley ^
    --selection_method random ^
    --sigma_squared 3.0 ^
    --output_folder %OUTPUT_FOLDER%
echo [6/6] Done!

echo.
echo ========================================
echo Experiment finished!
echo Results saved to: ../save/%OUTPUT_FOLDER%
echo ========================================
pause
