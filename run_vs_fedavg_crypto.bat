@echo off
chcp 65001 >nul
set KMP_DUPLICATE_LIB_OK=TRUE
REM ============================================
REM FedAvg vs Ours (SV + Energy + Lyapunov + AES-256-GCM)
REM ============================================

echo ========================================
echo FedAvg vs Ours (SV + Energy + Lyapunov + AES-256-GCM Encryption)
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
echo [1/2] Ours    : SV + Energy + Lyapunov + AES-256-GCM
echo [2/2] Ours (No Crypto): SV + Energy + Lyapunov (Ablation Study)
echo.
pause

cd /d "%~dp0"
cd src

for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c%%a%%b)
for /f "tokens=1-2 delims=/:" %%a in ('time /t') do (set mytime=%%a%%b)
set OUTPUT_FOLDER=crypto_%mydate%_%mytime%

REM ============================================
REM [1/2] Ours with AES-256-GCM Encryption
REM ============================================
echo.
echo [1/2] Running Ours (SV + Energy + Lyapunov + AES-256-GCM)...
python federated_main.py ^
    --dataset %DATASET% --model %MODEL% --epochs %EPOCHS% ^
    --num_users %NUM_USERS% --num_selected %NUM_SELECTED% ^
    --local_ep %LOCAL_EP% --local_bs %LOCAL_BS% --lr %LR% ^
    --dirichlet_alpha %DIRICHLET_ALPHA% --seed %SEED% ^
    --shapley_update_method mean ^
    --shapley_alpha 0.5 ^
    --shapley_epsilon 0.01 ^
    --use_energy ^
    --initial_energy 500.0 ^
    --energy_threshold 50.0 ^
    --use_lyapunov ^
    --lyapunov_V 10.0 ^
    --energy_budget 5.0 ^
    --use_crypto ^
    --output_folder %OUTPUT_FOLDER%
echo [1/2] Ours with AES-256-GCM completed!
timeout /t 5

REM ============================================
REM [2/2] FedAvg (No Encryption, Random)
REM ============================================
echo.
echo [2/2] Running FedAvg (Random Selection, No Encryption)...
python federated_main.py ^
    --dataset %DATASET% --model %MODEL% --epochs %EPOCHS% ^
    --num_users %NUM_USERS% --num_selected %NUM_SELECTED% ^
    --local_ep %LOCAL_EP% --local_bs %LOCAL_BS% --lr %LR% ^
    --dirichlet_alpha %DIRICHLET_ALPHA% --seed %SEED% ^
    --no_shapley ^
    --selection_method random ^
    --output_folder %OUTPUT_FOLDER%
echo [2/2] FedAvg completed!

echo.
echo ========================================
echo Experiment finished!
echo Results saved to: ../save/%OUTPUT_FOLDER%
echo ========================================
pause
