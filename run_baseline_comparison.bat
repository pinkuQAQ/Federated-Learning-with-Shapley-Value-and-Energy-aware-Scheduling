@echo off
chcp 65001 >nul
set KMP_DUPLICATE_LIB_OK=TRUE
REM ============================================
REM 基线对比实验：5种方法统一对比
REM [1/5] Ours        : SV + Energy + Lyapunov
REM [2/5] FedAvg      : Random Selection
REM [3/5] PoC         : Power of Choice
REM [4/5] UCB         : UCB1 Client Selection
REM [5/5] FedProx     : Proximal Term (mu=0.01)
REM ============================================

echo ========================================
echo Baseline Comparison: 5 Methods
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
echo [1/5] Ours     : SV + Energy + Lyapunov
echo [2/5] FedAvg   : Random Selection
echo [3/5] PoC      : Power of Choice
echo [4/5] UCB      : UCB1 Client Selection
echo [5/5] FedProx  : Proximal Term
echo.
pause

cd /d "%~dp0"
cd src

for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c%%a%%b)
for /f "tokens=1-2 delims=/:" %%a in ('time /t') do (set mytime=%%a%%b)
set OUTPUT_FOLDER=baseline_cmp_%mydate%_%mytime%

REM ============================================
REM [1/5] Ours: SV + Energy + Lyapunov
REM ============================================
echo.
echo [1/5] Running Ours (SV + Energy + Lyapunov)...
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
echo [1/5] Done!
timeout /t 5

REM ============================================
REM [2/5] FedAvg: Random Selection
REM ============================================
echo.
echo [2/5] Running FedAvg (Random)...
python federated_main.py ^
    --dataset %DATASET% --model %MODEL% --epochs %EPOCHS% ^
    --num_users %NUM_USERS% --num_selected %NUM_SELECTED% ^
    --local_ep %LOCAL_EP% --local_bs %LOCAL_BS% --lr %LR% ^
    --dirichlet_alpha %DIRICHLET_ALPHA% --seed %SEED% ^
    --no_shapley ^
    --selection_method random ^
    --output_folder %OUTPUT_FOLDER%
echo [2/5] Done!
timeout /t 5

REM ============================================
REM [3/5] PoC: Power of Choice
REM ============================================
echo.
echo [3/5] Running PoC (Power of Choice)...
python federated_main.py ^
    --dataset %DATASET% --model %MODEL% --epochs %EPOCHS% ^
    --num_users %NUM_USERS% --num_selected %NUM_SELECTED% ^
    --local_ep %LOCAL_EP% --local_bs %LOCAL_BS% --lr %LR% ^
    --dirichlet_alpha %DIRICHLET_ALPHA% --seed %SEED% ^
    --no_shapley ^
    --selection_method poc ^
    --output_folder %OUTPUT_FOLDER%
echo [3/5] Done!
timeout /t 5

REM ============================================
REM [4/5] UCB: UCB1 Client Selection
REM ============================================
echo.
echo [4/5] Running UCB1...
python federated_main.py ^
    --dataset %DATASET% --model %MODEL% --epochs %EPOCHS% ^
    --num_users %NUM_USERS% --num_selected %NUM_SELECTED% ^
    --local_ep %LOCAL_EP% --local_bs %LOCAL_BS% --lr %LR% ^
    --dirichlet_alpha %DIRICHLET_ALPHA% --seed %SEED% ^
    --no_shapley ^
    --selection_method ucb ^
    --ucb_c 1.0 ^
    --output_folder %OUTPUT_FOLDER%
echo [4/5] Done!
timeout /t 5

REM ============================================
REM [5/5] FedProx: Proximal Term
REM ============================================
echo.
echo [5/5] Running FedProx (mu=0.01)...
python federated_main.py ^
    --dataset %DATASET% --model %MODEL% --epochs %EPOCHS% ^
    --num_users %NUM_USERS% --num_selected %NUM_SELECTED% ^
    --local_ep %LOCAL_EP% --local_bs %LOCAL_BS% --lr %LR% ^
    --dirichlet_alpha %DIRICHLET_ALPHA% --seed %SEED% ^
    --no_shapley ^
    --selection_method random ^
    --use_fedprox ^
    --fedprox_mu 0.01 ^
    --output_folder %OUTPUT_FOLDER%
echo [5/5] Done!

echo.
echo ========================================
echo All experiments finished!
echo Results saved to: ../save/%OUTPUT_FOLDER%
echo ========================================
pause
