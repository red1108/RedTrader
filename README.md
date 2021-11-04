# RlTrader
RL trader

# Quick Setup (Tensorflow)
    venv\Scripts\activate.bat
    python main.py --stock_code 000070 --rl_method pg --net  dnn --learning --num_epoches 100 --lr 0.001 --start_epsilon 1 --discount_factor 0.9


# Quick Setup (Plaidml)
    venv\Scripts\activate.bat
    python main.py --backend plaidml --stock_code 000070 --rl_method pg --net  dnn --learning --num_epoches 100 --lr 0.001 --start_epsilon 1 --discount_factor 0.9
