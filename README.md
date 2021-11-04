# RlTrader
RL trader

#명령어
venv\Scripts\activate.bat

python main.py --stock_code 000070 --rl_method pg --net  dnn --learning --num_epoches 100 --lr 0.001 --start_epsilon 1 --discount_factor 0.9