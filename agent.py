import numpy as np
import utils


class Agent:
    # 에이전트 상태의 차원
    STATE_DIM = 2
    # 주식 가치 비율, 포트폴리오 가치 비율

    # 매매수수료 및 세금들
    TRADING_CHARGE = 0.00015  # 거래수수료(0.15%)
    TRADING_TAX = 0.0025  # 거래세(0.25%)

    # 행동
    ACTION_BUY = 0  # 매수
    ACTION_SELL = 1  # 매도
    ACTION_HOLD = 2  # 홀딩(관망)

    # 인공신경망에서 확률을 구할 행동들
    ACTIONS = [ACTION_BUY, ACTION_SELL]
    NUM_ACTIONS = len(ACTIONS)  # 신경망 출력 크기

    def __init__(
        self,
        environment,
        min_trading_unit=1,
        max_trading_unit=2,
        delayed_reward_threshold=0.05,
    ):

        # Environment 객체
        self.environment = environment

        # 최소 매매 단위, 최대 매매 단위, 지연보상 임계치
        self.min_trading_unit = min_trading_unit  # 최소 단일 거래 단위
        self.max_trading_unit = max_trading_unit  # 최대 단일 거래 단위

        # 자연보상 임계치
        self.delayed_reward_threshold = delayed_reward_threshold

        # Agent 클래스의 속성들
        self.initial_balance = 0  # 초기 자본금
        self.balance = 0  # 현재 현금잔고
        self.num_stocks = 0  # 보유 주식수

        self.portfolio_value = 0
        self.base_portfolio_value = 0  # 직전 학습 직전의 PV
        self.num_buy = 0  # 매수 횟수
        self.num_sell = 0  # 매도 횟수
        self.num_hold = 0  # 홀딩 횟수

        self.immediate_reward = 0  # 즉시 보상
        self.profitloss = 0  # 현재 손익
        self.base_profitloss = 0  # 직전 지연 보상 이후 손익
        self.exploration_base = 0  # 탐험 행동 결정 기준

        # Agent 클래스의 상태
        self.ratio_hold = 0  # 주식 보유 비율
        self.ratio_portfolio_value = 0  # 포트폴리오 가치 비율

    def reset(self):
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.base_portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.immediate_reward = 0
        self.ratio_hold = 0
        self.ratio_portfolio_value = 0

    def reset_exploration(self):
        self.exploration_base = 0.5 + np.random.rand() / 2

    def set_balance(self, balance):
        self.initial_balance = balance

    def get_states(self):
        self.ratio_hold = self.num_stocks / int(
            self.portfolio_value / self.environment.get_price()
        )
        self.ratio_portfolio_value = self.portfolio_value / self.base_portfolio_value

        return (self.ratio_hold, self.ratio_portfolio_value)

    def decide_action(self, pred_value, pred_policy, epsilon):
        confidence = 0

        pred = pred_policy
        if pred is None:
            pred = pred_value
            # 예측값이 없다면 탐험
            epsilon = 1
        else:
            # 값이 모두 같은경우 탐험
            maxpred = np.max(pred)
            if (pred == maxpred).all():
                epsilon = 1

        # 탐험 결정
        if np.random.rand() < epsilon:
            exploration = True
            if np.random.rand() < self.exploration_base:
                action = self.ACTION_BUY
            else:
                action = np.random.randint(self.NUM_ACTIONS - 1) + 1
        else:
            exploration = False
            action = np.argmax(pred)

        confidence = 0.5
        if pred_policy is not None:
            confidence = pred[action]
        elif pred_value is not None:
            confidence = utils.sigmoid(pred[action])

        return action, confidence, exploration

    def validate_action(self, action):
        if action == Agent.ACTION_BUY:
            # 적어도 1주를 살수 있는지 확인
            if (
                self.balance
                < self.environment.get_price()
                * (1 + self.TRADING_CHARGE)
                * self.min_trading_unit
            ):
                return False

        elif action == Agent.ACTION_SELL:
            # 주식 잔고가 있는지 확인
            if self.num_stocks <= 0:
                return False

        return True

    def decide_trading_unit(self, confidence):
        if np.isnan(confidence):
            return self.min_trading_unit
        added_trading = max(
            min(
                int(confidence * (self.max_trading_unit - self.min_trading_unit)),
                self.max_trading_unit - self.min_trading_unit,
            ),
            0,
        )
        return self.min_trading_unit + added_trading
    
    def act(self, action, confidence):
        if not self.validation_action(action):
            action = Agent.ACTION_HOLD
        
        #환경에서 현재 가격 얻기
        curr_price = self.environment.get_price()

        #즉시 보상 초기화
        self.immediate_reward = 0

        #매수
        if action == Agent.ACTION_BUY :
            #매수할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)
            balance = (
                self.balance - curr_price * (1 + self.TRADING_CHARGE) * trading_unit
            )

            #보유 현금이 모자란 경우 살수있는만큼 매수
            if balance<0:
                trading_unit = max(
                    min(
                        int(self.balance/(curr_price*(1+self.TRADING_CHARGE))), self.max_trading_unit
                    ),
                    self.min_trading_unit
                )
            #수수료를 적용하여 총 매수 금액 산정
            invest_amount = curr_price * (1+self.TRADING_CHARGE) * trading_unit

            if invest_amount > 0:
                self.balance -= invest_amount #보유현금 갱신
                self.num_stocks += trading_unit #보유주식 수를 갱신
                self.num_buy += 1
            
            #매도
            elif action == Agent.ACTION_SELL:
                #매도할 단위를 판단
                trading_unit = self.decide_trading_unit(confidence)
                #보유 주식이 모자란 경우 최대한 매도
                trading_unit = min(trading_unit, self.num_stocks)
                #매도
                invest_amount = curr_price * (1 - (self.TRADING_CHARGE + self.TRADING_TAX)) * trading_unit

                if invest_amount > 0:
                    self.num_stocks -= trading_unit #보유주식수 갱신
                    self.balance += invest_amount
                    self.num_sell += 1
                
            #홀딩
            elif action == Agent.ACTION_HOLD :
                self.num_hold += 1 #홀딩횟수 count
            
            #포트폴리오 가치 갱신
            self.portfolio_value = self.balance + curr_price * self.num_stocks

            self.profitloss = (
                (self.portfolio_value - self.initial_balance) / self.initial_balance
            )

            #즉시 보상 - 수익률
            self.immediate_reward = self.profitloss

            #지연보상 - 익절, 손절 기준
            delayed_reward = 0
            self.base_profitloss = (
                (self.portfolio_value - self.base_portfolio_value) / self.base_portfolio_value
            )
            if self.base_profitloss > self.delayed_reward_threshold or self.base_profitloss < -self.delayed_reward_threshold :
                #목표 수익을 달성하여 기준 포트폴리오 가치 갱신
                #또는 손실 기준치를 초과하여 기준 포트폴리오 가치 갱신
                self.base_portfolio_value = self.portfolio_value
                delayed_reward = self.immediate_reward
            else :
                delayed_reward = 0
              
            return self.immediate_reward, delayed_reward