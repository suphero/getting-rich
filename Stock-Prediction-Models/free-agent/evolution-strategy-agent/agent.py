import numpy as np
import matplotlib.pyplot as plt
from deepEvolutionStrategy import DeepEvolutionStrategy
import helper as hp


class Agent:
    POPULATION_SIZE = 15
    SIGMA = 0.1
    LEARNING_RATE = 0.03

    def __init__(self, model, money, train_close, test_close, window_size, skip, commission_rate):
        self.model = model
        self.initial_money = money
        self.train_close = train_close
        self.test_close = test_close
        self.window_size = window_size
        self.skip = skip
        self.commission_rate = commission_rate
        self.train_len = len(train_close) - 1
        self.test_len = len(test_close) - 1
        self.states_sell = []
        self.states_buy = []
        self.asset_values = []
        self.es = DeepEvolutionStrategy(
            self.model.get_weights(),
            self.get_reward,
            self.POPULATION_SIZE,
            self.SIGMA,
            self.LEARNING_RATE,
        )

    def act(self, sequence):
        decision, buy = self.model.predict(np.array(sequence))
        return np.argmax(decision[0]), int(buy[0])

    def get_reward(self, weights):
        starting_money = self.initial_money
        current_money = self.initial_money
        self.model.weights = weights
        state = hp.get_state(self.train_close, 0, self.window_size + 1)
        inventory = []
        quantity = 0
        for t in range(0, self.train_len, self.skip):
            action, buy = self.act(state)
            next_state = hp.get_state(self.train_close, t + 1, self.window_size + 1)
            iter_close = self.train_close[t]
            if action == 1:
                ask_price = iter_close * (1 + self.commission_rate)
                max_buy = current_money / ask_price
                if buy < 0:
                    buy = 1
                if buy > max_buy:
                    buy_units = max_buy
                else:
                    buy_units = buy
                total_buy = buy_units * ask_price
                current_money -= total_buy
                inventory.append(total_buy)
                quantity += buy_units
            elif action == 2 and len(inventory) > 0:
                bid_price = iter_close * (1 - self.commission_rate)
                sell_units = quantity
                quantity -= sell_units
                total_sell = sell_units * bid_price
                current_money += total_sell

            state = next_state
        return ((current_money - starting_money) / starting_money) * 100

    def fit(self, iterations, checkpoint):
        self.es.train(iterations, print_every = checkpoint)

    def simulate(self):
        starting_money = self.initial_money
        current_money = self.initial_money
        state = hp.get_state(self.test_close, 0, self.window_size + 1)
        inventory = []
        quantity = 0
        for t in range(0, self.test_len, self.skip):
            action, buy = self.act(state)
            next_state = hp.get_state(self.test_close, t + 1, self.window_size + 1)
            iter_close = self.test_close[t]
            if action == 1:
                ask_price = iter_close * (1 + self.commission_rate)
                max_buy = current_money / ask_price 
                if buy < 0:
                    buy = 1
                if buy > max_buy:
                    buy_units = max_buy
                else:
                    buy_units = buy

                total_buy = buy_units * ask_price
                current_money -= total_buy
                inventory.append(total_buy)
                quantity += buy_units
                self.states_buy.append(t)
                print(
                    'day %d: buy %d units at price %f, total balance %f, quantity %f'
                    % (t, buy_units, total_buy, current_money, quantity)
                )
            elif action == 2 and len(inventory) > 0:
                bid_price = iter_close * (1 - self.commission_rate)
                bought_price = inventory.pop(0)
                sell_units = quantity
                if sell_units < 0:
                    continue
                quantity -= sell_units
                total_sell = sell_units * bid_price
                current_money += total_sell
                self.states_sell.append(t)
                try:
                    invest = ((total_sell - bought_price) / bought_price) * 100
                except:
                    invest = 0
                print(
                    'day %d, sell %d units at price %f, investment %f %%, total balance %f, quantity %f'
                    % (t, sell_units, total_sell, invest, current_money, quantity)
                )
            state = next_state
            asset_value = quantity * iter_close + current_money
            self.asset_values.append(asset_value)

        invest = ((current_money - starting_money) / starting_money) * 100
        print(
            '\ntotal gained %f, total investment %f %%'
            % (current_money - starting_money, invest)
        )

    def print_history(self):
        plt.figure(figsize = (20, 10))
        plt.plot(self.test_close, label = 'true close', c = 'g')
        plt.plot(
            self.test_close, 'X', label = 'predict buy', markevery = self.states_buy, c = 'b'
        )
        plt.plot(
            self.test_close, 'o', label = 'predict sell', markevery = self.states_sell, c = 'r'
        )

        plt.legend()
        plt.show()

    def print_asset_value(self):
        plt.figure(figsize = (20, 10))
        plt.plot(self.asset_values, label = 'Asset Value')

        plt.legend(loc='upper left')
        plt.show()