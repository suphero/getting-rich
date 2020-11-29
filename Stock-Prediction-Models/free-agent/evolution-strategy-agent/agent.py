import numpy as np
import matplotlib.pyplot as plt
from deepEvolutionStrategy import DeepEvolutionStrategy
import helper as hp


class Agent:
    POPULATION_SIZE = 15
    SIGMA = 0.1
    LEARNING_RATE = 0.03

    def __init__(self, model, money, close, window_size, skip):
        self.model = model
        self.initial_money = money
        self.close = close
        self.window_size = window_size
        self.skip = skip
        self.data_len = len(close) - 1
        self.es = DeepEvolutionStrategy(
            self.model.get_weights(),
            self.get_reward,
            self.POPULATION_SIZE,
            self.SIGMA,
            self.LEARNING_RATE,
        )

    def act(self, sequence):
        decision, buy = self.model.predict(np.array(sequence))
        return np.argmax(decision[0]), buy[0]

    def get_reward(self, weights):
        starting_money = self.initial_money
        current_money = self.initial_money
        self.model.weights = weights
        state = hp.get_state(self.close, 0, self.window_size + 1)
        inventory = []
        quantity = 0
        for t in range(0, self.data_len, self.skip):
            action, buy = self.act(state)
            next_state = hp.get_state(self.close, t + 1, self.window_size + 1)
            max_buy = current_money / self.close[t]
            if action == 1:
                if buy < 0:
                    buy = 1
                if buy > max_buy:
                    buy_units = max_buy
                else:
                    buy_units = buy
                total_buy = buy_units * self.close[t]
                current_money -= total_buy
                inventory.append(total_buy)
                quantity += buy_units
            elif action == 2 and len(inventory) > 0:
                sell_units = quantity
                quantity -= sell_units
                total_sell = sell_units * self.close[t]
                current_money += total_sell

            state = next_state
        return ((current_money - starting_money) / starting_money) * 100

    def fit(self, iterations, checkpoint):
        self.es.train(iterations, print_every=checkpoint)

    def simulate(self):
        starting_money = self.initial_money
        current_money = self.initial_money
        state = hp.get_state(self.close, 0, self.window_size + 1)
        states_sell = []
        states_buy = []
        inventory = []
        quantity = 0
        for t in range(0, self.data_len, self.skip):
            action, buy = self.act(state)
            next_state = hp.get_state(self.close, t + 1, self.window_size + 1)
            max_buy = current_money / self.close[t]
            if action == 1:
                if buy < 0:
                    buy = 1
                if buy > max_buy:
                    buy_units = max_buy
                else:
                    buy_units = buy

                total_buy = buy_units * self.close[t]
                current_money -= total_buy
                inventory.append(total_buy)
                quantity += buy_units
                states_buy.append(t)
                print(
                    'day %d: buy %d units at price %f, total balance %f, quantity %f'
                    % (t, buy_units, total_buy, current_money, quantity)
                )
            elif action == 2 and len(inventory) > 0:
                bought_price = inventory.pop(0)
                sell_units = quantity
                if sell_units < 0:
                    continue
                quantity -= sell_units
                total_sell = sell_units * self.close[t]
                current_money += total_sell
                states_sell.append(t)
                try:
                    invest = ((total_sell - bought_price) / bought_price) * 100
                except:
                    invest = 0
                print(
                    'day %d, sell %d units at price %f, investment %f %%, total balance %f, quantity %f'
                    % (t, sell_units, total_sell, invest, current_money, quantity)
                )
            state = next_state

        invest = ((current_money - starting_money) / starting_money) * 100
        print(
            '\ntotal gained %f, total investment %f %%'
            % (current_money - starting_money, invest)
        )
        plt.figure(figsize=(20, 10))
        plt.plot(self.close, label='true close', c='g')
        plt.plot(
            self.close, 'X', label='predict buy', markevery=states_buy, c='b'
        )
        plt.plot(
            self.close, 'o', label='predict sell', markevery=states_sell, c='r'
        )
        plt.legend()
        plt.show()
