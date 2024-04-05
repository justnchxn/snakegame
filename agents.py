import torch
import random
import pygame
import time
import numpy as np
from collections import deque
from game import SnakeGameOne, SnakeGameTwo, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot, plot_two

MAX_MEMORY = 100_000
BATCH_SIZE = 1000

class Agent:
    def __init__(self):
        self.num_games = 0
        self.epsilon = SnakeGameOne().epsilon
        self.gamma = SnakeGameOne().gamma
        self.lr = SnakeGameOne().lr
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, learning_rate=self.lr, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        danger_left = game.direction == Direction.LEFT
        danger_right = game.direction == Direction.RIGHT
        danger_up = game.direction == Direction.UP
        danger_down = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (danger_right and game.check_collision(Point(head.x + 20, head.y))) or 
            (danger_left and game.check_collision(Point(head.x - 20, head.y))) or 
            (danger_up and game.check_collision(Point(head.x, head.y - 20))) or 
            (danger_down and game.check_collision(Point(head.x, head.y + 20))),

            # Danger right
            (danger_up and game.check_collision(Point(head.x + 20, head.y))) or 
            (danger_down and game.check_collision(Point(head.x - 20, head.y))) or 
            (danger_left and game.check_collision(Point(head.x, head.y - 20))) or 
            (danger_right and game.check_collision(Point(head.x, head.y + 20))),

            # Danger left
            (danger_down and game.check_collision(Point(head.x + 20, head.y))) or 
            (danger_up and game.check_collision(Point(head.x - 20, head.y))) or 
            (danger_right and game.check_collision(Point(head.x, head.y - 20))) or 
            (danger_left and game.check_collision(Point(head.x, head.y + 20))),
            
            # Move direction
            danger_left,
            danger_right,
            danger_up,
            danger_down,
            
            # Food location 
            game.food.x < head.x,  # Food left
            game.food.x > head.x,  # Food right
            game.food.y < head.y,  # Food up
            game.food.y > head.y  # Food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        
        for state, action, reward, next_state, done in mini_sample:
            self.trainer.train_step(state, action, reward, next_state, done)
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Random move = tradeoff of exploration/exploitation
        self.epsilon = 80 - self.num_games
        action = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            action[move] = 1
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()
            action[move] = 1
        return action

class AgentTwo:
    def __init__(self):
        self.num_games = 0
        self.epsilon = SnakeGameTwo().epsilon
        self.gamma = SnakeGameTwo().gamma
        self.lr = SnakeGameTwo().lr
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, learning_rate=self.lr, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake2[0]
        danger_left = game.direction == Direction.LEFT
        danger_right = game.direction == Direction.RIGHT
        danger_up = game.direction == Direction.UP
        danger_down = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (danger_right and game.check_collision_two(Point(head.x + 20, head.y))) or 
            (danger_left and game.check_collision_two(Point(head.x - 20, head.y))) or 
            (danger_up and game.check_collision_two(Point(head.x, head.y - 20))) or 
            (danger_down and game.check_collision_two(Point(head.x, head.y + 20))),

            # Danger right
            (danger_up and game.check_collision_two(Point(head.x + 20, head.y))) or 
            (danger_down and game.check_collision_two(Point(head.x - 20, head.y))) or 
            (danger_left and game.check_collision_two(Point(head.x, head.y - 20))) or 
            (danger_right and game.check_collision_two(Point(head.x, head.y + 20))),


            # Danger left
            (danger_down and game.check_collision_two(Point(head.x + 20, head.y))) or 
            (danger_up and game.check_collision_two(Point(head.x - 20, head.y))) or 
            (danger_right and game.check_collision_two(Point(head.x, head.y - 20))) or 
            (danger_left and game.check_collision_two(Point(head.x, head.y + 20))),
            
            # Move direction
            danger_left,
            danger_right,
            danger_up,
            danger_down,
            
            # Food location 
            game.food.x < head.x,  # Food left
            game.food.x > head.x,  # Food right
            game.food.y < head.y,  # Food up
            game.food.y > head.y  # Food down
        ]

        return np.array(state, dtype=int)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        
        for state, action, reward, next_state, done in mini_sample:
            self.trainer.train_step(state, action, reward, next_state, done)
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Random move = tradeoff of exploration/exploitation
        self.epsilon = 80 - self.num_games
        action = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            action[move] = 1
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()
            action[move] = 1
        return action


def train_one():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameOne()

    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_1]:
            game.lr = min(1.0, game.lr + 0.01)
            time.sleep(0.1)
        if keys[pygame.K_2]:
            game.lr = max(0.01, game.lr - 0.01)
            time.sleep(0.1)
        if keys[pygame.K_3]:
            game.epsilon = min(100, game.epsilon + 1)
            time.sleep(0.1)
        if keys[pygame.K_4]:
            game.epsilon = max(0, game.epsilon - 1)
            time.sleep(0.1)
        if keys[pygame.K_5]:
            game.gamma = min(1.0, game.gamma + 0.1)
            time.sleep(0.1)
        if keys[pygame.K_6]:
            game.gamma = max(0.1, game.gamma - 0.1)
            time.sleep(0.1)

        state_old = agent.get_state(game)

        # Get action
        action = agent.get_action(state_old)

        # Perform action and get new state
        reward, done, score = game.step(action)
        state_new = agent.get_state(game)

        # Train short memory
        agent.train_short_memory(state_old, action, reward, state_new, done)

        # Remember
        agent.remember(state_old, action, reward, state_new, done)

        if done:
            # Train long memory, plot result
            game.reset()
            agent.num_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.num_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.num_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

def train_two():
    plot_score_one = []
    plot_mean_score_one = []
    total_score_one = 0
    record = 0
    plot_score_two = []
    plot_mean_score_two = []
    total_score_two = 0
    record2 = 0
    winner = None
    agent = Agent()
    agent2 = AgentTwo()
    game = SnakeGameTwo()

    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_1]:
            game.lr = min(1.0, game.lr + 0.01)
            time.sleep(0.1)
        if keys[pygame.K_2]:
            game.lr = max(0.01, game.lr - 0.01)
            time.sleep(0.1)
        if keys[pygame.K_3]:
            game.epsilon = min(100, game.epsilon + 1)
            time.sleep(0.1)
        if keys[pygame.K_4]:
            game.epsilon = max(0, game.epsilon - 1)
            time.sleep(0.1)
        if keys[pygame.K_5]:
            game.gamma = min(1.0, game.gamma + 0.1)
            time.sleep(0.1)
        if keys[pygame.K_6]:
            game.gamma = max(0.1, game.gamma - 0.1)
            time.sleep(0.1)

        state_old = agent.get_state(game)
        state_old2 = agent2.get_state(game)
        action = agent.get_action(state_old)
        action2 = agent2.get_action(state_old2)
        reward, reward2, done, score, score2, winner = game.step(action, action2)
        state_new = agent.get_state(game)
        state_new2 = agent2.get_state(game)

        # Train short memory
        agent.train_short_memory(state_old, action, reward, state_new, done)
        agent2.train_short_memory(state_old2, action2, reward2, state_new2, done)

        # Remember
        agent.remember(state_old, action, reward, state_new, done)
        agent2.remember(state_old2, action2, reward2, state_new2, done)

        if done:
            # Train long memory, plot result
            game.reset()
            agent.num_games += 1
            agent.train_long_memory()
            agent2.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
            
            if score2 > record2:
                record2 = score2
                agent2.model.save()

            print('Game', agent.num_games, 'Blue Score/Record:', str(score) + '/' + str(record), 'Green Score/Record:', str(score2) + '/' + str(record2), 'Winner:', winner)

            plot_score_one.append(score)
            plot_score_two.append(score2)
            total_score_one += score
            total_score_two += score2
            mean_score_one = total_score_one / agent.num_games
            mean_score_two = total_score_one / agent.num_games
            plot_mean_score_one.append(mean_score_one)
            plot_mean_score_two.append(mean_score_two)
            plot_two(plot_score_one, plot_score_two, plot_mean_score_one, plot_mean_score_two)

if __name__ == '__main__':
    snake_count = int(input("Choose the number of snakes (1 or 2): "))
    if snake_count == 1:
        train_one()
    elif snake_count == 2:
        train_two()
    else:
        print("Please choose a number between 1 and 2")
