import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.SysFont('comicsans', 15)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')
INNER_BLUE = (100, 149, 237)
OUTER_BLUE = (65, 105, 225)
INNER_GREEN = (144, 238, 144)
OUTER_GREEN = (60, 179, 113)
WHITE = (255, 255, 255)
GREY = (128, 128, 128)
RED = (139, 0, 0)
BLACK = (0, 0, 0)
CHARCOAL = (255, 87, 51)


BLOCK_SIZE = 20
SPEED = 20

class SnakeGameOne:

    def __init__(self, width=600, height=400):
        self.lr = 0.001 # Learning rate
        self.epsilon = 0  # Randomness
        self.gamma = 0.9  # Discount rate (must be less than 1)
        self.width = width
        self.height = height
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.width / 2, self.height / 2)
        self.snake = [self.head, Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self.place_food()
        self.frame_iteration = 0

    def place_food(self):
        self.food = Point(random.randint(0, (self.width - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE,
                          random.randint(0, (self.height - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE)
        if self.food in self.snake:
            self.place_food()

    def step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.move(action)
        self.snake.insert(0, self.head)
        reward = 0
        game_over = False
        if self.check_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            self.place_food()
        else:
            self.snake.pop()

        self.update()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def check_collision(self, pt=None):
        if pt is None:
            pt = self.head
        return (pt.x > self.width - BLOCK_SIZE or pt.x < 0 or pt.y > self.height - BLOCK_SIZE or pt.y < 0 or
                pt in self.snake[1:])

    def update(self):
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, GREY, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, WHITE, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        # Display learning parameters
        params_text = f"LR: {self.lr:.3f}, ε: {self.epsilon}, γ: {self.gamma:.2f}"
        params_surface = font.render(params_text, True, WHITE)
        self.display.blit(params_surface, [0, 20])
    
        pygame.display.flip()

    def move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # Right turn
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # Left turn
        self.direction = new_dir
        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)

class SnakeGameTwo:

    def __init__(self, width=600, height=400):
        # Game Setup
        self.lr = 0.001 # Learning rate
        self.epsilon = 0  # Randomness
        self.gamma = 0.9  # Discount rate (must be less than 1)
        self.width = width
        self.height = height
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point((self.width / 2) - BLOCK_SIZE*5, self.height / 2)
        self.snake = [self.head, Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
 
        self.direction2 = Direction.LEFT
        self.head2 = Point((self.width / 2) + BLOCK_SIZE*5, self.height / 2)
        self.snake2 = [self.head2, Point(self.head2.x - BLOCK_SIZE, self.head2.y),
                      Point(self.head2.x - (2 * BLOCK_SIZE), self.head2.y)]
        self.score2 = 0

        self.food = None
        self.place_food()
        self.frame_iteration = 0


    def place_food(self):
        self.food = Point(random.randint(0, (self.width - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE,
                          random.randint(0, (self.height - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE)
        if (self.food in self.snake) or (self.food in self.snake2):
            self.place_food()

    def step(self, action, action2):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.move(action, snake_number=1)
        self.move(action2, snake_number=2)
        self.snake.insert(0, self.head)
        self.snake2.insert(0, self.head2)
        reward = 0
        reward2 = 0
        winner = None
        game_over=False

        if self.check_collision() or self.head in self.snake2 or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            winner = "Green"
            return reward, reward2, game_over, self.score, self.score2, winner
        
        if self.check_collision_two() or self.head2 in self.snake or self.frame_iteration > 100 * len(self.snake2):
            game_over = True
            reward2 = -10
            winner = "Blue"
            return reward, reward2, game_over, self.score, self.score2, winner
        
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.place_food()
        else:
            self.snake.pop()

        if self.head2 == self.food:
            self.score2 += 1
            reward2 = 10
            self.place_food()
        else:
            self.snake2.pop()

        self.update()
        self.clock.tick(SPEED)
        return reward, reward2, game_over, self.score, self.score2, winner
    
    def check_collision(self, pt=None):
        if pt is None:
            pt = self.head
        return (pt.x > self.width - BLOCK_SIZE or pt.x < 0 or pt.y > self.height - BLOCK_SIZE or pt.y < 0 or
                pt in self.snake[1:])
    
    def check_collision_two(self, pt2=None):
        if pt2 is None:
            pt2 = self.head2
        return (pt2.x > self.width - BLOCK_SIZE or pt2.x < 0 or pt2.y > self.height - BLOCK_SIZE or pt2.y < 0 or
                pt2 in self.snake2[1:])
    
    def update(self):
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, OUTER_BLUE, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, INNER_BLUE, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
        for pt in self.snake2:
            pygame.draw.rect(self.display, OUTER_GREEN, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, INNER_GREEN, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        text = font.render("Blue: " + str(self.score) + ", Green: " + str(self.score2), True, WHITE)
        self.display.blit(text, [0, 0])
        params_text = f"LR: {self.lr:.3f}, ε: {self.epsilon}, γ: {self.gamma:.2f}"
        params_surface = font.render(params_text, True, WHITE)
        self.display.blit(params_surface, [0, 20])
    
        pygame.display.flip()

    def move(self, action, snake_number):
        if snake_number == 1:
            clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
            idx = clock_wise.index(self.direction)
            if np.array_equal(action, [1, 0, 0]):
                new_dir = clock_wise[idx]  
            elif np.array_equal(action, [0, 1, 0]):
                next_idx = (idx + 1) % 4
                new_dir = clock_wise[next_idx]  # Right turn
            else:
                next_idx = (idx - 1) % 4
                new_dir = clock_wise[next_idx]  # Left turn
            self.direction = new_dir
            x, y = self.head.x, self.head.y
            if self.direction == Direction.RIGHT:
                x += BLOCK_SIZE
            elif self.direction == Direction.LEFT:
                x -= BLOCK_SIZE
            elif self.direction == Direction.DOWN:
                y += BLOCK_SIZE
            elif self.direction == Direction.UP:
                y -= BLOCK_SIZE
            self.head = Point(x, y)
        elif snake_number == 2:
            clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
            idx = clock_wise.index(self.direction2)
            if np.array_equal(action, [1, 0, 0]):
                new_dir = clock_wise[idx]  
            elif np.array_equal(action, [0, 1, 0]):
                next_idx = (idx + 1) % 4
                new_dir = clock_wise[next_idx]  # Right turn
            else:
                next_idx = (idx - 1) % 4
                new_dir = clock_wise[next_idx]  # Left turn
            self.direction2 = new_dir
            x, y = self.head2.x, self.head2.y
            if self.direction2 == Direction.RIGHT:
                x += BLOCK_SIZE
            elif self.direction2 == Direction.LEFT:
                x -= BLOCK_SIZE
            elif self.direction2 == Direction.DOWN:
                y += BLOCK_SIZE
            elif self.direction2 == Direction.UP:
                y -= BLOCK_SIZE
            self.head2 = Point(x, y)
    

