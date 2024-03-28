import pygame
import random
from enum import Enum
from collections import namedtuple, deque
import numpy as np

# Initialise pygame
pygame.init()

BLOCK_SIZE = 20
TICK_SPEED = 20
FONT = pygame.font.SysFont('comicsans', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x y')

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLACK = (0, 0, 0)

class SnakeGame:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()
        self.reset()
    
    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.width // 2, self.height // 2)
        self.snake = deque([self.head, Point(self.head.x - BLOCK_SIZE, self.head.y), Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)])
        self.score = 0
        self.food = None
        self.place_food()
        self.frame_iteration = 0
    
    def place_food(self):
        self.food = Point(random.randint(0, (self.width - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE,
                          random.randint(0, (self.height - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE)
        while self.food in self.snake:
            self.food = Point(random.randint(0, (self.width - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE,
                              random.randint(0, (self.height - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE)

    def step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.move(action) 
        self.snake.appendleft(self.head)

        reward = 0
        game_over = False
        if self.check_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return game_over, self.score, reward

        if self.head == self.food:
            self.score += 1
            reward = 10
            self.place_food()
        else:
            self.snake.pop()

        self.update()
        self.clock.tick(TICK_SPEED)
        return game_over, self.score, reward
    
    def check_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if not 0 <= pt.x < self.width or not 0 <= pt.y < self.height or pt in self.snake:
            return True
        return False
    
    def update(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, WHITE, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, WHITE, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        text = FONT.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [10, 10])
        pygame.display.flip()

    def move(self, action):
        direction_clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        direction_index = direction_clockwise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_direction = direction_clockwise[direction_index]  # Forward
        elif np.array_equal(action, [0, 1, 0]):
            next_direction_index = (direction_index + 1) % 4 
            new_direction = direction_clockwise[next_direction_index]  # Right turn
        else:
            next_direction_index = (direction_index - 1) % 4 
            new_direction = direction_clockwise[next_direction_index]  # Left turn
        
        self.direction = new_direction

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        self.head = Point(x, y)
