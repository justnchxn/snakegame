import pygame
import random

pygame.init()

class SnakeGame:

    def __init__(self, w=800, h=800):
        self.w = w
        self.h = h
        #initialise display
        
        #initialise game state
    def game_step(self):
        pass

if __name__ == '__main__':
    game = SnakeGame()

    # game loop
    while True:
        game.game_step()

        # end game loop when player loses

    pygame.quit()
