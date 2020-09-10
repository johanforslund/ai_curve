import pygame
import math
import numpy as np
import matplotlib.pyplot as plt

GAME_WIDTH = 288
GAME_HEIGHT = 512

pygame.init()
screen = pygame.display.set_mode((GAME_WIDTH, GAME_HEIGHT))
clock = pygame.time.Clock()

class GameState:
    def __init__(self):
        screen_center = screen.get_rect().center
        self.p_x = screen_center[0]
        self.p_y = screen_center[1]

        self.p_speed = 5.0
        self.p_angle = 0.0

    def frame_step(self, input_actions):
        pygame.event.pump()

        old_x = self.p_x
        old_y = self.p_y

        reward = 0.1
        terminal = False

        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')
        
        if input_actions[0] == 1: #Forward
            self.update_position()
        if input_actions[1] == 1: # Left
            self.p_angle -= 10
            self.update_position()
        if input_actions[2] == 1: # Right
            self.p_angle += 10
            self.update_position()

        if self.has_collided():
            reward = -1
            terminal = True
            self.__init__()

        pygame.draw.line(screen, (255, 0, 0), [self.p_x, self.p_y], [old_x, old_y], width=5)

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()

        clock.tick(30)

        return image_data, reward, terminal

    def update_position(self):
        self.p_x += int(self.p_speed * math.cos(math.radians(self.p_angle)))
        self.p_y += int(self.p_speed * math.sin(math.radians(self.p_angle)))

    def has_collided(self):
        if self.p_x < 0 or self.p_x > GAME_WIDTH:
            return True
        if self.p_y < 0 or self.p_y > GAME_HEIGHT:
            return True
        return False

actions = np.array([1, 0, 0])

game = GameState()

terminal = False

while not terminal:
    image_data, reward, terminal = game.frame_step(actions)
