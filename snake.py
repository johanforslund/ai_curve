#Snake Tutorial Python

import math
import random
import pygame
import tkinter as tk

WINDOW_SIZE = 500
ROWS = 20

pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
clock = pygame.time.Clock()

training = False

class Game:
    def __init__(self):
        self.snake = Snake((10,10))
        draw_border()

    def frame_step(self, action):
        pygame.event.pump()

        self.snake.move(action)

        self.snake.add_cube()

        terminal = self.snake.check_collision()
        if terminal:
            self.snake.reset((10, 10))
            screen.fill((0,0,0))
            draw_border()

        self.snake.draw()

        pygame.display.update()

class Snake(object):
    body = []
    turns = {}
    def __init__(self, pos):
        self.color = (255, 0, 0)
        self.head = Cube(pos)
        self.body.append(self.head)
        self.dirnx = 0
        self.dirny = 1

    def move(self, action):
        if action == None:
            pass
        elif action[0] == 1:
            self.dirnx = -1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action[1] == 1:
            self.dirnx = 1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action[2] == 1:
            self.dirnx = 0
            self.dirny = -1
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action[3] == 1:
            self.dirnx = 0
            self.dirny = 1
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]                    

        for i, c in enumerate(self.body):
            p = c.pos[:]
            if p in self.turns:
                turn = self.turns[p]
                c.move(turn[0],turn[1])
                if i == len(self.body)-1:
                    self.turns.pop(p)
            else:
                if c.dirnx == -1 and c.pos[0] <= 0: c.pos = (ROWS-1, c.pos[1])
                elif c.dirnx == 1 and c.pos[0] >= ROWS-1: c.pos = (0,c.pos[1])
                elif c.dirny == 1 and c.pos[1] >= ROWS-1: c.pos = (c.pos[0], 0)
                elif c.dirny == -1 and c.pos[1] <= 0: c.pos = (c.pos[0],ROWS-1)
                else: c.move(c.dirnx,c.dirny)
        

    def reset(self, pos):
        self.head = Cube(pos)
        self.body = []
        self.body.append(self.head)
        self.turns = {}
        self.dirnx = 0
        self.dirny = 1


    def add_cube(self):
        tail = self.body[-1]
        dx, dy = tail.dirnx, tail.dirny

        if dx == 1 and dy == 0:
            self.body.append(Cube((tail.pos[0]-1,tail.pos[1])))
        elif dx == -1 and dy == 0:
            self.body.append(Cube((tail.pos[0]+1,tail.pos[1])))
        elif dx == 0 and dy == 1:
            self.body.append(Cube((tail.pos[0],tail.pos[1]-1)))
        elif dx == 0 and dy == -1:
            self.body.append(Cube((tail.pos[0],tail.pos[1]+1)))

        self.body[-1].dirnx = dx
        self.body[-1].dirny = dy
        

    def draw(self):
        for i, c in enumerate(self.body):
            if i ==0:
                c.draw(eyes=True)
            else:
                c.draw()

    def check_collision(self):
        for x in range(len(self.body)):
            if self.body[x].pos in list(map(lambda z:z.pos, self.body[x+1:])):
                print('Score: ', len(self.body))
                self.reset((10,10))
                return True

        if self.head.pos[0] == ROWS-1 or self.head.pos[0] == 0 or self.head.pos[1] == ROWS -1 or self.head.pos[1] == 0:
            return True

        return False

class Cube(object):
    def __init__(self,start,dirnx=1,dirny=0,color=(255,0,0)):
        self.pos = start
        self.dirnx = 1
        self.dirny = 0
        self.color = color

    def move(self, dirnx, dirny):
        self.dirnx = dirnx
        self.dirny = dirny
        self.pos = (self.pos[0] + self.dirnx, self.pos[1] + self.dirny)

    def draw(self, eyes=False):
        dis = WINDOW_SIZE // ROWS
        i = self.pos[0]
        j = self.pos[1]

        pygame.draw.rect(screen, self.color, (i*dis+1,j*dis+1, dis-2, dis-2))
        if eyes:
            centre = dis//2
            radius = 3
            circleMiddle = (i*dis+centre-radius,j*dis+8)
            circleMiddle2 = (i*dis + dis -radius*2, j*dis+8)
            pygame.draw.circle(screen, (0,0,0), circleMiddle, radius)
            pygame.draw.circle(screen, (0,0,0), circleMiddle2, radius)

def draw_grid():
    cell_size = WINDOW_SIZE // ROWS

    x = 0
    y = 0
    for l in range(ROWS):
        x = x + cell_size
        y = y + cell_size

        pygame.draw.line(screen, (255,255,255), (x,0),(x, WINDOW_SIZE))
        pygame.draw.line(screen, (255,255,255), (0,y),(WINDOW_SIZE, y))

def draw_border():
    cell_size = WINDOW_SIZE // ROWS
    
    x = 0
    y = 0
    for l in range(ROWS):
        x = x + cell_size
        y = y + cell_size

        wall1 = Cube((l, 0), color=(0, 255, 0))
        wall2 = Cube((0, l), color=(0, 255, 0))
        wall3 = Cube((ROWS-1, l), color=(0, 255, 0))
        wall4 = Cube((l, ROWS-1), color=(0, 255, 0))
        wall1.draw()
        wall2.draw()
        wall3.draw()
        wall4.draw()

def play_controller(game):
    while True:
        pygame.event.pump()

        for event in pygame.event.get():
            keys = pygame.key.get_pressed()
            for key in keys:
                if keys[pygame.K_LEFT]:
                    action = [1, 0, 0, 0]
                elif keys[pygame.K_RIGHT]:
                    action = [0, 1, 0, 0]
                elif keys[pygame.K_UP]:
                    action = [0, 0, 1, 0]
                elif keys[pygame.K_DOWN]:
                    action = [0, 0, 0, 1]
                else:
                    action = None

        game.frame_step(action)

game = Game()
play_controller(game)