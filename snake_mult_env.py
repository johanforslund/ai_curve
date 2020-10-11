import math
import random
import pygame

WINDOW_SIZE = 500

ROWS = 13
CELL_SIZE = WINDOW_SIZE // ROWS

START_POS = (ROWS//2+4, ROWS//2)
RND_START_POS = False
PERSIT_MODIFICATIONS = False
USE_DIVIDER = True

pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
clock = pygame.time.Clock()

class Game:
    def __init__(self, pos_reward, neg_reward):
        self.terminal_cubes = []
        self.snake = Snake(self.get_startpos(), self)
        self.reset()
        self.add_border()
        self.high_score = 0
        self.pos_reward = pos_reward
        self.neg_reward = neg_reward
        
    def get_startpos(self):
      if RND_START_POS:
        return (random.randint(1, ROWS-2), random.randint(1, ROWS-2))
      else:
        return START_POS

    def step(self, action, mouse_click=None):
        #pygame.time.wait(200)
        reward = self.pos_reward
        terminal = False

        pygame.event.pump()

        self.snake.move(action)

        screen.fill((0,0,0))

        terminal = self.snake.check_collision()
        if terminal:
            self.reset()
            terminal = True
            reward = self.neg_reward
            score = len(self.snake.body)
            if score > self.high_score:
                self.high_score = score
            #print('Score: ', score, 'High score: ', self.high_score)

        if mouse_click:
            self.resolve_mouse_action(mouse_click)

        self.snake.extend_body()

        self.draw_terminal_cubes()
        self.snake.draw_head()

        pygame.display.update()
        next_state = pygame.surfarray.array3d(pygame.display.get_surface())

        return (next_state, reward, terminal)

    def add_cube(self, pos, color=(255,0,0)):
        cube = Cube(pos, color=color)
        
        # Check so we dont't append same cube again
        if not any(c.pos == pos for c in self.terminal_cubes):
            self.terminal_cubes.append(cube)

    def remove_cube(self, pos):
        for c in self.terminal_cubes:
            if c.pos == pos:
                self.terminal_cubes.remove(c)
            
    def draw_terminal_cubes(self):
        for c in self.terminal_cubes:
            c.draw()

    def resolve_mouse_action(self, mouse_click):
        mouse_cube_pos = (int((mouse_click[0] / WINDOW_SIZE) * ROWS), int((mouse_click[1] / WINDOW_SIZE) * ROWS))

        # Choose remove if cube already on the position
        if any(c.pos == mouse_cube_pos for c in self.terminal_cubes):
            self.remove_cube(mouse_cube_pos)
        else:
            self.add_cube(mouse_cube_pos, color=(0, 255 , 0))

    def draw_grid(self):
        x = 0
        y = 0
        for l in range(ROWS):
            x = x + CELL_SIZE
            y = y + CELL_SIZE

            pygame.draw.line(screen, (255,255,255), (x,0),(x, WINDOW_SIZE))
            pygame.draw.line(screen, (255,255,255), (0,y),(WINDOW_SIZE, y))

    def add_border(self):
        for l in range(ROWS):
            self.add_cube((l, 0), color=(0, 255, 0))
            self.add_cube((0, l), color=(0, 255, 0))
            self.add_cube((ROWS-1, l), color=(0, 255, 0))
            self.add_cube((l, ROWS-1), color=(0, 255, 0))
            
        # Add divider
        if USE_DIVIDER:
            for i in range(ROWS-2):
                self.add_cube((ROWS//2, i+1), color=(0, 255, 0))

    def step_with_random_click(self, action, divider_click_prob = 1):
        if USE_DIVIDER and divider_click_prob > random.random():
            mouse_pos_x = WINDOW_SIZE // 2
            mouse_pos_y = random.randint(CELL_SIZE + 1, WINDOW_SIZE - CELL_SIZE - 1) 
        else:
            mouse_pos_x = random.randint(CELL_SIZE + 1, WINDOW_SIZE - CELL_SIZE - 1)
            mouse_pos_y = random.randint(CELL_SIZE + 1, WINDOW_SIZE - CELL_SIZE - 1)
                
        return self.step(action, (mouse_pos_x, mouse_pos_y))

    def reset(self):
        if PERSIT_MODIFICATIONS:
            ##### TODO: Remove snake only #####
            self.terminal_cubes.clear()
            self.add_border()
        else:
            self.terminal_cubes.clear()
            self.add_border()
        
        self.snake.reset(self.get_startpos())
        screen.fill((0,0,0))
        
        self.draw_terminal_cubes()

        pygame.display.update()
        state = pygame.surfarray.array3d(pygame.display.get_surface())
        return state

    def complete_reset(self):
        self.terminal_cubes.clear()
        return self.reset()
        
class Snake(object):
    body = []
    turns = {}
    def __init__(self, pos, env):
        self.color = (255, 0, 0)
        self.head = Cube(pos)
        self.body.append(self.head)
        self.dirnx = 0
        self.dirny = 1
        self.env = env
        
    def move(self, action):
        if action == None:
            pass
        elif action[0] == 1:
            self.dirnx = -1
            self.dirny = 0
        elif action[1] == 1:
            self.dirnx = 1
            self.dirny = 0
        elif action[2] == 1:
            self.dirnx = 0
            self.dirny = -1
        elif action[3] == 1:
            self.dirnx = 0
            self.dirny = 1

        self.head.pos = (self.head.pos[0] + self.dirnx, self.head.pos[1] + self.dirny)

    def reset(self, pos):
        self.head = Cube(pos)
        self.body = []
        self.body.append(self.head)
        self.turns = {}
        self.dirnx = 0
        self.dirny = 1

    def extend_body(self):
        pos = (self.head.pos[0] - self.dirnx, self.head.pos[1] - self.dirny)
        self.body.append(Cube(pos))
        self.env.add_cube(pos)

    def draw_head(self):
        self.head.draw(eyes=True)

    def check_collision(self):
        if any(c.pos == self.head.pos for c in self.env.terminal_cubes):
            return True
        return False

class Cube(object):
    def __init__(self,start,dirnx=1,dirny=0,color=(255,0,0)):
        self.pos = start
        self.dirnx = 1
        self.dirny = 0
        self.color = color

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

def play_controller(game):
    while True:
        pygame.event.pump()
        mouse_click = None

        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONUP:
                mouse_click = pygame.mouse.get_pos()
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
        game.step(action, mouse_click)
       
#game = Game(0.1, -1)
#play_controller(game)
