import pygame
from pygame.locals import *
from sys import exit
import gridworld 
from random import *
from math import pi
import time

screen_width = 640
screen_height = 480
pygame.init()
screen = pygame.display.set_mode((screen_width, screen_height), 0, 32)
state = gridworld.initGridRand()
grid_row_count = state.shape[0]
grid_column_count = state.shape[1]
grid_width = screen_width / grid_column_count
grid_height = screen_height / grid_row_count
clock = pygame.time.Clock()

pic_player = pygame.transform.scale(pygame.image.load("resource/player.png").convert_alpha(), (grid_width / 2, grid_width / 2))
pic_goal = pygame.transform.scale(pygame.image.load("resource/goal.png").convert_alpha(), (grid_width / 2, grid_width / 2))
pic_stop = pygame.transform.scale(pygame.image.load("resource/stop.png").convert_alpha(), (grid_width / 2, grid_width / 2))
pic_danger = pygame.transform.scale(pygame.image.load("resource/danger.png").convert_alpha(), (grid_width / 2, grid_width / 2))
pic_win = pygame.transform.scale(pygame.image.load("resource/win.png").convert_alpha(), (grid_width / 2, grid_width / 2))
pic_lose = pygame.transform.scale(pygame.image.load("resource/lose.png").convert_alpha(), (grid_width / 2, grid_width / 2))

def draw_grid(state, timeout):
    rect = pygame.Rect(0, 0, grid_width, grid_height)
    rect.inflate_ip(2, 2)
    player_loc, wall, goal, pit = gridworld.findStuff(state)
    #print(gridworld.findStuff(state))
    for i in range(grid_row_count):
        for j in range(grid_column_count):
            drawing_rect = rect.move(j * grid_width, i * grid_height)
            pygame.draw.rect(screen, (255, 0, 0), drawing_rect, 1)
            if (player_loc[0] == i and player_loc[1] == j):
                if (player_loc == pit):
                    screen.blit(pic_loss, (drawing_rect.centerx - grid_width / 4 , drawing_rect.centery - grid_width / 4))
                elif (player_loc == goal):
                    screen.blit(pic_win, (drawing_rect.centerx - grid_width / 4 , drawing_rect.centery - grid_width / 4))
                elif (timeout == 1):
                    screen.blit(pic_lose, (drawing_rect.centerx - grid_width / 4 , drawing_rect.centery - grid_width / 4))
                else :
                    screen.blit(pic_player, (drawing_rect.centerx - grid_width / 4 , drawing_rect.centery - grid_width / 4))
            elif (goal[0] == i and goal[1] == j):
                if (player_loc != goal):
                    screen.blit(pic_goal, (drawing_rect.centerx - grid_width / 4 , drawing_rect.centery - grid_width / 4))
            elif (wall[0] == i and wall[1] == j):
                screen.blit(pic_stop, (drawing_rect.centerx - grid_width / 4 , drawing_rect.centery - grid_width / 4))
            elif (pit[0] == i and pit[1] == j):
                if (player_loc != pit):
                    screen.blit(pic_danger, (drawing_rect.centerx - grid_width / 4 , drawing_rect.centery - grid_width / 4))               

def draw_state(state, timeout):
    for event in pygame.event.get():
        if event.type == QUIT:
            exit()
    screen.fill((255, 255, 255))
    draw_grid(state, timeout)
    clock.tick(30)
    pygame.display.update()
