"""
Environment for Dino Run 
"""
import os
import cv2
from pygame import RLEACCEL
from pygame.image import load
from pygame.sprite import Sprite, Group, collide_mask
from pygame import Rect, init, time, display, mixer, transform, Surface
from pygame.surfarray import array3d
import torch
from random import randrange, choice
import numpy as np

mixer.pre_init(44100, -16, 2, 2048)
init()

screen_size = (width, height) = (600, 150)
fps = 30
gravity = 0.6

black = (0, 0, 0)
white = (255, 255, 255)
background_col = (235, 235, 235)

high_score = 0

scorelist = []
screen = display.set_mode(screen_size)
clock = time.Clock()
display.set_caption("AI Practicum Final Project")


def load_image(
        name,
        x=-1,
        y=-1,
        colorkey=None,
):
    fullname = os.path.join("assets/sprites", name)
    image = load(fullname)
    image = image.convert()
    if colorkey is not None:
        if colorkey == -1:
            colorkey = image.get_at((0, 0))
        image.set_colorkey(colorkey, RLEACCEL)

    #if image is found
    if x != -1 or y != -1:
        image = transform.scale(image, (x, y))

    return (image, image.get_rect())


def load_sprite_sheet(
        sheetname,
        nx,
        ny,
        scale_x=-1,
        scale_y=-1,
        colorkey=None,
):
    fullname = os.path.join("assets/sprites", sheetname)
    sheet = load(fullname)
    sheet = sheet.convert()
    s_rect = sheet.get_rect()
    sprites = []

    y = s_rect.height / ny
    if isinstance(nx, int):
        x = s_rect.width / nx
        for a in range(0, ny):
            for b in range(0, nx):
                rect = Rect((b * x, a * y, x, y))
                pic = Surface(rect.size)
                pic = pic.convert()
                pic.blit(sheet, (0, 0), rect)

                if colorkey is not None:
                    if colorkey == -1:
                        colorkey = pic.get_at((0, 0))
                    pic.set_colorkey(colorkey, RLEACCEL)

                if scale_x != -1 or scale_y != -1:
                    pic = transform.scale(pic, (scale_x, scale_y))

                sprites.append(pic)

    else:  #list
        x_ls = []
        for i_nx in nx:
            x_ls.append(s_rect.width / i_nx)
      
        for a in range(0, ny):
            for i_nx, x, i_scale_x in zip(nx, x_ls, scale_x):
                for b in range(0, i_nx):
                    rect = Rect((b * x, a * y, x, y))
                    pic = Surface(rect.size)
                    pic = pic.convert()
                    pic.blit(sheet, (0, 0), rect)

                    if colorkey is not None:
                        if colorkey == -1:
                            colorkey = pic.get_at((0, 0))
                        pic.set_colorkey(colorkey, RLEACCEL)

                    if i_scale_x != -1 or scale_y != -1:
                        pic = transform.scale(pic, (i_scale_x, scale_y))

                    sprites.append(pic)

    sprite_rect = sprites[0].get_rect()

    return sprites, sprite_rect


def get_digits(n):
    if n > -1:
        list_digits = []
        while n / 10 != 0:
            list_digits.append(n % 10)
            n = int(n / 10)

        list_digits.append(n % 10)
        for i in range(len(list_digits), 5):
            list_digits.append(0)
        list_digits.reverse()
        return list_digits


def pre_processing(pic, w=84, h=84):
    pic = pic[:300, :, :]
    pic = cv2.cvtColor(cv2.resize(pic, (w, h)), cv2.COLOR_BGR2GRAY)
    _, pic = cv2.threshold(pic, 127, 255, cv2.THRESH_BINARY)
    return pic[None, :, :].astype(np.float32)


class Dino():
    def __init__(self, x=-1, y=-1):
        self.firstImages, self.rect = load_sprite_sheet("dino.png", 5, 1, x, y, -1)
        self.secImage, self.rect1 = load_sprite_sheet("dino_ducking.png", 2, 1, 59, y, -1)
        self.stand_width = self.rect.width
        self.duck_width = self.rect1.width

        self.rect.bottom = int(0.98 * height)
        self.rect.left = width / 15
        self.image = self.firstImages[0]

        self.idx = 0
        self.count = 0
        self.score = 0
        self.movement = [0, 0]
        self.jumpSpeed = 11.5

        self.jumping = False
        self.dead = False
        self.ducking = False
        self.blinking = False

        

    def draw(self):
        screen.blit(self.image, self.rect)

    def checkboundarylines(self):
        if self.rect.bottom > int(0.98 * height):
            self.rect.bottom = int(0.98 * height)
            self.jumping = False

    def update(self):
        if self.jumping:
            self.movement[1] = self.movement[1] + gravity

        if self.jumping:
            self.idx = 0

        elif self.ducking:
            if self.count % 5 == 0:
                self.idx = (self.idx + 1) % 2

        elif self.blinking:
            if self.idx != 0:
                if self.count % 20 == 19:
                    self.idx = (self.idx + 1) % 2
            else:
                if self.count % 400 == 399:
                    self.idx = (self.idx + 1) % 2

        else:
            if self.count % 5 == 0:
                self.idx = (self.idx + 1) % 2 + 2

        if self.ducking:
            self.image = self.secImage[self.idx % 2]
            self.rect.width = self.duck_width

        else:
            self.image = self.firstImages[self.idx]
            self.rect.width = self.stand_width

        if self.dead:
            self.idx = 4


        self.rect = self.rect.move(self.movement)
        self.checkboundarylines()

        if not self.dead and self.count % 7 == 6 and self.blinking == False:
            self.score += 1

        self.count = (self.count + 1)


class Cactus(Sprite):
    def __init__(self, speed=5, x=-1, y=-1):
        Sprite.__init__(self, self.containers)
        self.images, self.rect = load_sprite_sheet("cacti-small.png", [2, 3, 6], 1, x, y, -1)
        self.image = self.images[randrange(0, 11)]
        self.rect.bottom = int(0.98 * height)
        self.rect.left = width + self.rect.width

        self.movement = [-1 * speed, 0]

    def draw(self):
        screen.blit(self.image, self.rect)

    def update(self):
        self.rect = self.rect.move(self.movement)
        if self.rect.right < 0:
            self.kill()


class Ptera(Sprite):
    def __init__(self, speed=5, x=-1, y=-1):
        Sprite.__init__(self, self.containers)
        self.images, self.rect = load_sprite_sheet("ptera.png", 2, 1, x, y, -1)
        self.ptera_height = [height * 0.82, height * 0.75, height * 0.60, height * 0.48]
        self.rect.centery = self.ptera_height[randrange(0, 4)]
        self.rect.left = width + self.rect.width
        self.idx = 0
        self.count = 0
        self.image = self.images[0]
        self.movement = [-1 * speed, 0]


    def draw(self):
        screen.blit(self.image, self.rect)

    def update(self):
        if self.count % 10 == 0:
            self.idx = (self.idx + 1) % 2
        self.image = self.images[self.idx]
        self.rect = self.rect.move(self.movement)
        self.count = (self.count + 1)
        if self.rect.right < 0:
            self.kill()


class Ground():
    def __init__(self, speed=-5):
        self.image, self.rect = load_image("ground.png", -1, -1, -1)
        self.image1, self.rect1 = load_image("ground.png", -1, -1, -1)
        self.rect.bottom = height
        self.rect1.bottom = height
        self.rect1.left = self.rect.right
        self.speed = speed

    def draw(self):
        screen.blit(self.image, self.rect)
        screen.blit(self.image1, self.rect1)

    def update(self):
        self.rect.left += self.speed
        self.rect1.left += self.speed

        if self.rect.right < 0:
            self.rect.left = self.rect1.right
        if self.rect1.right < 0:
            self.rect1.left = self.rect.right


class Cloud(Sprite):
    def __init__(self, x, y):
        Sprite.__init__(self, self.containers)
        self.image, self.rect = load_image("cloud.png", int(90 * 30 / 42), 30, -1)
        self.speed = 1
        self.movement = [-1 * self.speed, 0]
        self.rect.left = x
        self.rect.top = y
        

    def draw(self):
        screen.blit(self.image, self.rect)

    def update(self):
        self.rect = self.rect.move(self.movement)
        if self.rect.right < 0:
            self.kill()


class Scoreboard():
    def __init__(self, x=-1, y=-1):
        self.score = 0
        self.image = Surface((55, int(11 * 6 / 5)))
        self.tempimages, self.temprect = load_sprite_sheet("numbers.png", 12, 1, 11, int(11 * 6 / 5), -1)
        self.rect = self.image.get_rect()
        if x != -1:
            self.rect.left = x
        else:
            self.rect.left = width * 0.89
        if y != -1:
            self.rect.top = y
        else:
            self.rect.top = height * 0.1

    def draw(self):
        screen.blit(self.image, self.rect)

    def update(self, score):
        score_digits = get_digits(score)
        self.image.fill(background_col)
        #getting above 10000 points
        if len(score_digits) == 6:
            score_digits = score_digits[1:]
        for i in score_digits:
            self.image.blit(self.tempimages[i], self.temprect)
            self.temprect.left += self.temprect.width
        self.temprect.left = 0


class ChromeDino(object):
    def __init__(self):
        self.gameOver = False
        self.gameQuit = False
        self.gameSpeed = 5
        
        self.playerDino = Dino(44, 47)
        self.new_ground = Ground(-1 * self.gameSpeed)
        self.scorecount = Scoreboard()
        self.highsc = Scoreboard(width * 0.78)
        self.count = 0

        self.cacti = Group()
        self.pteras = Group()
        self.clouds = Group()
        self.last_obstacle = Group()

        Cactus.containers = self.cacti
        Ptera.containers = self.pteras
        Cloud.containers = self.clouds

        self.retbutton_image, self.retbutton_rect = load_image("replay_button.png", 35, 31, -1)
        self.gameover_image, self.gameover_rect = load_image("game_over.png", 190, 11, -1)

        self.temp_images, self.temp_rect = load_sprite_sheet("numbers.png", 12, 1, 11, int(11 * 6 / 5), -1)
        self.HI_image = Surface((22, int(11 * 6 / 5)))
        self.HI_rect = self.HI_image.get_rect()
        self.HI_image.fill(background_col)
        self.HI_image.blit(self.temp_images[10], self.temp_rect)
        self.temp_rect.left += self.temp_rect.width
        self.HI_image.blit(self.temp_images[11], self.temp_rect)
        self.HI_rect.top = height * 0.1
        self.HI_rect.left = width * 0.73


        

    def step(self, action, record=False):  # 0: Do nothing. 1: Jump. 2: Duck
        reward = 0.1
        if action == 0:
            reward += 0.01
            self.playerDino.ducking = False
        elif action == 1:
            self.playerDino.ducking = False
            if self.playerDino.rect.bottom == int(0.98 * height):
                self.playerDino.jumping = True
                self.playerDino.movement[1] = -1 * self.playerDino.jumpSpeed

        elif action == 2:
            if not (self.playerDino.jumping and self.playerDino.dead) and self.playerDino.rect.bottom == int(
                    0.98 * height):
                self.playerDino.ducking = True

        for c in self.cacti:
            c.movement[0] = -1 * self.gameSpeed
            if collide_mask(self.playerDino, c):
                self.playerDino.dead = True
                reward = -1
                break
            else:
                if c.rect.right < self.playerDino.rect.left < c.rect.right + self.gameSpeed + 1:
                    reward = 1
                    break

        for p in self.pteras:
            p.movement[0] = -1 * self.gameSpeed
            if collide_mask(self.playerDino, p):
                self.playerDino.dead = True
                reward = -1
                break
            else:
                if p.rect.right < self.playerDino.rect.left < p.rect.right + self.gameSpeed + 1:
                    reward = 1
                    break

        if len(self.cacti) < 2:
            if len(self.cacti) == 0 and len(self.pteras) == 0:
                self.last_obstacle.empty()
                self.last_obstacle.add(Cactus(self.gameSpeed, [60, 40, 20], choice([40, 45, 50])))
            else:
                for l in self.last_obstacle:
                    if l.rect.right < width * 0.7 and randrange(0, 50) == 10:
                        self.last_obstacle.empty()
                        self.last_obstacle.add(Cactus(self.gameSpeed, [60, 40, 20], choice([40, 45, 50])))

        # if len(self.pteras) == 0 and randrange(0, 200) == 10 and self.count > 500:
        if len(self.pteras) == 0 and len(self.cacti) < 2 and randrange(0, 50) == 10 and self.count > 500:
            for l in self.last_obstacle:
                if l.rect.right < width * 0.8:
                    self.last_obstacle.empty()
                    self.last_obstacle.add(Ptera(self.gameSpeed, 46, 40))

        if len(self.clouds) < 5 and randrange(0, 300) == 10:
            Cloud(width, randrange(height / 5, height / 2))

        self.playerDino.update()
        self.cacti.update()
        self.pteras.update()
        self.clouds.update()
        self.new_ground.update()
        self.scorecount.update(self.playerDino.score)

        state = display.get_surface()
        screen.fill(background_col)
        self.new_ground.draw()
        self.clouds.draw(screen)
        self.scorecount.draw()
        self.cacti.draw(screen)
        self.pteras.draw(screen)
        self.playerDino.draw()

        display.update()
        clock.tick(fps)

        temp = 0
        
        if self.playerDino.score != 0:
            temp = max(self.playerDino.score, temp)
       

            

        if self.playerDino.dead:
            self.gameOver = True

        self.count = (self.count + 1)

        if self.gameOver:
            self.__init__()
            scorelist.append(temp)
            print(scorelist)
        
            

        state = array3d(state)
        if record:
            return torch.from_numpy(pre_processing(state)), np.transpose(
                cv2.cvtColor(state, cv2.COLOR_RGB2BGR), (1, 0, 2)), reward, not (reward > 0)
        else:
            return torch.from_numpy(pre_processing(state)), reward, not (reward > 0)
