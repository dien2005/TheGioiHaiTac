import pygame
from level import Level
from settings import *
from os.path import join

class Menu:
    def __init__(self, game):
        self.game = game
        self.display_surface = pygame.display.get_surface()
        self.font = pygame.font.Font(join('..', 'graphics', 'ui', 'runescape_uf.ttf'), 40)

        # Khởi tạo đúng thứ tự
        self.selected = 0
        self.bg_color = '#33323d'
        self.text_color = '#f5f1de'
        self.selected_color = '#92a9ce'
        self.algorithm = 'DFS'
        self.options = ['Start Game', f'Algorithm: {self.algorithm}', 'Quit']

    def input(self):
        keys = pygame.key.get_pressed()

        # Di chuyển lên/xuống menu
        if keys[pygame.K_UP] and not self.prev_keys[pygame.K_UP]:
            self.selected = (self.selected - 1) % len(self.options)
        if keys[pygame.K_DOWN] and not self.prev_keys[pygame.K_DOWN]:
            self.selected = (self.selected + 1) % len(self.options)

        # Điều chỉnh thuật toán khi dòng hiện tại là "Algorithm: ..."
        if self.options[self.selected].startswith('Algorithm'):
            algos = ['DFS', 'A*', 'ANNEALING', 'NO_OBSERVATION', 'BACKTRACKING', 'Q_LEARNING', 'HILL CLIMBING', 'BFS', 'BEAM']
            current_index = algos.index(self.algorithm)

            if keys[pygame.K_RIGHT] and not self.prev_keys[pygame.K_RIGHT]:
                current_index = (current_index + 1) % len(algos)
            elif keys[pygame.K_LEFT] and not self.prev_keys[pygame.K_LEFT]:
                current_index = (current_index - 1) % len(algos)

            self.algorithm = algos[current_index]
            self.options[1] = f'Algorithm: {self.algorithm}'

        # Nhấn ENTER để chọn dòng
        if keys[pygame.K_RETURN]:
            if self.options[self.selected] == 'Start Game':
                self.game.algorithm = self.algorithm

                self.game.bg_music.stop()
                self.game.current_stage = Level(
                    self.game.tmx_maps[self.game.data.current_level],
                    self.game.level_frames,
                    self.game.audio_files,
                    self.game.data,
                    self.game.switch_stage,
                    self.game.algorithm
                )
                self.game.bg_music.play(-1)

            elif self.options[self.selected] == 'Quit':
                pygame.quit()
                sys.exit()

        self.prev_keys = keys


    def draw(self):
        self.display_surface.fill(self.bg_color)
        title_surf = self.font.render('Super Pirate World', False, self.text_color)
        title_rect = title_surf.get_frect(center=(WINDOW_WIDTH / 2, WINDOW_HEIGHT / 4))
        self.display_surface.blit(title_surf, title_rect)

        for index, option in enumerate(self.options):
            color = self.selected_color if index == self.selected else self.text_color
            text_surf = self.font.render(option, False, color)
            text_rect = text_surf.get_frect(center=(WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2 + index * 60))
            self.display_surface.blit(text_surf, text_rect)

    def run(self):
        self.prev_keys = pygame.key.get_pressed()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            self.input()
            self.draw()
            pygame.display.update()
            self.game.clock.tick()

            if self.game.current_stage:
                break
