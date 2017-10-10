import reloader
reloader.enable(blacklist=[
    'numpy',
    'types',
    '_frozen_importlib',
    '_frozen_importlib_external',
    'importlib',
    '_imp',
    'heapq',
    'keyword',
    'imp',
    'typing',
    'oinspect',
    'traitlets',
    'matplotlib.pyplot',
    'matplotlib.image',
    'pickle',
    'cv2',
    'sklearn',
    'skimage.color.colorconv',
    'skimage.color.colorlabel',
    'skimage.color.delta_e',
    'skimage.feature',
    'tqdm',
    'scipy.ndimage.measurements',
    ])

import numpy as np
from overlay import writeOverlayText
import pygame
import time
import sys
from moviepy.editor import VideoFileClip
import detect
import traceback
import logging
import threading
from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler


## TODO: 's' for save frame to the "interesting frames" folder.
## TODO: 'a' to switch to the "interesting frames" folder and run on that.

def cvimage_to_pygame(image):
    """Convert cvimage into a pygame image"""
    return pygame.image.frombuffer(image.tostring(), image.shape[1::-1],
                                   "RGB")


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')



    saved_frames = []
    current_frame = 0

    clip = VideoFileClip('project_video.mp4')

    # Set game screen
    screen = pygame.display.set_mode(clip.size)
    pygame.mixer.init()  # Initialize pygame

    clock = pygame.time.Clock()

    running = True
    paused = False
    disabled = False

    class Buffer:
        def __init__(self, frame_iterator):
            self.frame_iterator = frame_iterator
            self.frames = []
            self.i = 0

        def next(self):
            self.i += 1
            if self.i >= len(self.frames):
                try:
                    self.frames.append(next(self.frame_iterator))
                except StopIteration:
                    self.i -= 1

        def prev(self):
            self.i -= 1
            if self.i < 0:
                self.i = 0

        def frame(self):
            return self.frames[self.i-1]



    b = Buffer(clip.iter_frames())

    fps = 0

    def render(reloaded=False):
        frame = b.frame()
        texts = []
        if disabled:
            output = frame
        else:
            try:
                output = detect.pipeline(frame, frame=b.i)
            except Exception as e:
                traceback.print_tb(e.__traceback__)
                output = frame
                texts += '\n'.join(traceback.format_tb(e.__traceback__)).split('\n')
                texts.append(str(e))

        game_render = np.copy(output)
        texts.append('frame: {}, enabled: {}, paused: {}, fps: {}'.format(b.i, 'n' if disabled else 'y', 'y' if paused else 'n', fps))
        if reloaded:
            texts.append('reloaded')
        writeOverlayText(game_render, texts)
        screen.blit(cvimage_to_pygame(game_render), (0, 0))  # Load new image on screen

    def reload_module():
        try:
            reloader.reload(detect)
        except Exception as e:
            traceback.print_tb(e.__traceback__)
            texts += '\n'.join(traceback.format_tb(e.__traceback__)).split('\n')
            texts.append(str(e))
            game_render = np.copy(b.frame())
            writeOverlayText(game_render, texts)
            screen.blit(cvimage_to_pygame(game_render), (0, 0))  # Load new image on screen

        render(reloaded=True)

    class ReloadEventHandler(LoggingEventHandler):
        def __init__(self):
            super().__init__()
            self.timer = None

        def on_any_event(self, event):
            super().on_any_event(event)
            if self.timer == None:
                self.timer = threading.Timer(1.0, self.on_timer)
                self.timer.start()

        def on_timer(self):
            self.timer = None
            print('reloading')
            reload_module()

    path = '.'
    event_handler = ReloadEventHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()


    while running:
        start = time.time()

        if not paused:
            b.next()
            render()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_RIGHT:
                    b.next()
                    render()

                if event.key == pygame.K_LEFT:
                    b.prev()
                    render()

                if event.key == pygame.K_COMMA:
                    for i in range(20):
                        b.prev()
                    render()

                if event.key == pygame.K_PERIOD:
                    for i in range(20):
                        b.next()
                    render()

                if event.key == pygame.K_r:
                    try:
                        reloader.reload(detect)
                    except Exception as e:
                        print(e)
                    render(reloaded=True)
                if event.key == pygame.K_d:
                    disabled = not disabled
                    render()

        end = time.time()
        pygame.display.update()  # Update pygame display

        clock.tick(30)
        fps = 1.0/(end-start)

    observer.stop()
    observer.join()
