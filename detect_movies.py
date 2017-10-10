from movie import process_movie
from detect import pipeline
import matplotlib.image as mpimg


def main():
    movies = {
        'project': {
            'input': 'project_video.mp4',
            'output': 'project_video_output.mp4',
            'debug_folder': 'project_video_debug',
            'start_frame': 0,
            'end_frame': 5,
            'entire_clip': True,
            'debug_frames': [1, 31, 61, 91, 121, 151, 181, 211, 241, 271, 301],
        },
        'test': {
            'input': 'test_video.mp4',
            'output': 'test_video_output.mp4',
            'debug_folder': 'test_video_debug',
            'start_frame': 0,
            'end_frame': 5,
            'entire_clip': True,
            'debug_frames': [],
        },
    }

    import sys
    videos = sys.argv[1:]

    for video in videos:
        process_movie(movies[video], pipeline)

if __name__ == '__main__':
    main()
