def process_movie(movie, pipeline):
    from moviepy.editor import VideoFileClip

    def movie_pipeline(image):
        if movie['entire_clip'] or (movie_pipeline.frame >= movie['start_frame'] and movie_pipeline.frame <= movie['end_frame']):
            debug = movie_pipeline.debug_all or movie_pipeline.frame in movie_pipeline.debug_frames
            result = pipeline(
                image,
                write_images=debug,
                prefix='{}/image{}'.format(movie['debug_folder'], movie_pipeline.frame),
                frame=movie_pipeline.frame)

            movie_pipeline.frame += 1
            return result
        else:
            movie_pipeline.frame += 1
            return image

    movie_pipeline.frame = 1
    movie_pipeline.debug_frames = movie['debug_frames']
    movie_pipeline.debug_all = False

    clip = VideoFileClip(movie['input'])

    video_with_overlay = clip.fl_image(movie_pipeline)
    video_with_overlay.write_videofile(movie['output'], audio=False)
