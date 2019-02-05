import argparse
import ffmpeg
import logging
import numpy as np
import os
import subprocess
from TSNet import TSNet

parser = argparse.ArgumentParser(description='Streaming ffmpeg to tensorflow & numpy')
parser.add_argument('in_filename', help='Input filename')
parser.add_argument('out_filename', help='Output filename')

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def number(n):
    if type(n) == type("str"):
        if "/" in n:
            n = n.split("/")
            return int(n[0]) / int(n[1])
        elif "." in n:
            return float(n)
    try:
        return int(n)
    except:
        return n
        
def get_video_size(filename):
    logger.info('Getting video size for {!r}'.format(filename))
    probe = ffmpeg.probe(filename)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    # print(video_info)
    width = number(video_info['width'])
    height = number(video_info['height'])
    pix_fmt = number(video_info['pix_fmt'])
    sample_aspect_ratio = number(video_info['sample_aspect_ratio'])
    display_aspect_ratio = video_info['display_aspect_ratio'].replace("/",":")
    try:
        field_order = video_info['field_order']
    except:
        print("WARNING: could not find field order in video info")
        field_order = None
    #r_frame_rate is apparently deprecated
    rate = number(video_info['avg_frame_rate'])
    # duration = number(video_info['duration'])
    # codec_time_base = number(video_info['codec_time_base'])
    return width, height, rate, display_aspect_ratio, field_order


def start_ffmpeg_process1(in_filename, rate):
    logger.info('Starting ffmpeg process1')
    args = (
        ffmpeg
        .input(in_filename)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24', r=rate)
        .compile()
    )
    return subprocess.Popen(args, stdout=subprocess.PIPE)


def start_ffmpeg_process2(out_filename, width, height, rate, display_aspect_ratio):
    logger.info('Starting ffmpeg process2')
    args = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height), r=rate*2)
        .output(out_filename, pix_fmt='yuv420p', r=rate*2, aspect=display_aspect_ratio, preset="ultrafast", crf=0)
        .overwrite_output()
        .compile()
    )
    return subprocess.Popen(args, stdin=subprocess.PIPE)


def read_frame(process1, width, height):
    logger.debug('Reading frame')

    # Note: RGB24 == 3 bytes per pixel.
    frame_size = width * height * 3
    in_bytes = process1.stdout.read(frame_size)
    if len(in_bytes) == 0:
        frame = None
    else:
        assert len(in_bytes) == frame_size
        frame = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([height, width, 3])
        )
    return frame

    
def write_frame(process2, frame):
    logger.debug('Writing frame')
    process2.stdin.write(
        frame
        #astype is already done
        # .astype(np.uint8)
        .tobytes()
    )
    
def run(in_filename, out_filename):
    print(in_filename, out_filename)
    width, height, rate, display_aspect_ratio, field_order = get_video_size(in_filename)
    print(width, height, rate, display_aspect_ratio, field_order)
    model = TSNet()
    model.set_dimensions(width, height)
    process1 = start_ffmpeg_process1(in_filename, rate)
    process2 = start_ffmpeg_process2(out_filename, width, height, rate, display_aspect_ratio)
    while True:
        in_frame = read_frame(process1, width, height)
        if in_frame is None:
            logger.info('End of input stream')
            break

        logger.debug('Processing frame')
        out_frame1, out_frame2 = model.deinterlace_frame_pick(in_frame)
        # out_frame1, out_frame2 = model.bob_frame(in_frame)
        
        #bottom field first
        if field_order == "bb":
            write_frame(process2, out_frame2)
            write_frame(process2, out_frame1)
        #default, top field first
        else:
            write_frame(process2, out_frame1)
            write_frame(process2, out_frame2)
    logger.info('Waiting for ffmpeg process1')
    process1.wait()

    logger.info('Waiting for ffmpeg process2')
    process2.stdin.close()
    process2.wait()


if __name__ == '__main__':
    args = parser.parse_args()
    run(args.in_filename, args.out_filename)
