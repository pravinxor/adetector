#!/usr/bin/env python

import argparse
from colored import stylize, Style, Fore
import numpy as np
import onnx
import onnxruntime as ort
import queue
import subprocess
import threading
from tqdm import tqdm

SAMPLE_RATE = 32000


def seconds_to_hms(seconds):
    hours, remainder = divmod(seconds, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"


def subsample(frame: np.ndarray, scale_factor: int) -> np.ndarray:
    subframe = frame[:len(frame) - (len(frame) % scale_factor)].reshape(
        -1, scale_factor)
    subframe_mean = subframe.max(axis=1)

    subsample = subframe_mean

    if len(frame) % scale_factor != 0:
        residual_frame = frame[len(frame) - (len(frame) % scale_factor):]
        residual_mean = residual_frame.max()
        subsample = np.append(subsample, residual_mean)

    return subsample


def print_results(scores: np.ndarray, precision: int, offset: int,
                  top: np.ndarray, threshold: int):
    for i in top:
        score = int(scores[i] * 100)
        if score >= threshold:
            tqdm.write(
                seconds_to_hms(i * precision / 100 + offset) + ' ' +
                f'{score}%')
        else:
            break


def print_timestamps(
    framewise_output: np.ndarray,
    precision: int,
    threshold: int,
    focus_idx: int,
    offset: int,
):
    focus = framewise_output[:, focus_idx]
    # precision in the amount of milliseconds per timestamp sample (higher values will result in less precise timestamps)
    subsampled_scores = subsample(focus, precision)
    top_indices = np.argpartition(
        subsampled_scores, -len(subsampled_scores))[-len(subsampled_scores):]
    sorted_confidence = top_indices[np.argsort(
        -subsampled_scores[top_indices])]
    print_results(subsampled_scores, precision, offset, sorted_confidence,
                  threshold)


def load_audio(file: str, sr: int, frame_count: int):
    cmd = [
        'ffmpeg', '-i', file, '-f', 's16le', '-ac', '1', '-acodec',
        'pcm_s16le', '-ar',
        str(sr), '-'
    ]
    ffmpeg_process = subprocess.Popen(cmd,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE)

    # Create a queue to store the stdout data
    q = queue.Queue()

    # Define a function to run in a thread
    def enqueue_output(out, q, chunk_size):
        while True:
            data = out.read(chunk_size)
            if data:
                q.put(data)
            else:
                break
        out.close()

    # Start a thread to read data from stdout and write it to the queue
    t = threading.Thread(target=enqueue_output,
                         args=(ffmpeg_process.stdout, q, frame_count * 2))
    t.daemon = True
    t.start()

    while ffmpeg_process.poll() is None or not q.empty():
        try:
            # Read data from the queue
            chunk = q.get(timeout=1)
            if chunk is not None:
                yield np.frombuffer(chunk, dtype=np.int16).reshape(
                    1, -1).astype(np.float32) / (2.0**15)
            else:
                break
        except queue.Empty:
            pass

    _rc = ffmpeg_process.poll()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='bdetector', description='Scans audio files for sounds')
    parser.add_argument('files',
                        metavar='f',
                        nargs='+',
                        type=str,
                        help='files to be processed')

    parser.add_argument(
        '--precision',
        metavar='p',
        nargs='?',
        type=int,
        default=1 * 100,
        help=
        'the precision (in ms) of the timestamp selection process (higher is less precise)'
    )
    parser.add_argument(
        '--block_size',
        metavar='b',
        nargs='?',
        type=int,
        default=60 * 10,
        help=
        'the block size is the amount of seconds (of samples) to process at once. Larger sizes offer better performance, but will consume significantly more memory'
    )
    parser.add_argument(
        '--threshold',
        metavar='t',
        nargs='?',
        type=int,
        default=90,
        help='The confidence threshold for a sound to be reported')
    parser.add_argument('--focus_idx',
                        metavar='i',
                        nargs='?',
                        type=int,
                        help='the index of the audio class to track')
    parser.add_argument('--model',
                        metavar='m',
                        type=str,
                        help='the location of the model file',
                        required=True)

    args = parser.parse_args()

    model = onnx.load(args.model)
    onnx.checker.check_model(model)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # sess_options.optimized_model_filepath = args.model

    ort_session = ort.InferenceSession(args.model,
                                       sess_options,
                                       providers=ort.get_available_providers())

    for file in args.files:
        print(stylize('Inferencing', Style.BOLD), stylize(file, Fore.blue))
        # audio = load_audio(file, sr=samplerate, seq_len=args.batch_size)
        offset = 0

        blocks = load_audio(file, SAMPLE_RATE, SAMPLE_RATE * args.block_size)

        for block in tqdm(blocks, leave=False):
            block_resampled = block

            ort_inputs = {'input': block_resampled}
            framewise_output = ort_session.run(['output'], ort_inputs)[0]

            preds = framewise_output[0]
            print_timestamps(preds, args.precision, args.threshold,
                             args.focus_idx, offset)

            offset += args.block_size
