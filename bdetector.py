#!/usr/bin/env python

import argparse
import colored
from colored import stylize
import gc
import numpy as np
import panns_inference
from tqdm import tqdm
import subprocess


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
    offset: int = 0,
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


def chunker(seq: np.ndarray, size: int):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def load_audio(file: str, sr: int):
    cmd = [
        'ffmpeg', '-threads', '0', '-i', file, '-f', 's16le', '-ac', '1',
        '-acodec', 'pcm_s16le', '-ar',
        str(sr), '-'
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / (2.0**
                                                                        15)


if __name__ == '__main__':
    sample_rate = 32000
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
        default=100,
        help=
        'the precision (in ms) of the timestamp selection process (higher is less precise)'
    )

    parser.add_argument('--batch_size',
                        metavar='b',
                        nargs='?',
                        type=int,
                        default=sample_rate * 15 * 60)

    parser.add_argument('--threshold',
                        metavar='t',
                        nargs='?',
                        type=int,
                        default=0)
    parser.add_argument('--focus_idx',
                        metavar='i',
                        type=int,
                        help='The index of the audio class to track')

    args = parser.parse_args()

    sed = panns_inference.SoundEventDetection(
        checkpoint_path=
        '/home/pravin/Projects/bdetector/Cnn14_DecisionLevelMax.pth')

    for file in args.files:
        print(stylize('Inferencing', colored.attr('bold')),
              stylize(file, colored.fg('blue')))
        audio = load_audio(file, sr=sample_rate)
        offset = 0

        total_chunks = len(audio) // args.batch_size + int(
            len(audio) % args.batch_size > 0)
        for chunk in tqdm(chunker(audio, args.batch_size),
                          total=total_chunks,
                          leave=False):
            framewise_output = sed.inference(chunk[None, :])
            print_timestamps(framewise_output[0], args.precision,
                             args.threshold, args.focus_idx, offset)
            offset += len(chunk) / sample_rate
            del chunk
        del audio
