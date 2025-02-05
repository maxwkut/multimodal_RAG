import json
import os
from pathlib import Path

import cv2
import webvtt
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import WebVTTFormatter

from utils import get_video_id_from_url, maintain_aspect_ratio_resize, str2time


def download_video(video_url, path="/tmp/"):
    # Define output template with filename pattern
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
        "outtmpl": os.path.join(
            path, "%(title)s.%(ext)s"
        ),  # Use the provided path for the download
        "noplaylist": True,  # Ensures only the single video is downloaded (not the whole playlist)
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(
                video_url, download=True
            )  # Download and extract video info
            filename = ydl.prepare_filename(
                info_dict
            )  # Get the full path of the downloaded file

        print(f"Download completed successfully: {filename}")
        return filename

    except Exception as e:
        print(f"Error downloading video: {e}")
        return None  # Return None if there is an error


def get_transcript_vtt(video_url, path="/tmp"):
    video_id = get_video_id_from_url(video_url)
    filepath = os.path.join(path, "captions.vtt")
    if os.path.exists(filepath):
        return filepath

    transcript = YouTubeTranscriptApi.get_transcript(
        video_id, languages=["en-GB", "en"]
    )
    formatter = WebVTTFormatter()
    webvtt_formatted = formatter.format_transcript(transcript)

    with open(filepath, "w", encoding="utf-8") as webvtt_file:
        webvtt_file.write(webvtt_formatted)
    webvtt_file.close()
    return filepath


def extract_and_save_frames_and_metadata(
    path_to_video,
    path_to_transcript,
    path_to_save_extracted_frames,
    path_to_save_metadatas,
):
    """This function extracts frames from a video at specified times based on a transcript,
    resizes and saves those frames, and stores metadata related to each frame in a JSON file.
    """
    metadata = []
    # load video and transcript
    video = cv2.VideoCapture(path_to_video)
    trans = webvtt.read(path_to_transcript)

    # for each video segment specified in the transcript file
    for idx, transcript in enumerate(trans):
        start_time_ms = str2time(transcript.start)
        end_time_ms = str2time(transcript.end)
        mid_time_ms = (end_time_ms + start_time_ms) / 2
        # get the transcript, remove the next-line symbol
        text = transcript.text.replace("\n", " ")
        # get frame at the middle time
        video.set(cv2.CAP_PROP_POS_MSEC, mid_time_ms)
        success, frame = video.read()

        if success:
            # if the frame is extracted successfully, resize it
            image = maintain_aspect_ratio_resize(frame, height=350)
            # save frame as JPEG file
            img_fname = f"frame_{idx}.jpg"
            img_fpath = os.path.join(path_to_save_extracted_frames, img_fname)
            cv2.imwrite(img_fpath, image)

            # prepare the metadata
            single_metadata = {
                "extracted_frame_path": img_fpath,
                "transcript": text,
                "video_segment_id": idx,
                "video_path": path_to_video,
                "mid_time_ms": mid_time_ms,
            }
            metadata.append(single_metadata)

        else:
            print(f"ERROR! Cannot extract frame: idx = {idx}")

    # save metadata of all extracted frames
    fn = os.path.join(path_to_save_metadatas, "metadata.json")
    with open(fn, "w") as outfile:
        json.dump(metadata, outfile)
    return metadata


if __name__ == "__main__":
    vid1_url = "https://www.youtube.com/watch?v=OKJbaoIy9vk"
    vid1_dir = "./data/videos"

    vid1_filepath = download_video(vid1_url, vid1_dir)
    vid1_transcript_filepath = get_transcript_vtt(vid1_url, vid1_dir)
