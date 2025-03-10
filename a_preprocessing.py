import json
import os
from pathlib import Path
import time

import cv2
import webvtt
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import WebVTTFormatter

from utils import get_video_id_from_url, maintain_aspect_ratio_resize, str2time


def download_video(video_url, path="/tmp/"):
    # Define output template with filename pattern
    ydl_opts = {
        # Optimized for speed: Use a lower quality but faster to download
        "format": "best[height<=480][ext=mp4]/best[height<=480]/best",
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
    sample_rate=3,  # Only process every Nth transcript segment
):
    """This function extracts frames from a video at specified times based on a transcript,
    resizes and saves those frames, and stores metadata related to each frame in a JSON file.
    Optimized for speed by processing fewer frames and using smaller frame sizes.
    """
    start_time = time.time()
    metadata = []
    # load video and transcript
    video = cv2.VideoCapture(path_to_video)
    trans = webvtt.read(path_to_transcript)
    
    # Calculate how many frames we'll extract (for logging)
    total_segments = len(trans)
    segments_to_process = len(range(0, total_segments, sample_rate))
    print(f"Processing {segments_to_process} frames out of {total_segments} transcript segments")
    
    # for select video segments specified in the transcript file (using sample_rate)
    processed_count = 0
    for idx, transcript in enumerate(trans):
        # Skip frames based on sample_rate to speed up processing
        if idx % sample_rate != 0:
            continue
            
        processed_count += 1
        if processed_count % 10 == 0:
            print(f"Processed {processed_count}/{segments_to_process} frames")
            
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
            # Use a smaller height for faster processing
            image = maintain_aspect_ratio_resize(frame, height=240)
            # save frame as JPEG file
            img_fname = f"frame_{idx}.jpg"
            img_fpath = os.path.join(path_to_save_extracted_frames, img_fname)
            # Use JPEG quality parameter to reduce file size and improve write speed
            cv2.imwrite(img_fpath, image, [cv2.IMWRITE_JPEG_QUALITY, 85])

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
        
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Frame extraction completed in {processing_time:.2f} seconds. Extracted {len(metadata)} frames.")
    return metadata


if __name__ == "__main__":
    vid1_url = "https://www.youtube.com/watch?v=OKJbaoIy9vk"
    vid1_dir = "data/videos"

    vid1_filepath = download_video(vid1_url, vid1_dir)
    vid1_transcript_filepath = get_transcript_vtt(vid1_url, vid1_dir)

    # output paths to save extracted frames and their metadata
    extracted_frames_path = os.path.join(vid1_dir, "extracted_frame")
    metadatas_path = vid1_dir

    # create these output folders if not existing
    Path(extracted_frames_path).mkdir(parents=True, exist_ok=True)
    Path(metadatas_path).mkdir(parents=True, exist_ok=True)

    # call the function to extract frames and metadatas
    metadatas = extract_and_save_frames_and_metadata(
        vid1_filepath,
        vid1_transcript_filepath,
        extracted_frames_path,
        metadatas_path,
    )
