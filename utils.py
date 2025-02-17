import json

import cv2
from PIL import Image


def get_video_id_from_url(video_url):
    import urllib.parse

    url = urllib.parse.urlparse(video_url)
    if url.hostname == "youtu.be":
        return url.path[1:]
    if url.hostname in ("www.youtube.com", "youtube.com"):
        if url.path == "/watch":
            p = urllib.parse.parse_qs(url.query)
            return p["v"][0]
        if url.path[:7] == "/embed/":
            return url.path.split("/")[2]
        if url.path[:3] == "/v/":
            return url.path.split("/")[2]

    return video_url


def str2time(strtime):
    # strip character " if exists
    strtime = strtime.strip('"')
    # get hour, minute, second from time string
    hrs, mins, seconds = [float(c) for c in strtime.split(":")]
    # get the corresponding time as total seconds
    total_seconds = hrs * 60**2 + mins * 60 + seconds
    total_miliseconds = total_seconds * 1000
    return total_miliseconds


def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Grab the image size and initialize dimensions
    dim = None
    (h, w) = image.shape[:2]

    # Return original image if no need to resize
    if width is None and height is None:
        return image

    # We are resizing height if width is none
    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # We are resizing width if height is none
    else:
        # Calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Return the resized image
    return cv2.resize(image, dim, interpolation=inter)


def load_json_file(file_path):
    # Open the JSON file in read mode
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def display_retrieved_results(results):
    print(f"There is/are {len(results)} retrieved result(s)")
    print()
    for i, res in enumerate(results):
        print(
            f'The caption of the {str(i+1)}-th retrieved result is:\n"{results[i].page_content}"'
        )
        print()
        display(Image.open(results[i].metadata["extracted_frame_path"]))
        print("------------------------------------------------------------")
