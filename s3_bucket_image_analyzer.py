"""
AI S3 Image Analyzer

This project retrieves images from a public AWS S3 bucket using Boto3,
processes the images with PyTorch, and sends them to the OpenRouter
Vision API for AI-powered analysis. The program identifies the type
of image and provides a short description of its contents.
"""


import boto3
import requests
import os
from getpass import getpass

import torch
from torchvision.io import read_image

from botocore import UNSIGNED
from botocore.config import Config

from colorama import Fore, Style, init


# ==========================================================
# COLOR SETTINGS
# ==========================================================

init(autoreset=True)

GREEN = Fore.GREEN
BLUE = Fore.BLUE
CYAN = Fore.CYAN
YELLOW = Fore.YELLOW
RED = Fore.RED
MAGENTA = Fore.MAGENTA
WHITE = Fore.WHITE
RESET = Style.RESET_ALL


# ==========================================================
# AWS S3 CONFIGURATION
# ==========================================================

AWS_REGION = "ap-south-1"

BUCKET_NAME = "rahul-ai-image-analyzer-2026"


# ==========================================================
# OPENROUTER CONFIGURATION
# ==========================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    OPENROUTER_API_KEY = getpass("Enter your OpenRouter API key: ").strip()

MODEL = "google/gemini-2.5-flash"


# ==========================================================
# HEADER
# ==========================================================

print()
print(CYAN + "=" * 60)
print(CYAN + "        AI IMAGE ANALYZER")
print(CYAN + "=" * 60)
print()


# ==========================================================
# CONNECT TO PUBLIC S3
# ==========================================================

try:

    s3 = boto3.client(
        "s3",
        region_name=AWS_REGION,
        config=Config(signature_version=UNSIGNED)
    )

    print(GREEN + "✓ Connected to Public S3")

except Exception as e:

    print(RED + "✗ S3 connection failed")
    print(RED + str(e))
    exit()


# ==========================================================
# GET FILES FROM S3
# ==========================================================

try:

    response = s3.list_objects_v2(
        Bucket=BUCKET_NAME
    )

except Exception as e:

    print(RED + "\n✗ Could not access S3 bucket")
    print(RED + str(e))
    exit()


if "Contents" not in response:

    print(YELLOW + "\n⚠ No files found in the bucket.")
    exit()


print()
print(BLUE + "S3 BUCKET")
print(BLUE + "-" * 60)
print(WHITE + f"Bucket : {BUCKET_NAME}")
print(WHITE + f"Region : {AWS_REGION}")
print()


# ==========================================================
# PROCESS IMAGES
# ==========================================================

for file in response["Contents"]:

    image_name = file["Key"]

    # Only process images
    if not image_name.lower().endswith(
        (".jpg", ".jpeg", ".png", ".webp", ".gif")
    ):
        continue


    print(MAGENTA + "=" * 60)
    print(MAGENTA + "IMAGE FOUND")
    print(MAGENTA + "=" * 60)

    print(WHITE + f"File : {image_name}")


    # ======================================================
    # CREATE PUBLIC S3 URL
    # ======================================================

    image_url = (
        f"https://{BUCKET_NAME}.s3."
        f"{AWS_REGION}.amazonaws.com/"
        f"{image_name}"
    )

    print()
    print(CYAN + "Public S3 URL:")
    print(WHITE + image_url)


    # ======================================================
    # DOWNLOAD IMAGE FOR PYTORCH CHECK
    # ======================================================

    local_file = "temp_" + os.path.basename(image_name)

    try:

        print()
        print(YELLOW + "→ Downloading image...")

        s3.download_file(
            BUCKET_NAME,
            image_name,
            local_file
        )

        print(GREEN + "✓ Image downloaded")


        # ==================================================
        # PYTORCH IMAGE CHECK
        # ==================================================

        image_tensor = read_image(local_file)

        print(
            CYAN
            + f"✓ PyTorch loaded image "
            + f"({image_tensor.shape[1]} x "
            + f"{image_tensor.shape[2]})"
        )

        print(
            WHITE
            + f"  Channels: {image_tensor.shape[0]}"
        )


    except Exception as e:

        print(RED + "✗ Image processing failed")
        print(RED + str(e))

        continue


    # ======================================================
    # OPENROUTER
    # ======================================================

    print()
    print(YELLOW + "→ Sending image to AI...")

    headers = {

        "Authorization":
            f"Bearer {OPENROUTER_API_KEY}",

        "Content-Type":
            "application/json"
    }


    data = {

        "model": MODEL,

        # Keep response small
        "max_tokens": 100,

        "messages": [

            {
                "role": "user",

                "content": [

                    {
                        "type": "text",

                        "text": """
Analyze this image.

Give the answer exactly in this format:

Type: [what kind of image this is]

About: [one simple sentence describing the image]

Keep the answer short and easy to understand.
"""
                    },

                    {
                        "type": "image_url",

                        "image_url": {

                            "url": image_url
                        }
                    }
                ]
            }
        ]
    }


    # ======================================================
    # CALL OPENROUTER
    # ======================================================

    try:

        ai_response = requests.post(

            "https://openrouter.ai/api/v1/chat/completions",

            headers=headers,

            json=data,

            timeout=60
        )


        # ==================================================
        # SUCCESS
        # ==================================================

        if ai_response.status_code == 200:

            result = ai_response.json()

            answer = (
                result["choices"][0]
                ["message"]
                ["content"]
            )


            print()
            print(GREEN + "=" * 60)
            print(GREEN + "              AI ANALYSIS")
            print(GREEN + "=" * 60)

            print(WHITE + answer)

            print(GREEN + "=" * 60)


        # ==================================================
        # ERROR
        # ==================================================

        else:

            print()
            print(RED + "=" * 60)
            print(RED + "             OPENROUTER ERROR")
            print(RED + "=" * 60)

            print(
                RED
                + f"Status Code: {ai_response.status_code}"
            )

            print(RED + ai_response.text)


    except Exception as e:

        print(RED + "\n✗ API request failed")
        print(RED + str(e))


    # ======================================================
    # DELETE TEMPORARY FILE
    # ======================================================

    if os.path.exists(local_file):

        os.remove(local_file)

        print()
        print(CYAN + "✓ Temporary file removed")


print()
print(CYAN + "=" * 60)
print(CYAN + "              ANALYSIS COMPLETE")
print(CYAN + "=" * 60)