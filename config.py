import yt_logger
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

DISCORD_BOT_TOKEN = os.getenv("TOLYA_TOKEN") #TEST_TOKEN

# Contructrion above can be reduced just to
#
# DISCORD_BOT_TOKEN = YOUR_DISCORD_BOT_TOKEN
#

MAX_PLAYLIST_LENGTH = 250

COMMAND_PREFIX="-"

BOT_TOKEN=DISCORD_BOT_TOKEN

YDL_OPTIONS = {
    'format': 'bestaudio/best',
    'extractaudio': True,
    'simulate': True,
    'preferredquality': 0,
    'audioformat':'opus',
    'key': 'FFmpegExtractAudio',
    'quiet': True,
    'logger': yt_logger.MyLogger(),
    'player_client': 'web',
    'source_address': '0.0.0.0',
    'force-ipv4': True,
    'cachedir': False
    #'ignoreerrors': True,
}

FFMPEG_PATH = "ffmpeg.exe"
FFMPEG_OPTIONS = {
    "before_options": (
        "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5 "
        "-headers \"User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36\\r\\n"
        "Referer: https://soundcloud.com/\\r\\n\""
    ),
    "options": "-vn"
}