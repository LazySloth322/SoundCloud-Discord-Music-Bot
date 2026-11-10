"""
tolya.py -- SoundCloud music bot (discord.py 2.x, yt-dlp).

Internal refactor summary (public behaviour, commands, aliases, embeds,
messages, pagination and the 5-minute AFK rule are preserved):

* Per-guild AFK watchdog (``IdleMonitor``) replaces the polling loop that used
  to live inside ``audio_player``. One task per guild, event driven, monotonic
  clock, safely cancelled on manual disconnect / queue clear / voice loss /
  shutdown, re-armed on every playback activity, with a small transition grace
  period so stream extraction between tracks can never cause a false
  disconnect. A paused track still counts as idle time (legacy semantics).
* ``asyncio.run()`` is gone everywhere. The FFmpeg "after" callback (which
  runs on the *player thread*) schedules the next-track coroutine with
  ``asyncio.run_coroutine_threadsafe`` on the bot's main event loop.
* All blocking work (yt-dlp extraction, oEmbed cover lookup) is offloaded to
  worker threads; ``nest_asyncio`` is no longer needed and removed.
* Background playlist processing is a cancellable, guild-scoped task that
  stops on clear / disconnect / voice loss / shutdown (epoch + explicit
  cancel), handles ``None`` playlist entries, uses one YoutubeDL per task and
  a configurable rate-limit sleep.
* Queue state lives in tolya_queue.GuildQueueState: dataclasses, per-guild
  asyncio.Lock, no positional magic, KeyError-safe accessors.
* Magic numbers moved into named constants (config.py values are honoured via
  the wildcard import when present).
* ``-search`` is two-phase: the menu is built from a single flat search-API
  request (``process=False``), and the stream URL / full metadata are resolved
  only for the track the user actually picked.
* Startup voice-stack self-check: PyNaCl, davey, ffmpeg, the libopus encoder
  and FFMPEG_OPTIONS sanity are verified on ``on_ready`` and every missing
  piece is logged as CRITICAL with the exact fix ("bot connects but never
  speaks, no error" is diagnosed instead of silent). ffmpeg's stderr is
  captured per playback and dumped into the log when a track fails.
"""

from __future__ import annotations

import asyncio
import functools
import io
import importlib.util
import platform
import shutil
import subprocess
import time
from logging import FileHandler, Formatter, StreamHandler, WARNING, DEBUG, INFO, getLogger
from math import ceil
import datetime
from datetime import timezone
from typing import Any, Callable, Dict, List, Optional, Set

import discord
import requests
import yt_dlp
from discord import ClientException
from discord.ext import commands

from tolya_string import *  # noqa: F401,F403 -- user-facing strings (kept verbatim)
from config import *        # noqa: F401,F403 -- COMMAND_PREFIX, BOT_TOKEN, YDL_OPTIONS, ...

from tolya_queue import Queue, Track, CURRENT_MARKER

# --------------------------------------------------------------------------- #
#  Logging                                                                    #
# --------------------------------------------------------------------------- #

# One switch for all log output:
#   INFO    = normal operation (startup, playback, disconnects, errors only)
#   DEBUG   = verbose troubleshooting output (AFK monitor ticks, ffmpeg command
#             lines, search/playlist internals, voice handshake)
#   WARNING = nearly silent (problems only)
LOG_LEVEL = INFO

logger = getLogger("tolya")          # legacy code used the literal string '__name__'
logger.setLevel(LOG_LEVEL)

formatter = Formatter(fmt="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

console = StreamHandler()
console.setFormatter(formatter)
console.setLevel(LOG_LEVEL)

file = FileHandler(filename="main.log", encoding="utf-8")
file.setFormatter(formatter)
file.setLevel(LOG_LEVEL)

logger.addHandler(console)
logger.addHandler(file)
logger.propagate = False  # our two handlers are enough; avoid stderr duplication

# discord.py's own logger (voice handshake, DAVE E2EE transitions, player
# warnings) would otherwise vanish into an unconfigured root logger -- route it
# through the same two handlers so voice problems are always visible. At INFO
# it stays at WARNING so the handshake chatter is quiet; DEBUG restores it.
discord_logger = getLogger("discord")
discord_logger.setLevel(DEBUG if LOG_LEVEL == DEBUG else WARNING)
discord_logger.addHandler(console)
discord_logger.addHandler(file)
discord_logger.propagate = False
getLogger("discord.player").setLevel(DEBUG if LOG_LEVEL == DEBUG else WARNING)

# --------------------------------------------------------------------------- #
#  Configuration / constants                                                  #
# --------------------------------------------------------------------------- #

def _cfg(name: str, default):
    """Prefer a constant provided by config.py (via the wildcard import)."""
    value = globals().get(name)
    return default if value is None else value


AFK_TIMEOUT_SECONDS = float(_cfg("AFK_TIMEOUT", 300))              # the 5-minute AFK rule
TRACK_TRANSITION_GRACE_SECONDS = float(_cfg("TRACK_TRANSITION_GRACE", 15))
CONNECT_TIMEOUT_SECONDS = int(_cfg("CONNECT_TIMEOUT", 2))          # legacy ``timeout = 2``
PLAYLIST_RATE_LIMIT_SECONDS = float(_cfg("PLAYLIST_RATE_LIMIT", 0.3))
MAX_PLAYLIST_LENGTH = int(_cfg("MAX_PLAYLIST_LENGTH", 30))
SEARCH_LIMIT = 10                # fixed by product spec -- do not change
SEARCH_TIMEOUT_SECONDS = 30      # fixed by product spec -- do not change
DEFAULT_THUMBNAIL = "https://imgur.com/qSUM27B.png"
MAX_TRACK_LENGTH = datetime.timedelta(seconds=60 * 60 * 24 - 1)    # >24h rejected (legacy check)

# --------------------------------------------------------------------------- #
#  Voice stack self-check (startup diagnostics)                               #
# --------------------------------------------------------------------------- #

_VOICE_DIAGNOSTICS_DONE = False


def _run_voice_diagnostics() -> None:
    """One-time check of everything the voice playback pipeline needs.

    "Bot connects but never speaks, no error" is almost always one of the
    problems checked here: PyNaCl / davey missing (voice packet encryption /
    DAVE E2EE), ffmpeg missing, an ffmpeg build without the libopus encoder,
    or FFMPEG_OPTIONS overriding the opus muxer. Every failure is logged as
    CRITICAL with the exact fix instead of failing silently later inside the
    player thread.
    """
    global _VOICE_DIAGNOSTICS_DONE
    if _VOICE_DIAGNOSTICS_DONE:
        return
    _VOICE_DIAGNOSTICS_DONE = True

    logger.info(f"Voice stack check: discord.py {discord.__version__}, "
                f"yt-dlp {yt_dlp.version.__version__}, Python {platform.python_version()}")

    if importlib.util.find_spec("nacl") is None:
        logger.critical('Voice check: PyNaCl is MISSING -> voice packets cannot be '
                        'encrypted and no audio can be sent. Fix: '
                        'pip install "discord.py[voice]" (or pip install "PyNaCl>=1.5.0,<1.6")')
    if importlib.util.find_spec("davey") is None:
        logger.critical('Voice check: davey is MISSING -> Discord\'s DAVE (E2EE) voice '
                        'protocol cannot be used. Fix: pip install "discord.py[voice]>=2.7" '
                        '(or pip install davey)')

    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe is None:
        logger.critical('Voice check: ffmpeg NOT FOUND in PATH -> FFmpegOpusAudio cannot '
                        'spawn any audio process (join works, playback dies silently). '
                        'Install ffmpeg as a SYSTEM package and put it on PATH: '
                        'Windows: winget install Gyan.FFmpeg | Debian/Ubuntu: '
                        'sudo apt install ffmpeg | macOS: brew install ffmpeg')
        return
    logger.info(f"Voice check: ffmpeg found at {ffmpeg_exe}")

    try:
        encoders = subprocess.run([ffmpeg_exe, "-hide_banner", "-encoders"],
                                  capture_output=True, text=True, timeout=15).stdout
        if "libopus" not in encoders:
            logger.critical('Voice check: this ffmpeg build has NO libopus ENCODER -> '
                            'FFmpegOpusAudio produces silence with no error. Install a '
                            'full ffmpeg build (gyan.dev / BtbN builds on Windows, '
                            'distro package on Linux).')
        muxers = subprocess.run([ffmpeg_exe, "-hide_banner", "-muxers"],
                                capture_output=True, text=True, timeout=15).stdout
        if "opus" not in muxers:
            logger.warning('Voice check: this ffmpeg build has no opus (Ogg) muxer -> '
                           'FFmpegOpusAudio cannot work with it.')
    except Exception as exc:
        logger.warning(f"Voice check: could not probe ffmpeg capabilities: {exc!r}")

    # FFmpegOpusAudio appends FFMPEG_OPTIONS['options'] AFTER its own
    # '-f opus' argument, so a user-supplied '-f ...' (e.g. leftover PCM
    # options from an FFmpegPCMAudio setup) silently overrides the opus muxer
    # -> bot 'plays' perfectly but nobody hears anything.
    try:
        ffmpeg_options = _cfg("FFMPEG_OPTIONS", {})
        option_words = []
        for value in (ffmpeg_options or {}).values():
            if isinstance(value, str):
                option_words.extend(value.split())
        if "-f" in option_words or "s16le" in option_words:
            logger.critical('Voice check: FFMPEG_OPTIONS contains "-f ..."/PCM output '
                            'options -> they override FFmpegOpusAudio\'s opus muxer and '
                            'playback goes SILENT. Remove "-f"/"s16le" options from '
                            'FFMPEG_OPTIONS in config.py.')
    except Exception:
        pass

# --------------------------------------------------------------------------- #
#  Core singletons                                                            #
# --------------------------------------------------------------------------- #

songs_queue: Queue = Queue()

intents = discord.Intents.all()
intents.message_content = True
intents.voice_states = True


class TolyaBot(commands.Bot):
    """commands.Bot with deterministic cleanup of per-guild tasks on shutdown."""

    async def close(self) -> None:
        logger.info("Shutdown requested -- cancelling AFK monitors and background playlist tasks")
        idle_manager.cancel_all()
        _cancel_all_playlist_tasks()
        await super().close()


bot: TolyaBot = TolyaBot(command_prefix=COMMAND_PREFIX, intents=intents)
bot.remove_command("help")

# --------------------------------------------------------------------------- #
#  Small shared helpers                                                       #
# --------------------------------------------------------------------------- #

def _now_ts() -> float:
    """Wall-clock timestamp for track start/pause times (legacy semantics).

    The AFK watchdog does NOT use this -- it uses ``time.monotonic()`` so
    system clock changes cannot corrupt the idle timeout.
    """
    return datetime.datetime.now().timestamp()


async def msg_delete(msg) -> None:
    try:
        await msg.delete()
    except Exception as exc:
        logger.error(f"Message delete: {exc!r}")


async def _to_thread(func: Callable, /, *args, **kwargs):
    """Run a blocking callable in the default executor without blocking the loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))


def _extract_sync(ydl_opts: dict, url: str, process: bool):
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        return ydl.extract_info(url, download=False, process=process)


async def extract_info_async(ydl_opts: dict, url: str, process: bool = True):
    """yt-dlp extraction off the event loop; exceptions propagate to the caller."""
    return await _to_thread(_extract_sync, ydl_opts, url, process)


_http_session = requests.Session()   # connection reuse for the oEmbed cover lookups


def _fetch_cover_sync(url: str):
    response = _http_session.get(f"https://soundcloud.com/oembed?url={url}&format=json", timeout=10)
    return response.json().get("thumbnail_url")


async def getCover(url: str):
    """Cover-art lookup via SoundCloud oEmbed (offloaded to a worker thread).

    Returns ``None`` on any failure; callers fall back to DEFAULT_THUMBNAIL.
    """
    try:
        return await _to_thread(_fetch_cover_sync, url)
    except Exception as exc:
        logger.debug(f"Cover lookup failed for {url}: {exc!r}")
        return None

# --------------------------------------------------------------------------- #
#  Background playlist task registry                                          #
# --------------------------------------------------------------------------- #

_playlist_tasks: Dict[int, Set[asyncio.Task]] = {}


def _register_playlist_task(guild_id: int, task: asyncio.Task) -> None:
    _playlist_tasks.setdefault(guild_id, set()).add(task)

    def _discard(done: asyncio.Task, gid: int = guild_id) -> None:
        tasks = _playlist_tasks.get(gid)
        if tasks is not None:
            tasks.discard(done)

    task.add_done_callback(_discard)


def _cancel_playlist_tasks(guild_id: int) -> None:
    tasks = _playlist_tasks.pop(guild_id, None)
    if tasks:
        for task in tasks:
            task.cancel()
        logger.debug(f"Cancelled {len(tasks)} background playlist task(s) for guild {guild_id}")


def _cancel_all_playlist_tasks() -> None:
    for guild_id in list(_playlist_tasks):
        _cancel_playlist_tasks(guild_id)

# --------------------------------------------------------------------------- #
#  AFK / idle monitor (per guild)                                             #
# --------------------------------------------------------------------------- #

async def _idle_timeout_disconnect(guild: discord.Guild) -> None:
    """AFK watchdog fired: tear the guild down exactly like ``-disconnect`` did
    (clear queue, disconnect, send the usual timeout message, log the usual line)."""
    voice_client = guild.voice_client
    if voice_client is None or not voice_client.is_connected():
        return
    if voice_client.is_playing():
        # last-instant safety re-check: playback resumed while we were firing
        idle_manager.bump(guild.id)
        return
    idle_manager.cancel(guild.id)
    _cancel_playlist_tasks(guild.id)
    songs_queue.clear(guild.id, False)
    try:
        await voice_client.disconnect()
    except Exception as exc:
        logger.error(f'AFK disconnect failed in "{guild.name}": {exc!r}')
        return
    channel = songs_queue.get_text_channel(guild.id) or guild.system_channel
    if channel is not None:
        try:
            await channel.send(DISCONNECT_DISCONNECTED_BY_AFK)
        except Exception as exc:
            logger.error(f'Could not deliver AFK timeout message in "{guild.name}": {exc!r}')
    logger.info(f'Disconnected due to timeout of 5 minutes in "{guild.name}"')


class IdleMonitor:
    """Per-guild AFK watchdog (exactly one per guild, owned by IdleMonitorManager).

    Semantics (identical to the legacy in-player timer, minus its bugs):

    * the bot disconnects after ``AFK_TIMEOUT_SECONDS`` of *continuous*
      inactivity while connected to voice. "Inactive" means nothing is
      playing -- a paused track therefore counts as idle time, exactly as it
      did in the legacy implementation;
    * every playback activity (play start, resume, track transition, skip,
      jump, pause, clear) re-arms the deadline;
    * track transitions get a small grace period so stream-URL extraction
      between tracks never triggers a false disconnect;
    * a long track re-arms the deadline automatically whenever it expires
      while audio is still playing;
    * the monotonic clock is used, so system time changes are irrelevant;
    * the guild and voice client are looked up fresh on every check -- no
      stale ``ctx`` or ``voice_client`` references are kept.
    """

    def __init__(self, bot_ref: TolyaBot, guild_id: int,
                 on_timeout: Callable[[discord.Guild], Any]) -> None:
        self._bot = bot_ref
        self._guild_id = guild_id
        self._on_timeout = on_timeout
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._task: Optional[asyncio.Task] = None
        self._wake = asyncio.Event()
        self._deadline: float = 0.0

    # -- lifecycle -------------------------------------------------------- #

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._loop = asyncio.get_running_loop()
        self._wake = asyncio.Event()
        self._deadline = time.monotonic() + AFK_TIMEOUT_SECONDS
        state = songs_queue.get_state(self._guild_id, create=True)
        state.idle_deadline = self._deadline
        self._task = asyncio.create_task(self._run(), name=f"tolya-idle-monitor-{self._guild_id}")

    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def cancel(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()

    # -- activity --------------------------------------------------------- #

    def bump(self, extra: float = 0.0) -> None:
        """Re-arm the idle deadline. Must be called from the event loop thread."""
        self._bump_impl(extra)

    def _bump_impl(self, extra: float = 0.0) -> None:
        self._deadline = time.monotonic() + AFK_TIMEOUT_SECONDS + extra
        state = songs_queue.get_state(self._guild_id)
        if state is not None:
            state.idle_deadline = self._deadline
        self._wake.set()

    # -- main loop -------------------------------------------------------- #

    async def _run(self) -> None:
        guild_id = self._guild_id
        logger.debug(f"AFK idle monitor running for guild {guild_id} "
                     f"(timeout {AFK_TIMEOUT_SECONDS:.0f}s)")
        try:
            while True:
                remaining = self._deadline - time.monotonic()
                if remaining > 0:
                    self._wake.clear()
                    try:
                        # event-driven sleep: any bump wakes us to recompute the wait
                        await asyncio.wait_for(self._wake.wait(), timeout=remaining)
                    except asyncio.TimeoutError:
                        pass
                    continue

                # Deadline reached -- validate against live state before acting.
                if idle_manager.monitors.get(guild_id) is not self:
                    return                                   # replaced by a newer monitor
                if self._wake.is_set():                      # a bump raced with the timeout
                    self._wake.clear()
                    continue
                guild = self._bot.get_guild(guild_id)
                voice_client = guild.voice_client if guild is not None else None
                if voice_client is None or not voice_client.is_connected():
                    logger.debug(f"AFK idle monitor: voice client gone in guild {guild_id}")
                    return
                if voice_client.is_playing():
                    self._bump_impl(0.0)                     # long track: keep re-arming
                    continue
                # Silent (or paused) for the full window -> AFK timeout.
                await self._on_timeout(guild)
                return
        except asyncio.CancelledError:
            logger.debug(f"AFK idle monitor cancelled for guild {guild_id}")
            raise
        finally:
            logger.debug(f"AFK idle monitor stopped for guild {guild_id}")


class IdleMonitorManager:
    """Registry of per-guild AFK watchdogs (guarantees one task per guild)."""

    def __init__(self, bot_ref: TolyaBot) -> None:
        self._bot = bot_ref
        self.monitors: Dict[int, IdleMonitor] = {}   # guild_id -> active monitor

    def ensure(self, guild_id: int) -> IdleMonitor:
        """Return the running monitor for the guild, creating it if needed."""
        monitor = self.monitors.get(guild_id)
        if monitor is not None and monitor.is_running():
            return monitor
        monitor = IdleMonitor(self._bot, guild_id, _idle_timeout_disconnect)
        self.monitors[guild_id] = monitor
        monitor.start()
        logger.debug(f"AFK idle monitor armed for guild {guild_id} ({AFK_TIMEOUT_SECONDS:.0f}s)")
        return monitor

    def bump(self, guild_id: int, extra: float = 0.0) -> bool:
        monitor = self.monitors.get(guild_id)
        if monitor is not None and monitor.is_running():
            monitor.bump(extra)
            return True
        return False

    def cancel(self, guild_id: int) -> bool:
        monitor = self.monitors.pop(guild_id, None)
        if monitor is None:
            return False
        monitor.cancel()
        logger.debug(f"AFK idle monitor cancelled for guild {guild_id}")
        return True

    def cancel_all(self) -> None:
        for guild_id in list(self.monitors):
            self.cancel(guild_id)


idle_manager = IdleMonitorManager(bot)

# --------------------------------------------------------------------------- #
#  Playback lifecycle helpers                                                 #
# --------------------------------------------------------------------------- #

def _make_after_callback(guild_id: int,
                         stderr_sink: Optional[io.BytesIO] = None):
    """Build the FFmpeg player "after" callback.

    This callback runs on the *player thread*, never on the event loop, so the
    next-track coroutine is scheduled with ``asyncio.run_coroutine_threadsafe``
    (the legacy code called ``asyncio.run()`` here, spinning up a second event
    loop inside the bot's own loop -- the classic source of deadlocks and
    "different loop" errors).

    ``stderr_sink`` is the buffer FFmpegOpusAudio collects ffmpeg's stderr
    into; when playback fails, its content is logged so ffmpeg-level failures
    (missing libopus encoder, dead stream URL, ...) are diagnosable instead of
    silent.
    """
    def _after_track(error: Optional[Exception]) -> None:
        if error is not None:
            logger.error(f"Player error in guild {guild_id}: {error!r}")
            if stderr_sink is not None:
                try:
                    stderr_tail = stderr_sink.getvalue()[-2000:].decode(errors="ignore").strip()
                except Exception:
                    stderr_tail = ""
                if stderr_tail:
                    logger.error(f"ffmpeg stderr (guild {guild_id}): {stderr_tail}")
        try:
            future = asyncio.run_coroutine_threadsafe(_on_track_ended(guild_id), bot.loop)
            future.add_done_callback(_log_track_end_failure)
        except RuntimeError as exc:
            logger.error(f"Could not schedule track-end handler for guild {guild_id}: {exc!r}")
    return _after_track


def _log_track_end_failure(future) -> None:
    try:
        future.result()
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logger.error(f"Track-end handler failed: {exc!r}")


async def _on_track_ended(guild_id: int) -> None:
    """Runs on the event loop after a track finishes or is skipped.

    Advances the queue (respecting loop), starts the next track and re-arms
    the AFK watchdog. Replaces the legacy ``auto_skip`` helper.
    """
    guild = bot.get_guild(guild_id)
    if guild is None:
        idle_manager.cancel(guild_id)
        _cancel_playlist_tasks(guild_id)
        return
    voice_client = guild.voice_client
    if voice_client is None or not voice_client.is_connected():
        # voice disappeared -- nothing valid left to schedule
        idle_manager.cancel(guild_id)
        _cancel_playlist_tasks(guild_id)
        return

    state = songs_queue.get_state(guild_id)
    if state is None:
        return

    will_play = False
    async with state.get_lock():
        if songs_queue.if_queue_exist(guild_id):
            # SoundCloud stream links expire quickly -> always drop the old one
            songs_queue.set_stream_link(guild_id, "")
            songs_queue.queue_next(guild_id)          # no-op while looped
            will_play = songs_queue.get_index(guild_id) < songs_queue.queue_len(guild_id)
            if will_play:
                songs_queue.set_stream_link(guild_id, "")
                songs_queue.set_start_time(guild_id, _now_ts())
                songs_queue.set_no_play_time(guild_id, 0.0)

    if will_play:
        # transition grace: extraction between tracks must never look like AFK
        idle_manager.bump(guild_id, extra=TRACK_TRANSITION_GRACE_SECONDS)
        try:
            await audio_player(guild, voice_client)
        except Exception as exc:
            logger.error(f'audio_player after track end in "{guild.name}": {exc!r}')
    else:
        logger.info(f'Queue ended in "{guild.name}"')
        idle_manager.bump(guild_id)


async def audio_player(ctx, voice_client=None, interrupt_paused: bool = False) -> int:
    """Resolve the stream URL for the current track and start playback.

    Focused purely on *starting* playback -- idle/AFK handling lives entirely
    in :class:`IdleMonitor`. Accepts a ``commands.Context`` (command layer)
    or a ``discord.Guild`` (internal track-end path) as ``ctx``.

    Returns ``0`` on success / nothing to do, ``1`` on failure (legacy contract).
    """
    guild = getattr(ctx, "guild", None) or ctx   # Context -> .guild; Guild passed directly -> itself
    serverid = guild.id
    if voice_client is None:
        voice_client = guild.voice_client

    # remember the channel of the latest playback command so the AFK timeout
    # message lands in the same kind of place the legacy code sent it
    channel = getattr(ctx, "channel", None)
    if channel is not None:
        songs_queue.set_text_channel(serverid, channel)

    if voice_client is None or not voice_client.is_connected():
        logger.debug(f'audio_player: no active voice client in "{guild.name}"')
        return 1

    try:
        if voice_client.is_playing():
            return 0                       # already playing -- queued tracks follow naturally
        if voice_client.is_paused():
            if not interrupt_paused:
                return 0                   # a paused track still owns the channel; just queue
            voice_client.stop()            # explicit request (jump): replace the paused track

        while True:
            yt_link = songs_queue.get_yt_link(serverid)
            if not yt_link:
                logger.info(f'Queue ended in "{guild.name}"')
                return 0
            try:
                info = await extract_info_async(YDL_OPTIONS, yt_link)
                stream_link = info.get("url") if isinstance(info, dict) else None
            except Exception as exc:
                logger.info(f"Skipping unavaliable video in \"{guild.name}\".")
                logger.debug(f"Stream extraction failed for {yt_link}: {exc!r}")
                stream_link = None

            if stream_link:
                if voice_client.is_playing():   # raced with skip/jump during extraction
                    return 0
                songs_queue.set_stream_link(serverid, stream_link)
                songs_queue.set_start_time(serverid, _now_ts())
                songs_queue.set_no_play_time(serverid, 0.0)
                songs_queue.set_jump_used(serverid, False)   # jump flag has served its purpose
                # collect ffmpeg's stderr so playback failures can be diagnosed
                stderr_sink = io.BytesIO()
                ffmpeg_kwargs = dict(FFMPEG_OPTIONS)
                ffmpeg_kwargs.setdefault("stderr", stderr_sink)
                voice_client.play(
                    discord.FFmpegOpusAudio(source=stream_link, **ffmpeg_kwargs),
                    after=_make_after_callback(serverid, stderr_sink),
                )
                idle_manager.ensure(serverid)
                idle_manager.bump(serverid)                  # playing a track resets the idle timer
                logger.info(f'Now playing "{songs_queue.get_track_name(serverid)}" in "{guild.name}"')
                return 0

            # unresolvable track: mark it and move on (legacy behaviour)
            songs_queue.mark_current_unavailable(serverid)
            if not songs_queue.if_queue_exist(serverid):
                logger.info(f'Queue ended in "{guild.name}"')
                return 0
            songs_queue.queue_next(serverid)
    except ClientException as exc:
        logger.error(f'audio_player ClientException in "{guild.name}": {exc!r}')
        return 1
    except Exception as exc:
        logger.error(f'audio_player error in "{guild.name}": {exc!r}', exc_info=True)
        return 1

# --------------------------------------------------------------------------- #
#  Events                                                                     #
# --------------------------------------------------------------------------- #

@bot.event
async def on_ready():
    logger.info(f"logged as {bot.user} successfully")
    try:
        await _to_thread(_run_voice_diagnostics)   # never block the gateway loop
    except Exception as exc:
        logger.warning(f"Voice stack check failed to run: {exc!r}")
    await bot.change_presence(activity=discord.Activity(type=discord.ActivityType.listening,
                                                        name="SoundCloud"))


@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        await ctx.message.reply(COMMAND_ERROR)
        return
    # log everything else instead of letting it vanish (legacy behaviour);
    # no new user-visible messages are introduced
    logger.error(f"Command '{ctx.command}' raised: {error!r}", exc_info=error)


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    await bot.process_commands(message)


@bot.event
async def on_disconnect():
    logger.info("Disconnected")


@bot.event
async def on_voice_state_update(member, before, after):
    """Keep per-guild tasks consistent with the bot's own voice state."""
    if member.id != bot.user.id:
        return
    guild = member.guild
    if after.channel is None and before.channel is not None:
        # bot left voice (manually, kicked, or the channel was deleted)
        idle_manager.cancel(guild.id)
        _cancel_playlist_tasks(guild.id)
        songs_queue.bump_epoch(guild.id)
        logger.info(f'Left voice channel "{before.channel}" in "{guild.name}"')
    elif before.channel is None and after.channel is not None:
        # joined voice without -join (e.g. dragged into a channel)
        idle_manager.ensure(guild.id)


# --------------------------------------------------------------------------- #
#  Default commands                                                           #
# --------------------------------------------------------------------------- #

@bot.command(aliases=["HELP", "Help", "рудз", "РУДЗ", "Рудз"])
async def help(ctx, page: int = 1):
    # cac=current available commands
    cac = [[HELP_PING, HELP_HELLO, HELP_PLAY, HELP_SEARCH, HELP_LOOP, HELP_UNLOOP, HELP_NOWPLAYING,
            HELP_PAUSE, HELP_RESUME, HELP_SKIP, HELP_JUMP, HELP_LEAVE, HELP_QUEUE, HELP_CLEAR],
           [HELP_PLAY_SHORTCUT, HELP_SEARCH_SHORTCUT, HELP_QUEUE_SHORTCUT, HELP_LEAVE_SHORTCUT]]
    maxpage = len(cac)
    if (page < 1 or page > maxpage):
        await ctx.message.reply(HELP_NO_PAGE + f" ({page})")
        return
    else:
        await ctx.message.reply(f'{"".join(cac[page - 1])}\n'
                f'{HELP_CURRENT_PAGE[0]} `{page} {HELP_CURRENT_PAGE[1]} {maxpage}` ({HELP_CURRENT_PAGE[2]} `-help 2`)')


@bot.command(aliases=["PING", "Ping", "зштп", "ЗШТП", "Зштп"])
async def ping(ctx):
    await ctx.message.reply(f"{round(bot.latency * 1000)}ms")


@bot.command(aliases=["HELLO", "Hello", "руддщ", "РУДДЩ", "Руддщ"])
async def hello(ctx):
    await ctx.send(f"{ctx.message.author.mention} :wave:")


# --------------------------------------------------------------------------- #
#  Play / join                                                                #
# --------------------------------------------------------------------------- #

@bot.command(description="plays music by url or keyword search",
             aliases=["p", "PLAY", "P", "здфн", "з", "ЗДФН", "З", "Play", "Здфн"])
async def play(ctx, *url):
    try:
        if not url:
            await ctx.message.reply(PLAY_NO_URL)
            return

        if await join(ctx) == 1:
            return
        
        #await asyncio.sleep(2.0)
        # NOTE: the legacy 2.1s "waiting for join func" sleep was removed --
        # join() is awaited directly, so the sleep only added latency.

        if (await add(ctx, " ".join(url)) == 1):
            logger.error("Play error: Cannot extract info")
            return

        if (await audio_player(ctx) == 1):
            logger.error("Play error: AudioPlayer error")
            return
    except Exception as exc:
        logger.error(f"Play command failed: {exc!r}", exc_info=True)


@bot.command(aliases=["j", "J", "Join", "JOIN", "о", "О", "ощфшт", "Ощшт", "ОЩШТ"])
async def join(ctx):
    try:
        if ctx.message.author.voice:
            if not ctx.voice_client:
                await ctx.author.voice.channel.connect(timeout=CONNECT_TIMEOUT_SECONDS,
                                                       reconnect=False, self_deaf=True)
                idle_manager.ensure(ctx.guild.id)     # AFK watchdog starts with the voice session
                logger.info(f'Joined voice channel "{ctx.author.voice.channel}" in "{ctx.guild.name}"')
            else:
                if ctx.message.author.voice.channel.id != ctx.voice_client.channel.id:
                    if ctx.voice_client.is_playing():
                        await ctx.message.reply(JOIN_IS_IN_OTHER_CHANNEL)
                        return 1
                    else:
                        await ctx.voice_client.disconnect()
                        await ctx.author.voice.channel.connect(timeout=CONNECT_TIMEOUT_SECONDS,
                                                               reconnect=False, self_deaf=True)
                        idle_manager.ensure(ctx.guild.id)
                        logger.info(f'Moved to voice channel "{ctx.author.voice.channel}" '
                                    f'in "{ctx.guild.name}"')
                else:
                    idle_manager.ensure(ctx.guild.id)
        else:
            await ctx.message.reply(JOIN_AUTHOR_NOT_IN_VOICE)
            return 1
    except asyncio.TimeoutError as exc:
        await ctx.channel.send(JOIN_ERROR)
        logger.error(f"Bot cannot connect: {exc!r}")
        return 1

# --------------------------------------------------------------------------- #
#  Add / playlist processing                                                  #
# --------------------------------------------------------------------------- #

async def process_remaining_tracks_in_background(guild_id: int, author_name: str,
                                                 entries, epoch: int) -> None:
    """Add remaining playlist tracks in the background.

    * tied to the guild: aborts when the queue state is reset (epoch changed),
      the bot leaves the guild or drops out of voice;
    * cancellable via the playlist task registry (clear / disconnect / shutdown);
    * one YoutubeDL instance per task, extraction off the event loop;
    * rate-limit sleep moved into PLAYLIST_RATE_LIMIT_SECONDS.
    """
    logger.info(f"Background playlist processing started for guild {guild_id} "
                f"({len(entries)} track(s))")
    added = 0
    ydl = yt_dlp.YoutubeDL(YDL_OPTIONS)
    ydl.params["quiet"] = True
    try:
        for entry in entries:
            # lifecycle guards: stop as soon as the guild state is no longer valid
            if epoch != songs_queue.get_epoch(guild_id):
                logger.info(f"Background playlist processing aborted (queue state reset) "
                            f"for guild {guild_id}")
                return
            guild = bot.get_guild(guild_id)
            if guild is None or guild.voice_client is None:
                logger.info(f"Background playlist processing aborted (bot left voice) "
                            f"for guild {guild_id}")
                return

            sc_url = entry.get("url") if isinstance(entry, dict) else None
            if not sc_url:
                logger.warning(f"Skipping malformed playlist entry in guild {guild_id}: {entry!r}")
                continue
            try:
                info = await _to_thread(ydl.extract_info, sc_url, False, process=False)
                if not info:
                    continue
                name = info.get("title")
                duration = info.get("duration")
                if not name or duration is None:
                    logger.warning(f"Skipping incomplete playlist entry ({sc_url}) "
                                   f"in guild {guild_id}")
                    continue
                length = datetime.timedelta(seconds=int(duration))
                try:
                    thumbs = info["thumbnails"]
                    thumbnail_url = thumbs[len(thumbs) - 1]["url"]
                except Exception:
                    thumbnail_url = DEFAULT_THUMBNAIL

                songs_queue.queue_add(
                    Track(name, length, 0.0, "", sc_url, thumbnail_url, author_name, 0.0, 0.0),
                    guild_id,
                )
                added += 1
                await asyncio.sleep(PLAYLIST_RATE_LIMIT_SECONDS)  # avoid SoundCloud rate limits
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(f"Playlist bg track ({sc_url}) add error: {exc!r}")
        logger.info(f"Background playlist processing finished for guild {guild_id} "
                    f"({added}/{len(entries)} added)")
    except asyncio.CancelledError:
        logger.info(f"Background playlist processing cancelled for guild {guild_id}")
        raise
    finally:
        try:
            ydl.close()
        except Exception:
            pass


async def add(ctx, url: str):
    """Parse the requested URL/keyword and queue it (returns 1 on failure, legacy contract)."""
    # legacy anti-spam: "-play -play -play ..." -> keep the last token only
    if ((f"{COMMAND_PREFIX}]" in url) or (f"{COMMAND_PREFIX}З" in url) or (f"{COMMAND_PREFIX}з" in url)
            or (f"{COMMAND_PREFIX}p" in url) or (f"{COMMAND_PREFIX}P" in url)):
        url = url.split(" ")[-1]

    author = ctx.message.author.name
    serverid = ctx.guild.id

    if "/discover/" in url:
        # "My mix" and "Liked" check (personalised URLs cannot be resolved)
        await ctx.channel.send(ADD_MYMIX_WARNING)
        return 1

    url = url.split("?si=")[0]

    if "/sets/" in url:
        return await _add_playlist(ctx, url, author, serverid)
    return await _add_single(ctx, url, author, serverid)


async def _add_playlist(ctx, url: str, author: str, serverid: int) -> int:
    """Queue a SoundCloud playlist: first track immediately, rest in background."""
    msg = await ctx.channel.send(ADD_ON_PL_ADDING)

    try:
        playlist_info = await extract_info_async(YDL_OPTIONS, url, process=False)
    except Exception as exc:
        await msg_delete(msg)
        await ctx.message.reply(ADD_PL_PARSE_ERROR)
        logger.error(f'Playlist parse error for "{url}": {exc!r}')
        return 1

    entries = list(playlist_info.get("entries") or []) if isinstance(playlist_info, dict) else []
    entries = [entry for entry in entries if isinstance(entry, dict) and entry.get("url")]
    playlist_len = len(entries)
    if playlist_len == 0:
        await msg_delete(msg)
        await ctx.message.reply(ADD_PL_PARSE_ERROR)
        logger.error(f'Playlist "{url}" has no usable entries')
        return 1

    if playlist_len > MAX_PLAYLIST_LENGTH:
        await ctx.message.reply(f"{ADD_PL_LEN_WARNING[0]} ({playlist_len}) "
                                f"{ADD_PL_LEN_WARNING[1]} - {MAX_PLAYLIST_LENGTH} !\n"
                                f"{ADD_PL_LEN_WARNING[2]} {MAX_PLAYLIST_LENGTH} {ADD_PL_LEN_WARNING[3]}.")
        playlist_len = MAX_PLAYLIST_LENGTH

    playlist_name = playlist_info.get("title") or "Unknown playlist"

    try:
        first_entry_url = entries[0]["url"]
        info = await extract_info_async(YDL_OPTIONS, first_entry_url, process=False)
        name = info["title"]
        length = datetime.timedelta(seconds=int(info["duration"]))
    except Exception as exc:
        await msg_delete(msg)
        await ctx.message.reply(ADD_PL_PARSE_ERROR)
        logger.error(f'Playlist first-track error for "{url}": {exc!r}')
        return 1

    main_thumb = await getCover(entries[0].get("url")) or DEFAULT_THUMBNAIL

    songs_queue.queue_add(
        Track(name, length, _now_ts(), "", first_entry_url, main_thumb, author, 0.0, 0.0),
        serverid,
    )

    if playlist_len > 1:
        remaining_entries = entries[1:playlist_len]
        epoch = songs_queue.get_epoch(serverid)
        task = asyncio.create_task(
            process_remaining_tracks_in_background(serverid, author, remaining_entries, epoch),
            name=f"tolya-playlist-bg-{serverid}",
        )
        _register_playlist_task(serverid, task)

    embed = discord.Embed(
        description=f"**{ADD_PL_FINAL_INFO[0]}**\n[{playlist_name}]({url}) "
                    f"({playlist_len} tracks)\n{ADD_PL_FINAL_INFO[1]}\n\n",
        colour=discord.Colour.green())
    embed.set_thumbnail(url=main_thumb)

    await msg_delete(msg)
    await ctx.message.reply(embed=embed)
    logger.info(f"Playlist \"{playlist_name}\" ({url}) requested by {author} in \"{ctx.guild.name}\"")
    return 0


async def _add_single(ctx, url: str, author: str, serverid: int) -> int:
    """Resolve a single track (direct SoundCloud URL or keyword search) and queue it."""
    have_url = "soundcloud.com" in url
    info = None
    hint_msg = None
    try:
        if have_url:
            info = await extract_info_async(YDL_OPTIONS, url.split(" ")[0], process=False)
        else:
            hint_msg = await ctx.channel.send(ADD_SEARCH_HINT)
            raw = await extract_info_async(YDL_OPTIONS, f"scsearch1:{url}")
            info = list(raw["entries"])[0]
        url = info["webpage_url"] if not have_url else info["original_url"]
        name = info["title"]
        length = datetime.timedelta(seconds=int(info["duration"]))
    except Exception as exc:
        if hint_msg is not None:
            await msg_delete(hint_msg)
        await ctx.message.reply(ADD_UNAVAILABLE_OR_AGE_RESTRICTION)
        logger.error(f"Cannot extract info: {exc!r} (Probably unavailable or DRM protected)")
        return 1

    thumbnail_url = await getCover(url) or DEFAULT_THUMBNAIL

    if length > MAX_TRACK_LENGTH:  # more than 24h check (legacy)
        embed = discord.Embed(description=f"{ADD_TRACK_LEN_ERROR[0]} - "
                                          f"[{name}]({url}) {ADD_TRACK_LEN_ERROR[1]}!",
                              colour=discord.Colour.red())
        logger.error(f"Longer than 24h by {author} in \"{ctx.guild.name}\"")
    else:
        songs_queue.queue_add(
            Track(name, length, _now_ts(), "", url, thumbnail_url, author, 0.0, 0.0),
            serverid,
        )
        embed = discord.Embed(description=f"**{ADD_TRACK_FINAL_INFO[0]}**\n"
                                          f"[{name}]({url}) ({length})\n{ADD_TRACK_FINAL_INFO[1]}",
                              colour=discord.Colour.green())
        embed.set_thumbnail(url=thumbnail_url)
        logger.info(f"{name} ({url}) requested by {author} in \"{ctx.guild.name}\"")

    if hint_msg is not None:
        await msg_delete(hint_msg)
    await ctx.message.reply(embed=embed)
    return 0

# --------------------------------------------------------------------------- #
#  Search                                                                     #
# --------------------------------------------------------------------------- #

@bot.command(description="search menu for tracks",
             aliases=["s", "S", "SEARCH", "Ы", "ы", "ыуфкср", "ЫУФКСР", "Search", "Ыуфкср"])
async def search(ctx, *user_request):
    if not user_request:
        await ctx.message.reply(SEARCH_NO_REQUEST)
        return
    if await join(ctx) == 1:
        return
    backslash_n = "\n"

    search_opts = {"quiet": True, "extract_flat": True}  # metadata only (legacy options)
    try:
        # Phase 1 (fast): fetch ONLY the search results.
        # ``process=False`` returns the raw search-API playlist -- exactly one
        # API request, no per-entry track-page / stream-URL resolution, so the
        # menu appears as fast as SoundCloud's search endpoint itself. The
        # stream link and full metadata for the *chosen* track are resolved
        # lazily in phase 2 by add() (a single extraction, only for the picked
        # entry). The query is joined with spaces (the old ``{user_request}``
        # tuple-repr formatting literally searched for "('word', 'word')").
        query = " ".join(user_request)
        info = await extract_info_async(search_opts, f"scsearch{SEARCH_LIMIT}:{query}",
                                        process=False)
        entries = list(info.get("entries") or []) if isinstance(info, dict) else []
        search_result = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            search_result.append({
                "title": entry.get("title"),
                "artist": entry.get("uploader"),
                "url": entry.get("webpage_url") or entry.get("url"),
                "duration": entry.get("duration"),  # kept for future use (legacy)
            })
    except Exception as exc:
        logger.error(f"Search error: {exc!r}")   # legacy code did "Search error: "+e -> TypeError
        await ctx.channel.send(PLAY_ERROR)
        return

    if not search_result:
        # legacy code rendered an empty menu and crashed later on selection
        await ctx.message.reply(ADD_UNAVAILABLE_OR_AGE_RESTRICTION)
        return

    def _format_result(position: int, result: dict) -> str:
        artist = result.get("artist") or ""
        title = result.get("title") or ""
        return f"{position}) {artist} - {title}"

    await ctx.message.reply(f"```"
        f"{backslash_n.join(_format_result(i, r) for i, r in enumerate(search_result, 1))}\n"
        f"\n{SEARCH_MENU_HINT}```")

    try:
        msg = await bot.wait_for("message", timeout=SEARCH_TIMEOUT_SECONDS,
                                 check=lambda m: m.author.id == ctx.author.id)
    except asyncio.TimeoutError:
        logger.debug("[search module] given time ended")
        await ctx.message.reply(SEARCH_OUT_OF_TIME)
        return

    if msg.content:
        msg = msg.content

    if msg == "cancel":
        await ctx.message.reply(SEARCH_CANCELED)
        return

    try:
        choice = int(msg)
        if not (1 <= choice <= SEARCH_LIMIT):
            raise ValueError
        url = search_result[choice - 1]["url"]
        if not url:
            raise ValueError
    except (ValueError, TypeError, IndexError):
        await ctx.message.reply(SEARCH_INVALID_ARG)
        return

    if (await add(ctx, url) == 1):
        return

    await audio_player(ctx)

# --------------------------------------------------------------------------- #
#  Playback control commands                                                  #
# --------------------------------------------------------------------------- #

@bot.command(description="loops current playing track",
             aliases=["LOOP", "Loop", "дщщз", "ДЩЩЗ", "Дщщз"])
async def loop(ctx):
    if ctx.voice_client:
        if ctx.voice_client.is_playing():
            songs_queue.loop(ctx.guild.id)
            idle_manager.bump(ctx.guild.id)
            await ctx.message.reply(BOT_DEFAULT_REPLY_ON_SUCCESS)
            logger.info(f"Loop enabled in \"{ctx.guild.name}\" by {ctx.message.author}")
        else:
            await ctx.message.reply(BOT_NOTHING_IS_PLAYING)
    else:
        await ctx.message.reply(BOT_DISCONNECTED)


@bot.command(description="unloops current playing looped track",
             aliases=["UNLOOP", "Unloop", "гтдщщз", "ГТДЩЩЗ", "Гтдщщз"])
async def unloop(ctx):
    if ctx.voice_client:
        if ctx.voice_client.is_playing():
            songs_queue.unloop(ctx.guild.id)
            idle_manager.bump(ctx.guild.id)
            await ctx.message.reply(BOT_DEFAULT_REPLY_ON_SUCCESS)
            logger.info(f"Loop disabled in \"{ctx.guild.name}\" by {ctx.message.author}")
        else:
            await ctx.message.reply(BOT_NOTHING_IS_PLAYING)
    else:
        await ctx.message.reply(BOT_DISCONNECTED)


@bot.command(description="pauses current track",
             aliases=["PAUSE", "Pause", "stop", "STOP", "Stop",
                      "зфгыу", "ЗФГЫУ", "Зфгыу", "ыещз", "ЫЕЩЗ", "Ыещз"])
async def pause(ctx):
    if ctx.voice_client:
        if ctx.voice_client.is_playing():
            voice = discord.utils.get(bot.voice_clients, guild=ctx.guild)
            if voice:
                voice.pause()
                songs_queue.set_pause_time(ctx.guild.id, _now_ts())
                # the idle clock restarts at the pause moment (legacy semantics:
                # paused time counts towards the 5-minute AFK timeout)
                idle_manager.bump(ctx.guild.id)
                await ctx.message.reply(BOT_DEFAULT_REPLY_ON_SUCCESS)
                logger.info(f"Paused by {ctx.message.author} in \"{ctx.guild.name}\"")
        else:
            await ctx.message.reply(BOT_NOTHING_IS_PLAYING)
    else:
        await ctx.message.reply(BOT_DISCONNECTED)


@bot.command(description="resumes paused track",
    aliases=["RESUME", "Resume", "unpause", "Unpause", "UNPAUSE",
             "куыгьу", "КУЫГЬУ", "Куыгьу", "гтзфгыу", "ГТЗФЫУ", "Гтзфгыу"])
async def resume(ctx):
    if ctx.voice_client:
        voice = discord.utils.get(bot.voice_clients, guild=ctx.guild)
        if voice:
            if voice.is_paused():
                serverid = ctx.guild.id
                now = _now_ts()
                pause_time = songs_queue.get_pause_time(serverid)
                no_play = songs_queue.get_no_play_time(serverid)
                if isinstance(pause_time, (int, float)) and pause_time > 0:
                    base = no_play if isinstance(no_play, (int, float)) and no_play > 0 else 0.0
                    songs_queue.set_no_play_time(serverid, base + (now - pause_time))
                voice.resume()
                idle_manager.bump(serverid)   # resuming playback resets the idle timer
                await ctx.message.reply(BOT_DEFAULT_REPLY_ON_SUCCESS)
                logger.info(f"Resumed by {ctx.message.author} in \"{ctx.guild.name}\"")
        else:
            await ctx.message.reply(BOT_NOTHING_IS_PLAYING)
    else:
        await ctx.message.reply(BOT_DISCONNECTED)


@bot.command(description="skips current track",
             aliases=["ылшз", "ЫЛШЗ", "SKIP", "Skip", "Ылшз"])
async def skip(ctx, fromjump=False):
    voice = discord.utils.get(bot.voice_clients, guild=ctx.guild)
    if ctx.voice_client:
        if ctx.voice_client.is_playing():
            if voice:
                voice.stop()  # the after-callback advances the queue and re-arms the AFK timer
                if not fromjump:
                    logger.info(f"Skipped in server \"{ctx.guild.name}\" by {ctx.message.author}")
                await ctx.message.reply(BOT_DEFAULT_REPLY_ON_SUCCESS)
        else:
            await ctx.message.reply(BOT_NOTHING_IS_PLAYING)
    else:
        await ctx.message.reply(BOT_DISCONNECTED)


@bot.command(description="disconnets bot from voice channel",
    aliases=["DISCONNECT", "DC", "dc", "Dc", "leave", "LEAVE", "Leave", "l", "L",
             "вс", "ВС", "Вс", "дуфму", "ДУФМУ", "Дуфму"])
async def disconnect(ctx, fromTimeout=False):
    if ctx.voice_client:
        idle_manager.cancel(ctx.guild.id)          # criterion: manual disconnect cancels the monitor
        _cancel_playlist_tasks(ctx.guild.id)
        songs_queue.clear(ctx.guild.id, False)
        await ctx.voice_client.disconnect()
        if not fromTimeout:
            await ctx.message.reply(BOT_DEFAULT_REPLY_ON_SUCCESS)
            logger.info(f"Disconnected in server \"{ctx.guild.name}\" by {ctx.message.author}")
        else:
            await ctx.send(DISCONNECT_DISCONNECTED_BY_AFK)
            logger.info(f"Disconnected due to timeout of 5 minutes in \"{ctx.guild.name}\"")
    else:
        if not fromTimeout:
            await ctx.message.reply(BOT_DISCONNECTED)

# --------------------------------------------------------------------------- #
#  Queue commands                                                             #
# --------------------------------------------------------------------------- #

@bot.command(description="shows current queue",
             aliases=["QUEUE", "q", "Q", "йгугу", "ЙГУГУ", "й", "Й", "Queue", "Йгугу"])
async def queue(ctx, page: int = 0):
    serverid = ctx.guild.id
    if ctx.voice_client:
        is_playing = ctx.voice_client.is_playing()
        tempvar = songs_queue.queue(serverid, is_playing)
        if tempvar == -1:
            await ctx.message.reply(QUEUE_EMPTY)
            return
        pages = songs_queue.get_qfp(serverid)
        fq_pages = len(pages)
        if page == 0:
            found = next((i for i, page_text in enumerate(pages) if CURRENT_MARKER in page_text), None)
            if found is not None:
                await ctx.message.reply(f"{pages[found]}\n\n{QUEUE_CURRENT_PAGE[0]} {found + 1} "
                                        f"{QUEUE_CURRENT_PAGE[1]} {fq_pages}. ({QUEUE_CURRENT_PAGE[2]} `-queue 2`)")
            else:
                # legacy fallback (no current marker, e.g. queue viewed while idle): last page
                await ctx.message.reply(f"{pages[-1]}\n{QUEUE_CURRENT_PAGE[0]} {fq_pages} "
                                        f"{QUEUE_CURRENT_PAGE[1]} {fq_pages}. ({QUEUE_CURRENT_PAGE[2]} `-queue 2`)")
        else:
            if page > fq_pages or page < 1:
                await ctx.message.reply(QUEUE_NO_PAGE + f" ({page})")
                return
            await ctx.message.reply(f"{pages[page - 1]}\n{QUEUE_CURRENT_PAGE[0]} {page} "
                                    f"{QUEUE_CURRENT_PAGE[1]} {fq_pages}. ({QUEUE_CURRENT_PAGE[2]} `-queue 2`)")
        songs_queue.set_qfp(serverid, [])
    else:
        await ctx.message.reply(QUEUE_EMPTY)


@bot.command(description="clears queue",
             aliases=["CLEAR", "сдуфк", "СДУФК", "Clear", "Сдуфк"])
async def clear(ctx):
    serverid = ctx.guild.id
    if ctx.voice_client:
        playing = ctx.voice_client.is_playing()
        state = songs_queue.get_state(serverid, create=True)
        async with state.get_lock():
            songs_queue.clear(serverid, playing)
        _cancel_playlist_tasks(serverid)
        idle_manager.bump(serverid)
        await ctx.message.reply(BOT_DEFAULT_REPLY_ON_SUCCESS)
        logger.info(f"Queue cleared in \"{ctx.guild.name}\" by {ctx.message.author} "
                    f"({'kept current track' if playing else 'full reset'})")
    else:
        await ctx.message.reply(BOT_DISCONNECTED)


@bot.command(description="Playing selected track from queue by its number",
             aliases=["JUMP", "огьз", "ОГЬЗ", "Jump", "Огьз"])
async def jump(ctx, value: int = -1):
    serverid = ctx.guild.id
    try:
        queue_check = songs_queue.queue(serverid)
    except Exception:
        logger.debug("[jump] queue render failed; treating queue as present")
        queue_check = 0

    if queue_check == -1:
        await ctx.message.reply(QUEUE_EMPTY)
        return

    if value < 1:
        await ctx.message.reply(f"{JUMP_INVALID_ARG} **`-jump 1`**")
        return

    if value > songs_queue.queue_len(serverid):
        # legacy code sent an empty reply here (accidental bug); a proper
        # range error is the obvious intent
        await ctx.message.reply(f"{JUMP_INVALID_ARG} **`-jump 1`**")
        return

    voice_client = ctx.voice_client
    if voice_client is None:
        # legacy code crashed with AttributeError here; match the other commands
        await ctx.message.reply(BOT_DISCONNECTED)
        return

    current_time = _now_ts()
    state = songs_queue.get_state(serverid, create=True)
    if voice_client.is_playing():
        async with state.get_lock():
            songs_queue.jump(value - 2, serverid)          # skip() advances by one afterwards
            songs_queue.set_start_time(serverid, current_time)
            songs_queue.set_jump_used(serverid, True)
        idle_manager.bump(serverid)
        await skip(ctx)
    else:
        async with state.get_lock():
            songs_queue.jump(value - 1, serverid)
            songs_queue.set_start_time(serverid, current_time)
            songs_queue.set_jump_used(serverid, True)
        idle_manager.bump(serverid)
        await audio_player(ctx, voice_client, interrupt_paused=True)
        await ctx.message.reply(BOT_DEFAULT_REPLY_ON_SUCCESS)
    logger.info(f"Jumped to track №{value} in server \"{ctx.guild.name}\" by {ctx.message.author}")


@bot.command(description="Current playing track", aliases=["np", "тз", "NP", "ТЗ", "Np", "Тз"])
async def now_playing(ctx):
    try:
        if ctx.voice_client:
            if ctx.voice_client.is_playing():
                serverid = ctx.guild.id
                track_length = songs_queue.get_track_length(serverid)
                name = songs_queue.get_track_name(serverid)
                yt_link = songs_queue.get_yt_link(serverid)
                thumbnail_url = songs_queue.get_thumbnail_url(serverid)
                author = songs_queue.get_author(serverid)
                if not isinstance(track_length, datetime.timedelta) or name in (-1, None):
                    logger.debug(f'now_playing: inconsistent queue state in "{ctx.guild.name}"')
                    return

                now = _now_ts()
                start = songs_queue.get_start_time(serverid)
                no_play = songs_queue.get_no_play_time(serverid)
                elapsed = 0.0
                if isinstance(start, (int, float)) and start > 0:
                    elapsed = now - (start + no_play)
                if elapsed < 0:
                    elapsed = 0.0
                current_time = datetime.datetime.fromtimestamp(
                    elapsed, tz=timezone.utc).strftime("%H:%M:%S")

                # progress bar math preserved; zero/negative lengths and
                # overrun positions are guarded instead of crashing
                total_seconds = int(track_length.total_seconds())
                if total_seconds > 0:
                    track_len_part = ceil(total_seconds / 12)
                    dot_place = int(elapsed) // track_len_part
                    dot_place = max(0, min(11, dot_place))
                else:
                    dot_place = 0
                bar = "—" * dot_place + "⬤" + (11 - dot_place) * "—"

                embed = discord.Embed(
                    description=f"**{NOWPLAYING_FINAL_INFO[0]}**\n[{name}]"
                                f"({yt_link})\n\n"
                                f"{current_time} {bar} "
                                f"{track_length}\n\n"
                                f"**{NOWPLAYING_FINAL_INFO[1]}**:\n{author}",
                    colour=discord.Colour.green())
                if thumbnail_url:
                    embed.set_thumbnail(url=thumbnail_url)
                await ctx.message.reply(embed=embed)
            else:
                await ctx.message.reply(BOT_NOTHING_IS_PLAYING)
        else:
            await ctx.message.reply(BOT_DISCONNECTED)
    except Exception as exc:
        logger.error(f"NOW_PLAYING function: {exc!r}", exc_info=True)


# --------------------------------------------------------------------------- #
#  Bot run                                                                    #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    bot.run(BOT_TOKEN, log_formatter=formatter, log_handler=None)
