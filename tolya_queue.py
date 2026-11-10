"""
tolya_queue.py -- per-guild queue state for the Tolya SoundCloud Discord bot.

Internal refactor (the public queue behaviour is preserved exactly):

* ``Track`` / ``GuildQueueState`` dataclasses replace the fragile positional
  lists ``[trackindex, [tracks], loop_flag, timer, isJumpUsed]``.
* Queue rendering builds a *display copy* -- the stored track title is never
  temporarily mutated any more (removes the display race condition).
* Pagination uses ``PAGE_SIZE`` tracks per page and never emits an empty
  trailing page when the queue length is an exact multiple of ``PAGE_SIZE``.
* Every accessor tolerates unknown guilds / missing tracks and keeps the
  legacy sentinel return values (``-1`` / ``False`` / ``0`` / ``[]``)
  instead of raising ``KeyError`` / ``IndexError``.
* Each guild state carries its own ``asyncio.Lock`` (lazily created) so the
  playback layer can serialise multi-step mutations (clear / jump / track
  transitions).
* ``idle_deadline`` (monotonic clock) stores the AFK deadline; the actual
  watchdog lives in tolya.py (``IdleMonitor``) and owns this value.
* ``epoch`` is bumped on every clear/reset and lets background playlist
  tasks detect that their guild state was reset.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Dict, List, Optional, Union

from tolya_string import *  # noqa: F401,F403 -- QUEUE_CURRENT / QUEUE_LOOPED markers

# --------------------------------------------------------------------------- #
#  Constants                                                                  #
# --------------------------------------------------------------------------- #

PAGE_SIZE: int = 20  # tracks per queue page (user-facing -- do not change)

#: substring that identifies the queue page containing the currently playing track
CURRENT_MARKER: str = f"\t\t — {QUEUE_CURRENT}"

#: prefix prepended to a track title when its stream URL can no longer be resolved
UNAVAILABLE_PREFIX: str = "(UNAVAILABLE) "

QueueId = Union[int, str]  # Discord guild IDs are ints; str kept for legacy tolerance


# --------------------------------------------------------------------------- #
#  Data model                                                                 #
# --------------------------------------------------------------------------- #

@dataclass
class Track:
    """One queue entry.

    Field order mirrors the legacy 9-element list format
    ``[name, length, start_time, stream_link, yt_link, thumbnail, author,
    pause_time, no_play_time]`` so that old positional constructions keep
    working through :meth:`Queue.queue_add`.
    """

    title: str
    length: timedelta
    start_time: float = 0.0        # wall-clock ts when playback started (legacy semantics)
    stream_link: str = ""          # direct stream URL (SoundCloud links expire quickly)
    yt_link: str = ""              # source page URL used for stream extraction
    thumbnail_url: str = ""
    author: str = ""               # who requested the track
    pause_time: float = 0.0        # wall-clock ts of the last pause
    no_play_time: float = 0.0      # accumulated paused time for progress calculation


@dataclass
class GuildQueueState:
    """All per-server playback state.

    Replaces the legacy ``[index, tracks, loop_flag, timer, isJumpUsed]`` list.
    """

    index: int = 0                                  # current track position (0-based)
    tracks: List[Track] = field(default_factory=list)
    loop_flag: bool = False
    jump_used: bool = False                         # set by -jump, cleared once playback starts
    idle_deadline: float = 0.0                      # monotonic deadline, owned by IdleMonitor
    epoch: int = 0                                  # bumped on clear/reset; invalidates bg tasks
    print_cache: List[str] = field(default_factory=list)   # last `-queue` render (pages)
    text_channel: Optional[Any] = None              # last channel used for playback commands
    lock: Optional["asyncio.Lock"] = None           # lazily created per-guild lock

    def get_lock(self) -> "asyncio.Lock":
        """Return the guild lock, creating it on first use (inside the running loop)."""
        if self.lock is None:
            self.lock = asyncio.Lock()
        return self.lock

    def is_pristine(self) -> bool:
        """True when nothing was ever queued for this guild (legacy ``-1`` condition)."""
        return (
            self.index == 0
            and not self.tracks
            and not self.loop_flag
            and not self.jump_used
        )


# --------------------------------------------------------------------------- #
#  Queue container                                                             #
# --------------------------------------------------------------------------- #

class Queue:
    """Per-guild queue container.

    All original public methods keep their names, parameters and return
    conventions (sentinels ``-1`` / ``False`` on missing data) so the
    playback layer in tolya.py stays compatible.
    """

    def __init__(self) -> None:
        self._states: Dict[QueueId, GuildQueueState] = {}

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                  #
    # ------------------------------------------------------------------ #

    def get_state(self, serverid: QueueId, create: bool = False) -> Optional[GuildQueueState]:
        """Return the guild state or ``None`` (optionally create it)."""
        state = self._states.get(serverid)
        if state is None and create:
            state = GuildQueueState()
            self._states[serverid] = state
        return state

    def _require(self, serverid: QueueId) -> GuildQueueState:
        """Return the guild state, creating it if missing."""
        state = self._states.get(serverid)
        if state is None:
            state = GuildQueueState()
            self._states[serverid] = state
        return state

    def _current(self, serverid: QueueId) -> Optional[Track]:
        """Return the current track or ``None`` when the index is out of range."""
        state = self._states.get(serverid)
        if state is None or not (0 <= state.index < len(state.tracks)):
            return None
        return state.tracks[state.index]

    # ------------------------------------------------------------------ #
    #  Print cache (-queue rendering)                                    #
    # ------------------------------------------------------------------ #

    def set_qfp(self, serverid: QueueId, data) -> int:
        """Store the queue-for-print cache (legacy name kept)."""
        self._require(serverid).print_cache = list(data)
        return 0

    def get_qfp(self, serverid: QueueId) -> List[str]:
        """Return a copy of the queue-for-print cache (never raises)."""
        state = self._states.get(serverid)
        return list(state.print_cache) if state is not None else []

    # ------------------------------------------------------------------ #
    #  Queue mutation                                                    #
    # ------------------------------------------------------------------ #

    def queue_add(self, element, serverid: QueueId) -> None:
        """Append a track. Accepts a :class:`Track` or a legacy 9-element list."""
        track = element if isinstance(element, Track) else Track(*element)
        state = self._require(serverid)
        state.tracks.append(track)

    def clear(self, serverid: QueueId, isVoice: bool = False) -> None:
        """Reset queue state.

        ``isVoice=False`` -> full reset (used by ``-clear`` while idle,
        ``-disconnect`` and the AFK timeout): index, tracks, loop flag and
        jump flag are all cleared.

        ``isVoice=True``  -> "clear while playing" (used by ``-clear`` when
        something is playing): the current track is kept, everything else is
        dropped. The loop flag is intentionally *kept* in this branch (legacy
        behaviour: the kept track keeps looping).

        Bumps ``epoch`` either way so background playlist tasks notice.
        """
        state = self._require(serverid)
        state.epoch += 1
        if isVoice:
            current = self._current(serverid)
            state.tracks = [current] if current is not None else []
            state.index = 0
            state.jump_used = False
        else:
            state.index = 0
            state.tracks = []
            state.loop_flag = False
            state.jump_used = False
        state.print_cache = []

    def queue_next(self, serverid: QueueId) -> int:
        """Advance to the next track.

        Returns ``0`` when advanced (or when staying put because loop is on),
        ``1`` when the queue is exhausted. When loop is enabled the index
        intentionally does not move (legacy behaviour). ``-1 <= index`` is
        preserved so ``-jump 1`` while playing (which parks the index at -1
        before a skip) still lands on track 0.
        """
        state = self._states.get(serverid)
        if state is None:
            return 1
        if state.loop_flag:
            return 0
        if -1 <= state.index < len(state.tracks):
            state.index += 1
            return 0
        return 1

    def if_queue_exist(self, serverid: QueueId) -> bool:
        """True while the current index points at an existing track."""
        state = self._states.get(serverid)
        return bool(state is not None and state.index < len(state.tracks))

    def queue_len(self, serverid: QueueId) -> int:
        state = self._states.get(serverid)
        return len(state.tracks) if state is not None else 0

    def get_index(self, serverid: QueueId) -> int:
        state = self._states.get(serverid)
        return state.index if state is not None else 0

    def loop(self, serverid: QueueId) -> None:
        self._require(serverid).loop_flag = True

    def unloop(self, serverid: QueueId) -> None:
        self._require(serverid).loop_flag = False

    def is_looped(self, serverid: QueueId) -> bool:
        state = self._states.get(serverid)
        return bool(state is not None and state.loop_flag)

    def jump(self, value: int, serverid: QueueId) -> None:
        """Set the current index directly (legacy semantics: 0-based index)."""
        self._require(serverid).index = value

    def set_jump_used(self, serverid: QueueId, value: bool) -> None:
        self._require(serverid).jump_used = value

    def mark_current_unavailable(self, serverid: QueueId,
                                 prefix: str = UNAVAILABLE_PREFIX) -> bool:
        """Prefix the current track title with the unavailable marker (once)."""
        track = self._current(serverid)
        if track is None:
            return False
        if not track.title.startswith(prefix):
            track.title = f"{prefix}{track.title}"
        return True

    # ------------------------------------------------------------------ #
    #  Epoch (background-task lifecycle)                                 #
    # ------------------------------------------------------------------ #

    def get_epoch(self, serverid: QueueId) -> int:
        state = self._states.get(serverid)
        return state.epoch if state is not None else -1

    def bump_epoch(self, serverid: QueueId) -> None:
        self._require(serverid).epoch += 1

    # ------------------------------------------------------------------ #
    #  Track field accessors (legacy sentinel returns preserved)         #
    # ------------------------------------------------------------------ #

    def get_track_name(self, serverid: QueueId):
        track = self._current(serverid)
        return track.title if track is not None else -1

    def get_track_length(self, serverid: QueueId):
        track = self._current(serverid)
        return track.length if track is not None else -1

    def set_start_time(self, serverid: QueueId, time) -> bool:
        track = self._current(serverid)
        if track is None:
            return False
        track.start_time = time
        return True

    def get_start_time(self, serverid: QueueId):
        track = self._current(serverid)
        return track.start_time if track is not None else -1

    def set_stream_link(self, serverid: QueueId, streamlink) -> bool:
        track = self._current(serverid)
        if track is None:
            return False
        track.stream_link = streamlink
        return True

    def get_stream_link(self, serverid: QueueId):
        track = self._current(serverid)
        return track.stream_link if track is not None else False

    def get_yt_link(self, serverid: QueueId):
        track = self._current(serverid)
        return track.yt_link if track is not None else False

    def get_thumbnail_url(self, serverid: QueueId):
        track = self._current(serverid)
        return track.thumbnail_url if track is not None else False

    def get_author(self, serverid: QueueId):
        track = self._current(serverid)
        return track.author if track is not None else False

    def set_pause_time(self, serverid: QueueId, time) -> bool:
        track = self._current(serverid)
        if track is None:
            return False
        track.pause_time = time
        return True

    def get_pause_time(self, serverid: QueueId):
        track = self._current(serverid)
        return track.pause_time if track is not None else -1

    def set_no_play_time(self, serverid: QueueId, time) -> bool:
        track = self._current(serverid)
        if track is None:
            return False
        track.no_play_time = time
        return True

    def get_no_play_time(self, serverid: QueueId):
        track = self._current(serverid)
        return track.no_play_time if track is not None else -1

    # ------------------------------------------------------------------ #
    #  Text channel (AFK timeout message target)                         #
    # ------------------------------------------------------------------ #

    def set_text_channel(self, serverid: QueueId, channel) -> None:
        self._require(serverid).text_channel = channel

    def get_text_channel(self, serverid: QueueId) -> Optional[Any]:
        state = self._states.get(serverid)
        return state.text_channel if state is not None else None

    # ------------------------------------------------------------------ #
    #  Queue rendering                                                   #
    # ------------------------------------------------------------------ #

    def queue(self, serverid: QueueId, is_playing: bool = True) -> int:
        """Render the queue into the print cache.

        * ``PAGE_SIZE`` tracks per page, numbering continuous from 1.
        * No empty trailing page when the length is an exact multiple of
          ``PAGE_SIZE`` (legacy ``length // 20 + 1`` bug fixed).
        * The current track (and loop state) is marked on a *display copy* --
          the stored title is never mutated (legacy temporary-mutation race
          fixed).
        * Returns ``-1`` when the guild has no queue content (legacy
          contract), else ``0``. Never raises for unknown guilds.
        """
        state = self._states.get(serverid)
        if state is None or state.is_pristine():
            return -1

        lines: List[str] = []
        for n, track in enumerate(state.tracks):
            title = track.title
            if is_playing and n == state.index:
                suffix = f" \t\t — {QUEUE_CURRENT}"
                if state.loop_flag:
                    suffix += f" ({QUEUE_LOOPED})"
                title = f"{title}{suffix}"
            lines.append(f"{n + 1}) {title}\n")

        total_pages = max(1, (len(state.tracks) + PAGE_SIZE - 1) // PAGE_SIZE)
        cache: List[str] = []
        for page in range(total_pages):
            chunk = "".join(lines[page * PAGE_SIZE:(page + 1) * PAGE_SIZE])
            cache.append(f"```{chunk}```")

        state.print_cache = cache
        return 0
