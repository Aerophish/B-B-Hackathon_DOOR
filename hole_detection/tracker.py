"""Replaceable frame-to-frame tracker used to calculate temporal score T."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .config import DetectorConfig
from .models import Hole


@dataclass
class _Track:
    """Internal short-term memory; this is not a permanent mapped landmark."""

    track_id: int
    center: tuple[float, float]
    # Consecutive observations only. Any missed frame resets this streak.
    hits: int = 1
    misses: int = 0  # Consecutive frames without a match.


class HoleTracker:
    """Calculate T using nearest-centre association between consecutive frames.

    This is appropriate for a stationary/slow-camera prototype. Robot motion shifts
    all image positions, so later replace or assist this with optical flow, odometry,
    SLAM/world coordinates, or a Kalman filter. The Hole output interface can remain
    unchanged, protecting the navigation and memory modules from that replacement.
    """

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self._tracks: Dict[int, _Track] = {}
        self._next_id = 1

    def update(self, holes: Iterable[Hole]) -> List[Hole]:
        """Match current candidates and write identity/T values onto each Hole."""
        holes = list(holes)

        # A track may be assigned at most once during the current frame.
        unmatched = set(self._tracks)
        for hole in holes:
            track = self._nearest_available(hole.center, unmatched)
            if track is None:
                # First observation: create a new, not-yet-confirmed identity.
                track = _Track(self._next_id, hole.center)
                self._tracks[track.track_id] = track
                self._next_id += 1
            else:
                # Re-observation: update last image position and observation count.
                unmatched.remove(track.track_id)
                track.center = hole.center
                # A track retained through a miss keeps its ID, but its confirmation
                # streak starts again. This prevents intermittent false candidates
                # from gradually accumulating 60 non-consecutive observations.
                track.hits = 1 if track.misses > 0 else track.hits + 1
                track.misses = 0

            hole.track_id = track.track_id
            # With confirmation_frames=60, consecutive hits produce T=1/60 ... 1.
            hole.temporal_stability = min(1.0, track.hits / self.config.confirmation_frames)
            hole.confirmed = track.hits >= self.config.confirmation_frames

        # Age unmatched tracks instead of immediately forgetting them. This lets an
        # identity survive brief occlusion, motion blur, or a one-frame detector miss.
        for track_id in unmatched:
            self._tracks[track_id].misses += 1
            # Persistence must be consecutive, so a miss invalidates the streak even
            # though we retain the identity briefly for association purposes.
            self._tracks[track_id].hits = 0
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if track.misses <= self.config.max_track_misses
        }
        return holes

    def _nearest_available(
        self,
        center: tuple[float, float],
        allowed: set[int],
    ) -> Optional[_Track]:
        """Return the closest unclaimed track within association_distance_px."""
        candidates = []
        for track_id in allowed:
            track = self._tracks[track_id]
            distance = (
                (center[0] - track.center[0]) ** 2
                + (center[1] - track.center[1]) ** 2
            ) ** 0.5
            if distance <= self.config.association_distance_px:
                candidates.append((distance, track))

        # No allowed nearby track means this observation receives a new identity.
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]
