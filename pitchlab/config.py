"""Central settings. Override with environment variables when deployed."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("PITCHLAB_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Distance from the pitching rubber to the front of home plate, in feet.
MOUND_DISTANCE_FT = 60.5

# Minimum pitches of a given type before a pitcher appears on leaderboards.
MIN_PITCHES = int(os.environ.get("PITCHLAB_MIN_PITCHES", 50))

# Shrinkage constant: a pitch type thrown K times gets half its raw residual.
# Keeps tiny samples from topping the outlier board.
SHRINK_K = int(os.environ.get("PITCHLAB_SHRINK_K", 75))

# Statcast pitch codes grouped into families that share a movement profile.
# The expected-movement model is fit separately for each family.
PITCH_FAMILIES = {
    "FF": "Four-Seam",
    "SI": "Sinker",
    "FC": "Cutter",
    "SL": "Slider",
    "ST": "Sweeper",
    "CU": "Curveball",
    "KC": "Curveball",
    "SV": "Curveball",
    "CH": "Changeup",
    "FS": "Splitter",
    "FO": "Splitter",
}

# Columns kept from the Savant CSV. Everything else is dropped on ingest.
STATCAST_COLUMNS = [
    "game_date", "game_year", "pitcher", "player_name", "p_throws",
    "pitch_type", "release_speed", "effective_speed", "release_spin_rate",
    "spin_axis", "release_extension", "release_pos_x", "release_pos_y",
    "release_pos_z", "pfx_x", "pfx_z", "plate_x", "plate_z", "arm_angle",
    "vx0", "vy0", "vz0", "ax", "ay", "az", "description", "game_type",
    # outcomes, used by the Stuff+ model and the "does it matter" analysis
    "delta_run_exp", "type", "events", "bb_type", "estimated_woba_using_speedangle",
    "launch_speed", "launch_angle", "balls", "strikes", "stand", "zone",
    "sz_top", "sz_bot", "game_pk", "at_bat_number", "pitch_number",
]

# If cached raw data lacks this column it predates the outcome columns and is
# re-downloaded in full.
REQUIRED_RAW_COLUMN = "delta_run_exp"

for d in (RAW_DIR, PROCESSED_DIR):
    d.mkdir(parents=True, exist_ok=True)
