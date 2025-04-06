from __future__ import annotations

from pathlib import Path
from typing import Final, Dict

DATA_DIR: Final[Path] = Path("../data")
RUNS_DIR: Final[Path] = Path("../runs")
MIDI_FOLDER: Final[Path] = DATA_DIR / "midi"
VOCAB_PATH: Final[Path] = RUNS_DIR / "data" / "vocab.pkl"
CACHE_DIR: Final[Path] = RUNS_DIR / "data" / "cache"
MODEL_SAVE_PATH: Final[Path] = RUNS_DIR / "checkpoints"
TENSORBOARD_LOGDIR: Final[Path] = RUNS_DIR / "logs"
EPOCHS: Final[int] = 200
BATCH_SIZE: Final[int] = 2
SEQ_LEN = 512
MAX_TRACKS = 4
LR: Final[float] = 3e-5
Z_NOISE_SCALE: Final[float] = 1.0
RANDOM_SEED: Final[int] = 42
WEIGHT_DECAY_G: Final[float] = 1e-4
WEIGHT_DECAY_D: Final[float] = 1e-4
WEIGHT_DECAY_R: Final[float] = 1e-4
D_LR_MULTIPLIER: Final[float] = 1.0
EARLY_STOPPING_PATIENCE: Final[int] = 20
DROPOUT: Final[float] = 0.2
BETA: Final[float] = 0.1
EMBED_DIM: Final[int] = 8
DRUM_EMBED_DIM: Final[int] = 8
Z_DIM: Final[int] = 8
SEGMENT_DIM: Final[int] = 8
CLIP_NORM = 1.0
MAX_MEASURES: Final[int] = 120
MEASURE_LENGTH: Final[int] = 480
TICKS_PER_STEP: Final[int] = 120
TIME_STEP: Final[float] = 0.25
SEGMENT_MAP: Final[list[int]] = [0, 1, 2, 3, 4]
NUM_SEGMENTS: Final[int] = len(SEGMENT_MAP)
TRANSPOSE_RANGE: Final[int] = 2
VELOCITY_RANGE: Final[int] = 10
MASK_PROBABILITY: Final[float] = 0.15
VELOCITY_THRESHOLDS: Final[list[int]] = [32, 64, 96]
DURATION_THRESHOLDS: Final[list[int]] = [
    30, 60, 90, 120, 180, 240, 360, 480, 720, 960, 1440, 1920
]
_SEMITONE_LIMITS: Final[tuple[int, int]] = (0, 127)
_VELOCITY_LIMITS: Final[tuple[int, int]] = (0, 127)
_TRANSPOSE_PADDING: Final[int] = 2
_VELOCITY_PADDING: Final[int] = 5
UNISON_THRESHOLD: Final[float] = 0.9
MIN_NOTES_THRESHOLD: Final[int] = 5
GROUP_NORM_GROUPS: Final[int] = 8
TOP_K_DEFAULT = 0
TOP_P_DEFAULT = 0.95
RD_HIDDEN: Final[int] = 32
RD_DROPOUT: Final[float] = 0.3
NUM_WORKERS: Final[int] = 0
REST_TOKEN: Final[str] = "REST"
GM_PROGRAMS: Final[dict[int, str]] = {
    0: "Acoustic Grand Piano",
    1: "Bright Acoustic Piano",
    2: "Electric Grand Piano",
    3: "Honky-tonk Piano",
    4: "Electric Piano 1",
    5: "Electric Piano 2",
    6: "Harpsichord",
    7: "Clavinet",
    8: "Celesta",
    9: "Glockenspiel",
    10: "Music Box",
    11: "Vibraphone",
    12: "Marimba",
    13: "Xylophone",
    14: "Tubular Bells",
    15: "Dulcimer",
    16: "Drawbar Organ",
    17: "Percussive Organ",
    18: "Rock Organ",
    19: "Church Organ",
    20: "Reed Organ",
    21: "Accordion",
    22: "Harmonica",
    23: "Tango Accordion",
    24: "Acoustic Guitar (nylon)",
    25: "Acoustic Guitar (steel)",
    26: "Electric Guitar (jazz)",
    27: "Electric Guitar (clean)",
    28: "Electric Guitar (muted)",
    29: "Overdriven Guitar",
    30: "Distortion Guitar",
    31: "Guitar Harmonics",
    32: "Acoustic Bass",
    33: "Electric Bass (finger)",
    34: "Electric Bass (pick)",
    35: "Fretless Bass",
    36: "Slap Bass 1",
    37: "Slap Bass 2",
    38: "Synth Bass 1",
    39: "Synth Bass 2",
    40: "Violin",
    41: "Viola",
    42: "Cello",
    43: "Contrabass",
    44: "Tremolo Strings",
    45: "Pizzicato Strings",
    46: "Orchestral Harp",
    47: "Timpani",
    48: "String Ensemble 1",
    49: "String Ensemble 2",
    50: "Synth Strings 1",
    51: "Synth Strings 2",
    52: "Choir Aahs",
    53: "Voice Oohs",
    54: "Synth Choir",
    55: "Orchestra Hit",
    56: "Trumpet",
    57: "Trombone",
    58: "Tuba",
    59: "Muted Trumpet",
    60: "French Horn",
    61: "Brass Section",
    62: "Synth Brass 1",
    63: "Synth Brass 2",
    64: "Soprano Sax",
    65: "Alto Sax",
    66: "Tenor Sax",
    67: "Baritone Sax",
    68: "Oboe",
    69: "English Horn",
    70: "Bassoon",
    71: "Clarinet",
    72: "Piccolo",
    73: "Flute",
    74: "Recorder",
    75: "Pan Flute",
    76: "Blown Bottle",
    77: "Shakuhachi",
    78: "Whistle",
    79: "Ocarina",
    80: "Lead 1 (square)",
    81: "Lead 2 (sawtooth)",
    82: "Lead 3 (calliope)",
    83: "Lead 4 (chiff)",
    84: "Lead 5 (charang)",
    85: "Lead 6 (voice)",
    86: "Lead 7 (fifths)",
    87: "Lead 8 (bass + lead)",
    88: "Pad 1 (new age)",
    89: "Pad 2 (warm)",
    90: "Pad 3 (polysynth)",
    91: "Pad 4 (choir)",
    92: "Pad 5 (bowed)",
    93: "Pad 6 (metallic)",
    94: "Pad 7 (halo)",
    95: "Pad 8 (sweep)",
    96: "FX 1 (rain)",
    97: "FX 2 (soundtrack)",
    98: "FX 3 (crystal)",
    99: "FX 4 (atmosphere)",
    100: "FX 5 (brightness)",
    101: "FX 6 (goblins)",
    102: "FX 7 (echoes)",
    103: "FX 8 (sci-fi)",
    104: "Sitar",
    105: "Banjo",
    106: "Shamisen",
    107: "Koto",
    108: "Kalimba",
    109: "Bag pipe",
    110: "Fiddle",
    111: "Shanai",
    112: "Tinkle Bell",
    113: "Agogo",
    114: "Steel Drums",
    115: "Woodblock",
    116: "Taiko Drum",
    117: "Melodic Tom",
    118: "Synth Drum",
    119: "Reverse Cymbal",
    120: "Guitar Fret Noise",
    121: "Breath Noise",
    122: "Seashore",
    123: "Bird Tweet",
    124: "Telephone Ring",
    125: "Helicopter",
    126: "Applause",
    127: "Gunshot",
    128: "Drums"
}

MODES: Final[dict[str, list[int]]] = {
    "ionian": [0, 2, 4, 5, 7, 9, 11],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "aeolian": [0, 2, 3, 5, 7, 8, 10],
    "locrian": [0, 1, 3, 5, 6, 8, 10],
}
INSTRUMENT_LOSS_DRUM_WEIGHT: Final[float] = 2.0
INSTRUMENT_LOSS_DRUM_IDX: Final[int] = 128
LOSS_WEIGHT_KL = 0.1
LOSS_WEIGHT_HARMONIC = 0.05
LOSS_WEIGHT_DIVERSITY = 0.05
LOSS_WEIGHT_REGULARIZATION = 0.05
LOSS_WEIGHT_INSTRUMENT: Final[float] = 0.1
LOSS_WEIGHT_RHYTHM: Final[float] = 0.05
duration_map: Final[Dict[int, int]] = {
    0: 60, 1: 120, 2: 240, 3: 480,
    4: 960, 5: 1920
}
velocity_map: Final[Dict[int, int]] = {
    0: 32, 1: 64, 2: 96, 3: 127
}
