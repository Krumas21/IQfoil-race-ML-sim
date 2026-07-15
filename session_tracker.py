"""IQfoil Session Tracker — part of The 35 to 24 Project.

Logs iQFoil training/race sessions to a local JSON file and surfaces
patterns: average wind, sessions per month, which equipment setup
scored best. Later feeds a small PyTorch model (M8, not built yet).

Usage:
    python session_tracker.py add --date 2026-07-14 --wind 18 --board 95 \
        --sail 8.0 --location "Kaunas Lagoon" --minutes 90 --type training
    python session_tracker.py list
    python session_tracker.py filter --min-wind 15 --max-wind 25
    python session_tracker.py filter --type race
    python session_tracker.py stats
"""

import argparse
import json
from pathlib import Path

import numpy as np

# Data lives next to this script, not in whatever folder you run it from.
# __file__ = path of this .py file; .parent = its folder.
DATA_FILE = Path(__file__).parent / "sessions.json"


# ---------------------------------------------------------------- M4: class
class Session:
    """One training or race session.

    Why a class instead of a dict? Same reason PyTorch uses nn.Module:
    a class bundles DATA (self.wind_speed_knots) with BEHAVIOR
    (self.describe()) in one object. In PyTorch you subclass nn.Module,
    store layers in __init__ (data) and define forward() (behavior) —
    identical pattern, just with tensors instead of wind speeds.
    """

    def __init__(self, date, wind_speed_knots, board_size, sail_size,
                 location, duration_minutes, session_type,
                 result=None, notes=""):
        # __init__ runs once when you do Session(...). "self" is the new
        # object being built; each assignment attaches a field to it.
        self.date = date                        # "YYYY-MM-DD" string
        self.wind_speed_knots = wind_speed_knots
        self.board_size = board_size            # liters
        self.sail_size = sail_size              # m²
        self.location = location
        self.duration_minutes = duration_minutes
        self.session_type = session_type        # "training" | "race"
        self.result = result                    # race place (int) or None
        self.notes = notes

    def to_dict(self):
        """Convert to a plain dict so json.dump can store it.

        vars(self) returns the object's fields as a dict — the reverse
        of what __init__ built.
        """
        return vars(self)

    @classmethod
    def from_dict(cls, data):
        """Build a Session from a dict loaded out of JSON.

        @classmethod means this is called on the CLASS, not an instance:
        Session.from_dict({...}). "cls" is the class itself.
        **data unpacks the dict into keyword arguments, so
        cls(**data) == Session(date=..., wind_speed_knots=..., ...).
        """
        return cls(**data)

    def describe(self):
        """Return a readable multi-line summary (M1 f-string print)."""
        # Inline if-expression: None result (training day) prints as "—".
        result_text = f"P{self.result}" if self.result else "—"
        return (
            f"{self.date} | {self.location}\n"
            f"  Wind: {self.wind_speed_knots}kt | "
            f"Board: {self.board_size}L | Sail: {self.sail_size}m²\n"
            f"  {self.session_type} | {self.duration_minutes}min | "
            f"Result: {result_text}\n"
            f"  Notes: {self.notes}"
        )


# --------------------------------------------------------- M5: persistence
def load_sessions():
    """Load all sessions from the JSON file. Empty list if none yet."""
    if not DATA_FILE.exists():
        return []
    with open(DATA_FILE) as f:
        raw = json.load(f)  # raw = list of dicts
    # List comprehension: build a new list by applying an expression to
    # every item. Same as a for-loop with .append(), one line.
    return [Session.from_dict(d) for d in raw]


def save_sessions(sessions):
    """Write all sessions back to the JSON file."""
    with open(DATA_FILE, "w") as f:
        json.dump([s.to_dict() for s in sessions], f, indent=2)


# ---------------------------------------------------- M2: add / list
def add_session(session):
    """Append one session and persist."""
    sessions = load_sessions()
    sessions.append(session)
    save_sessions(sessions)
    print(f"Saved. {len(sessions)} sessions total.")


def list_sessions(sessions):
    """Print every session, oldest first."""
    if not sessions:
        print("No sessions logged yet.")
        return
    # sorted() with key=: sort by whatever the lambda returns per item.
    # ISO dates ("2026-07-14") sort correctly as plain strings.
    for s in sorted(sessions, key=lambda s: s.date):
        print(s.describe())
        print()


# ------------------------------------------------------------- M3: filters
def filter_by_wind(sessions, min_kt, max_kt):
    """Sessions with wind inside [min_kt, max_kt]."""
    # Comprehension with a condition: keep only items where the
    # "if" part is true. Chained comparison a <= x <= b is valid Python.
    return [s for s in sessions
            if min_kt <= s.wind_speed_knots <= max_kt]


def filter_by_type(sessions, session_type):
    """Sessions of one type: 'training' or 'race'."""
    return [s for s in sessions if s.session_type == session_type]


# --------------------------------------------------------- M6: numpy stats
def show_stats(sessions):
    """Averages, monthly counts, and best-scoring conditions."""
    if not sessions:
        print("No sessions — nothing to analyze.")
        return

    # np.array turns a Python list into a vector. After that, math runs
    # on ALL elements at once (vectorized) — no loop needed. This is the
    # core habit for PyTorch: tensors work exactly like this.
    winds = np.array([s.wind_speed_knots for s in sessions])
    minutes = np.array([s.duration_minutes for s in sessions])

    print(f"Sessions logged: {len(sessions)}")
    print(f"Wind: avg {winds.mean():.1f}kt, "
          f"min {winds.min()}kt, max {winds.max()}kt")
    print(f"Time on water: {minutes.sum() / 60:.1f}h total, "
          f"avg {minutes.mean():.0f}min/session")

    # Sessions per month: slice "2026-07" out of each date string, then
    # np.unique with return_counts=True gives (unique values, how many
    # of each) — a one-line group-by.
    months = np.array([s.date[:7] for s in sessions])
    unique_months, counts = np.unique(months, return_counts=True)
    print("\nSessions per month:")
    # zip() walks two sequences in parallel: (month, count) pairs.
    for month, count in zip(unique_months, counts):
        print(f"  {month}: {count}")

    # Best conditions: among races with a result, lowest place wins.
    races = [s for s in sessions
             if s.session_type == "race" and s.result is not None]
    if not races:
        print("\nNo race results yet — log races to see best conditions.")
        return

    results = np.array([s.result for s in races])
    # argmin returns the INDEX of the smallest value (place 1 < place 5),
    # so races[best] is the session object behind that best result.
    best = races[int(results.argmin())]
    print(f"\nBest result: P{best.result} on {best.date} at {best.location}")
    print(f"  Conditions: {best.wind_speed_knots}kt | "
          f"{best.board_size}L board | {best.sail_size}m² sail")


# ---------------------------------------------------------------- M7: CLI
def build_parser():
    """Define the command-line interface.

    argparse pattern: one main parser, then sub-parsers per command
    (like git: `git add`, `git log`). Each add_argument defines one flag.
    """
    parser = argparse.ArgumentParser(
        description="iQFoil session tracker — The 35 to 24 Project")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", help="log a new session")
    add.add_argument("--date", required=True, help="YYYY-MM-DD")
    # type=int makes argparse convert the string "18" to 18 for you.
    add.add_argument("--wind", required=True, type=int,
                     help="wind speed in knots")
    add.add_argument("--board", required=True, type=int,
                     help="board size in liters")
    add.add_argument("--sail", required=True, type=float,
                     help="sail size in m²")
    add.add_argument("--location", required=True)
    add.add_argument("--minutes", required=True, type=int,
                     help="session duration")
    add.add_argument("--type", required=True,
                     choices=["training", "race"], dest="session_type")
    add.add_argument("--result", type=int, default=None,
                     help="race placing (omit for training)")
    add.add_argument("--notes", default="")

    sub.add_parser("list", help="show all sessions")

    filt = sub.add_parser("filter", help="filter sessions")
    filt.add_argument("--min-wind", type=int, default=0)
    filt.add_argument("--max-wind", type=int, default=99)
    filt.add_argument("--type", choices=["training", "race"],
                      dest="session_type", default=None)

    sub.add_parser("stats", help="numpy stats across all sessions")

    return parser


def main():
    args = build_parser().parse_args()

    if args.command == "add":
        add_session(Session(
            date=args.date,
            wind_speed_knots=args.wind,
            board_size=args.board,
            sail_size=args.sail,
            location=args.location,
            duration_minutes=args.minutes,
            session_type=args.session_type,
            result=args.result,
            notes=args.notes,
        ))
    elif args.command == "list":
        list_sessions(load_sessions())
    elif args.command == "filter":
        sessions = filter_by_wind(load_sessions(),
                                  args.min_wind, args.max_wind)
        if args.session_type:
            sessions = filter_by_type(sessions, args.session_type)
        list_sessions(sessions)
    elif args.command == "stats":
        show_stats(load_sessions())


# Runs main() only when executed directly (python session_tracker.py),
# not when imported by another file — e.g. the future PyTorch script.
if __name__ == "__main__":
    main()
