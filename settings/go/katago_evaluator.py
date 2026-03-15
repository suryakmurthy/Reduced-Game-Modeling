import subprocess
import time
import json
import os


MODEL_PATH = "/Users/edamame/Desktop/UT Austin/Autonomous Systems Group/chess_rogm/go/weights/kata1-b18c384nbt-s5810472448-d3214865830.bin.gz"


class KataGoEvaluator:
    """
    Persistent KataGo process for position evaluation.

    Maintains an in-memory cache of position evaluations.
    Checkpoints cache to disk periodically.

    Evaluation convention:
        f(sequence) = P(Black wins) in [0, 1]
        where sequence = (move_0, move_1, ...) alternating B/W from move_0=B
    """

    def __init__(self, cache_file="go_cache.json", checkpoint_every=1000):
        self.cache_file = cache_file
        self.checkpoint_every = checkpoint_every
        self.cache = {}
        self.eval_count = 0
        self.proc = None
        self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file) as f:
                raw = json.load(f)
            # JSON keys are strings — convert back to tuples
            self.cache = {tuple(k.split("|")): v for k, v in raw.items()}
            print(f"  Loaded {len(self.cache)} cached positions from {self.cache_file}")
        else:
            self.cache = {}

    def _save_cache(self):
        serializable = {"|".join(k): v for k, v in self.cache.items()}
        with open(self.cache_file, "w") as f:
            json.dump(serializable, f)

    def start(self):
        """Start KataGo process and warm up."""
        print("Starting KataGo...")
        self.proc = subprocess.Popen(
            ["katago", "gtp", "-model", MODEL_PATH,
             "-override-config", "maxVisits=1"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        time.sleep(3)
        if self.proc.poll() is not None:
            raise RuntimeError(f"KataGo failed to start:\n{self.proc.stderr.read()}")

        self._send("boardsize 9")
        self._send("komi 6.5")

        # Warmup evaluation
        print("  Warming up...")
        self._send("clear_board")
        self._query_winrate()
        print("  KataGo ready.")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.stdin.write("quit\n")
            self.proc.stdin.flush()
        self._save_cache()
        print(f"  Cache saved ({len(self.cache)} positions).")

    def _send(self, cmd):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()
        lines = []
        while True:
            line = self.proc.stdout.readline().strip()
            if line == "":
                break
            lines.append(line)
        return lines

    def _set_position(self, move_sequence):
        """Replay move sequence from empty board."""
        self._send("clear_board")
        for i, move in enumerate(move_sequence):
            color = "B" if i % 2 == 0 else "W"
            self._send(f"play {color} {move}")

    def _query_winrate(self):
        """
        Query KataGo for Black win probability at current position.
        kata-raw-nn gives whiteLoss = P(Black wins) directly.
        """
        self.proc.stdin.write("kata-raw-nn 0\n")
        self.proc.stdin.flush()

        black_winrate = None
        while True:
            line = self.proc.stdout.readline().strip()
            if line.startswith("whiteLoss"):
                black_winrate = float(line.split()[1])
            if line == "":
                break

        return black_winrate

    def evaluate(self, move_sequence):
        """
        Evaluate a position given a move sequence.

        Parameters
        ----------
        move_sequence : tuple of str
            GTP moves alternating B/W, starting with Black.
            e.g. ('D5', 'F6', 'C3', 'G3')

        Returns
        -------
        float in [0, 1] : P(Black wins), or None if evaluation failed
        """
        key = tuple(move_sequence)

        if key in self.cache:
            return self.cache[key]

        self._set_position(move_sequence)
        winrate = self._query_winrate()

        if winrate is not None:
            self.cache[key] = winrate
            self.eval_count += 1
            if self.eval_count % self.checkpoint_every == 0:
                self._save_cache()

        return winrate


if __name__ == "__main__":
    evaluator = KataGoEvaluator()
    evaluator.start()

    # Test a few positions
    test_lines = [
        (),                           # empty board
        ("D5",),                      # Black tengen
        ("D5", "F6"),                 # Black tengen, White approach
        ("D5", "F6", "C3", "G3"),     # 4-move line
    ]

    print("\nTest evaluations:")
    for line in test_lines:
        wr = evaluator.evaluate(line)
        moves_str = " ".join(line) if line else "(empty)"
        print(f"  {moves_str:<30} Black winrate = {wr:.4f}")

    evaluator.stop()
