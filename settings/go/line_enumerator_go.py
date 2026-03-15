import subprocess
import time

MODEL_PATH = "/Users/edamame/Desktop/UT Austin/Autonomous Systems Group/chess_rogm/go/weights/kata1-b18c384nbt-s5810472448-d3214865830.bin.gz"

# GTP column labels (I is skipped in Go)
COLS = "ABCDEFGHJKLMNOPQRST"


def coord_to_gtp(row, col, board_size=9):
    """Convert (row, col) 0-indexed from top-left to GTP move string."""
    return f"{COLS[col]}{board_size - row}"


def parse_policy(lines, board_size=9):
    """
    Parse kata-raw-nn output and return list of (gtp_move, prior) tuples.
    NAN entries are occupied squares — skip them.
    """
    moves = []
    in_policy = False
    row = 0
    for line in lines:
        if line.strip() == "policy":
            in_policy = True
            continue
        if in_policy:
            if line.startswith("policyPass"):
                break
            if row >= board_size:
                break
            values = line.strip().split()
            for col, val in enumerate(values):
                if val != "NAN":
                    prior = float(val)
                    gtp = coord_to_gtp(row, col, board_size)
                    moves.append((gtp, prior))
            row += 1
    return moves


class GoLineEnumerator:
    """
    Enumerates opening lines for 9x9 Go using KataGo's policy network.

    A line is a tuple of GTP moves: ('D5', 'F6', 'C3', ...)
    alternating Black (even indices) and White (odd indices).

    Uses undo to traverse the tree without replaying from scratch.
    Line count is exactly top_k^depth.
    """

    def __init__(self, depth, top_k=3, board_size=9):
        self.depth = depth
        self.top_k = top_k
        self.board_size = board_size
        self.proc = None

    def _start(self):
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
            raise RuntimeError(f"KataGo failed to start: {self.proc.stderr.read()}")
        self._send("boardsize 9")
        self._send("komi 6.5")

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

    def _get_policy(self):
        """Query raw policy at current board position."""
        self.proc.stdin.write("kata-raw-nn 0\n")
        self.proc.stdin.flush()
        lines = []
        while True:
            line = self.proc.stdout.readline().strip()
            if line == "":
                break
            lines.append(line)
        return parse_policy(lines, self.board_size)

    def enumerate(self):
        """
        DFS through the opening tree using undo for efficiency.
        Returns list of tuples, each of length self.depth.
        Count is exactly top_k^depth.
        """
        self._start()
        self._send("clear_board")
        results = []
        self._dfs([], results)
        self.proc.stdin.write("quit\n")
        self.proc.stdin.flush()
        return results

    def _dfs(self, current_line, results):
        if len(current_line) == self.depth:
            results.append(tuple(current_line))
            return

        policy = self._get_policy()

        # Take top-k moves by prior — no threshold, fixed branching factor
        candidates = sorted(policy, key=lambda x: -x[1])[:self.top_k]

        for move, prior in candidates:
            color = "B" if len(current_line) % 2 == 0 else "W"
            self._send(f"play {color} {move}")
            self._dfs(current_line + [move], results)
            self._send("undo")


if __name__ == "__main__":
    print("Go Line Enumerator — sizing test")
    print(f"{'depth':<8} {'top_k':<8} {'lines':<8} {'pairs':<14} {'expected'}")
    print("-" * 50)

    for depth in [4, 6, 8]:
        for top_k in [2, 3, 4]:
            enumerator = GoLineEnumerator(depth=depth, top_k=top_k)
            lines = enumerator.enumerate()
            n = len(lines)
            expected = top_k ** depth
            print(f"{depth:<8} {top_k:<8} {n:<8} {n*n:<14,} {expected}")
