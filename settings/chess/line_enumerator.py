import chess
import chess.polyglot
from typing import List, Tuple


class PolyglotLineEnumerator:
    """
    Enumerate all opening lines up to a fixed depth from a Polyglot book.

    Parameters
    ----------
    book_path  : path to a .bin Polyglot opening book (e.g. gm2001.bin)
    min_weight : minimum move weight to include; filters out rare moves.
                 Higher = fewer, more popular lines.
                 Lower  = more lines including rare sidelines.
    """

    def __init__(self, book_path: str, min_weight: int = 1):
        self.book_path = book_path
        self.min_weight = min_weight

    def enumerate(self, depth: int) -> List[Tuple[str, ...]]:
        """
        Enumerate all lines to exactly `depth` plies.

        Parameters
        ----------
        depth : number of half-moves per line
                depth=6  -> 3 moves each side
                depth=8  -> 4 moves each side
                depth=10 -> 5 moves each side

        Returns
        -------
        List of lines, each a tuple of UCI move strings of length `depth`.
        """
        lines = []

        def dfs(board, current_line, d):
            if d == depth:
                lines.append(tuple(current_line))
                return

            with chess.polyglot.open_reader(self.book_path) as reader:
                entries = [
                    e for e in reader.find_all(board)
                    if e.weight >= self.min_weight
                ]

            for entry in entries:
                board.push(entry.move)
                current_line.append(entry.move.uci())
                dfs(board, current_line, d + 1)
                current_line.pop()
                board.pop()

        print(f"Enumerating lines to depth {depth} (min_weight={self.min_weight}, book={self.book_path})...")
        dfs(chess.Board(), [], 0)
        print(f"  Found {len(lines)} lines.")
        return lines


# ------------------------------------------------------------------
# Quick test
# ------------------------------------------------------------------
if __name__ == "__main__":
    enumerator = PolyglotLineEnumerator("gm2001.bin", min_weight=10)
    lines = enumerator.enumerate(depth=6)
    print(f"\nSample lines:")
    for line in lines[:5]:
        print(" ", " ".join(line))
