import chess
import chess.engine
import json
import os
from typing import Optional, List


class LC0Evaluator:
    """
    Wraps LC0 via python-chess UCI interface.

    Usage
    -----
        with LC0Evaluator(nodes=1) as ev:
            score = ev.evaluate_line(["e2e4", "c7c5", "g1f3"])
    """

    def __init__(
        self,
        engine_path: str = "lc0",
        weights_path: Optional[str] = None,
        nodes: int = 1,
        checkpoint_file: Optional[str] = None,
        checkpoint_every: int = 10000,
    ):
        """
        Parameters
        ----------
        engine_path      : path or name of the lc0 binary
        weights_path     : optional explicit path to .pb.gz weights file;
                           if None, LC0 uses its autodiscovery
        nodes            : number of MCTS nodes per evaluation.
                           nodes=1  -> raw network eval, fast, deterministic
                           nodes=10 -> slightly more accurate, slower
        checkpoint_file  : if set, cache is saved to this JSON file every
                           `checkpoint_every` new evaluations
        checkpoint_every : how often to write the cache to disk
        """
        self.nodes = nodes
        self.checkpoint_file = checkpoint_file
        self.checkpoint_every = checkpoint_every
        self._cache = {}
        self._evals_since_checkpoint = 0

        # Load existing cache from checkpoint if available
        if checkpoint_file and os.path.exists(checkpoint_file):
            with open(checkpoint_file) as f:
                self._cache = json.load(f)
            print(f"  Loaded {len(self._cache)} cached positions from {checkpoint_file}")

        print(f"Starting LC0 (nodes={nodes})...")
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path, setpgrp=True)
        self.engine.configure({"UCI_ShowWDL": "true"})
        if weights_path:
            self.engine.configure({"WeightsFile": weights_path})

        self._warmup()
        print("LC0 ready.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _warmup(self):
        """Warm up the Metal backend with a dummy evaluation."""
        print("  Warming up Metal backend...")
        self.engine.analyse(chess.Board(), chess.engine.Limit(nodes=self.nodes))
        print("  Warmup complete.")

    def _save_checkpoint(self):
        if self.checkpoint_file:
            with open(self.checkpoint_file, "w") as f:
                json.dump(self._cache, f)

    def _evaluate_board(self, board: chess.Board) -> float:
        """Evaluate a board position, using cache."""
        fen = board.fen()
        if fen in self._cache:
            return self._cache[fen]

        info = self.engine.analyse(board, chess.engine.Limit(nodes=self.nodes))
        wdl = info["score"].white().wdl()
        score = (wdl.wins + 0.5 * wdl.draws) / 1000.0

        self._cache[fen] = score
        self._evals_since_checkpoint += 1

        if self._evals_since_checkpoint >= self.checkpoint_every:
            self._save_checkpoint()
            self._evals_since_checkpoint = 0
            print(f"  [Checkpoint] Saved {len(self._cache)} positions to {self.checkpoint_file}")

        return score

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_line(self, moves: List[str]) -> Optional[float]:
        """
        Play out a list of UCI moves from the starting position and evaluate.

        Returns White's win probability in [0, 1], or None if any move is illegal.
        """
        board = chess.Board()
        for move_str in moves:
            try:
                move = chess.Move.from_uci(move_str)
                if move not in board.legal_moves:
                    return None
                board.push(move)
            except Exception:
                return None
        return self._evaluate_board(board)

    def cache_size(self) -> int:
        return len(self._cache)

    def quit(self):
        self._save_checkpoint()
        self.engine.quit()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.quit()


# ------------------------------------------------------------------
# Quick test
# ------------------------------------------------------------------
if __name__ == "__main__":
    with LC0Evaluator(nodes=1) as ev:
        # Sicilian Najdorf
        najdorf = ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6"]
        score = ev.evaluate_line(najdorf)
        print(f"Sicilian Najdorf  -- White win prob: {score:.4f}")

        # Starting position
        score = ev.evaluate_line([])
        print(f"Starting position -- White win prob: {score:.4f}")

        # Illegal line
        score = ev.evaluate_line(["e2e4", "e2e4"])
        print(f"Illegal line      -- result: {score}")
