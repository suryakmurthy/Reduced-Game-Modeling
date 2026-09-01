"""
matrix_construction.py  (HAND-CONDITIONED VERSION)

Now constructs a payoff matrix where strategies are conditioned on
private hole cards:

  A[(h1, s1), (h2, s2)] = payoff when P1 holds h1 and plays sequence s1,
                          and P2 holds h2 and plays sequence s2.

This removes the symmetry issue and produces meaningful poker structure.
"""

import numpy as np
from itertools import combinations
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

from scipy.sparse import lil_matrix


# =============================================================================
# 1. CARD REPRESENTATION
# =============================================================================

RANKS = "23456789TJQKA"
SUITS = "cdhs"


def parse_card(s: str) -> int:
    return RANKS.index(s[0].upper()) * 4 + SUITS.index(s[1].lower())


def card_str(c: int) -> str:
    return RANKS[c // 4] + SUITS[c % 4]


# =============================================================================
# 2. HAND EVALUATION
# =============================================================================

def evaluate(hole: Tuple[int, int], board: List[int]) -> tuple:
    return max(_score5(combo) for combo in combinations(list(hole) + board, 5))


def _score5(cards) -> tuple:
    ranks = sorted([c // 4 for c in cards], reverse=True)
    suits = [c % 4 for c in cards]

    is_flush = (len(set(suits)) == 1)

    unique = sorted(set(ranks), reverse=True)
    is_straight, s_high = False, None
    if len(unique) == 5:
        if unique[0] - unique[4] == 4:
            is_straight, s_high = True, unique[0]
        elif unique == [12, 3, 2, 1, 0]:
            is_straight, s_high = True, 3

    cnt = defaultdict(int)
    for r in ranks:
        cnt[r] += 1

    groups = sorted(cnt.items(), key=lambda x: (x[1], x[0]), reverse=True)
    counts = tuple(n for _, n in groups)
    key_ranks = tuple(r for r, _ in groups)

    if is_straight and is_flush: return (8, s_high)
    if counts == (4, 1): return (7,) + key_ranks
    if counts == (3, 2): return (6,) + key_ranks
    if is_flush: return (5,) + tuple(ranks)
    if is_straight: return (4, s_high)
    if counts == (3, 1, 1): return (3,) + key_ranks
    if counts == (2, 2, 1): return (2,) + key_ranks
    if counts == (2, 1, 1, 1): return (1,) + key_ranks
    return (0,) + tuple(ranks)


# =============================================================================
# 3. GAME TREE
# =============================================================================

@dataclass
class TerminalPath:
    p1_seq: tuple
    p2_seq: tuple
    outcome: str
    p1_invested: float
    p2_invested: float


def enumerate_paths(initial_pot=1.0, bet_fractions=(0.5, 1.0), max_raises=0):
    paths = []

    def go(actor, p1_seq, p2_seq, pot, p1_in, p2_in, to_call, raises_left):
        if to_call == 0:
            if actor == 1:
                go(2, p1_seq + ('check',), p2_seq,
                   pot, p1_in, p2_in, 0, raises_left)
            else:
                paths.append(TerminalPath(
                    p1_seq, p2_seq + ('check',),
                    'showdown', p1_in, p2_in
                ))

            for frac in bet_fractions:
                bet = frac * pot
                if actor == 1:
                    go(2, p1_seq + (f'bet_{frac}',), p2_seq,
                       pot + bet, p1_in + bet, p2_in, bet, raises_left)
                else:
                    go(1, p1_seq, p2_seq + (f'bet_{frac}',),
                       pot + bet, p1_in, p2_in + bet, bet, raises_left)

        else:
            if actor == 1:
                paths.append(TerminalPath(
                    p1_seq + ('fold',), p2_seq,
                    'p1_fold', p1_in, p2_in
                ))
            else:
                paths.append(TerminalPath(
                    p1_seq, p2_seq + ('fold',),
                    'p2_fold', p1_in, p2_in
                ))

            if actor == 1:
                paths.append(TerminalPath(
                    p1_seq + ('call',), p2_seq,
                    'showdown', p1_in + to_call, p2_in
                ))
            else:
                paths.append(TerminalPath(
                    p1_seq, p2_seq + ('call',),
                    'showdown', p1_in, p2_in + to_call
                ))

    go(1, (), (), initial_pot, 0.0, 0.0, 0.0, max_raises)
    return paths


# =============================================================================
# 4. PAYOFF
# =============================================================================

def path_payoff(path, result, initial_pot):
    half = initial_pot / 2.0

    if path.outcome == 'p2_fold':
        return half + path.p2_invested
    if path.outcome == 'p1_fold':
        return -(half + path.p1_invested)

    if result == 1: return half + path.p2_invested
    if result == -1: return -(half + path.p1_invested)
    return 0.0


# =============================================================================
# 5. BUILD A (HAND-CONDITIONED)
# =============================================================================

def build_A(board_cards, initial_pot=1.0, bet_fractions=(0.5, 1.0), max_raises=0):

    board = [parse_card(c) for c in board_cards]
    board_set = set(board)

    paths = enumerate_paths(initial_pot, bet_fractions, max_raises)

    remaining = [c for c in range(52) if c not in board_set]
    all_hands = list(combinations(remaining, 2))

    print(f"  Hands per player: {len(all_hands)}")

    # Precompute hand strengths
    print("  Evaluating hand strengths...")
    hand_strength = {
        h: evaluate(h, board)
        for h in all_hands
    }

    # Expand sequences
    p1_seqs = [(h1, p.p1_seq) for h1 in all_hands for p in paths]
    p2_seqs = [(h2, p.p2_seq) for h2 in all_hands for p in paths]

    p1_idx = {s: i for i, s in enumerate(p1_seqs)}
    p2_idx = {s: j for j, s in enumerate(p2_seqs)}

    path_map = {(p.p1_seq, p.p2_seq): p for p in paths}

    m, n = len(p1_seqs), len(p2_seqs)
    print(f"  Matrix size: {m} x {n}")

    A = lil_matrix((m, n))

    for i, (h1, s1) in enumerate(p1_seqs):
        for j, (h2, s2) in enumerate(p2_seqs):

            if set(h1).intersection(h2):
                continue

            if (s1, s2) not in path_map:
                continue

            path = path_map[(s1, s2)]

            if path.outcome in ('p1_fold', 'p2_fold'):
                A[i, j] = path_payoff(path, 0, initial_pot)
            else:
                s1_val = hand_strength[h1]
                s2_val = hand_strength[h2]

                result = 1 if s1_val > s2_val else (-1 if s2_val > s1_val else 0)

                A[i, j] = path_payoff(path, result, initial_pot)

    return A.tocsr(), p1_seqs, p2_seqs


# =============================================================================
# 6. BUILD F
# =============================================================================

def build_F(A):
    m, n = A.shape
    F = lil_matrix((m + n, m + n))
    F[:m, m:] = A
    F[m:, :m] = -A.T
    return F.tocsr()


# =============================================================================
# 7. MAIN
# =============================================================================

if __name__ == '__main__':
    board_cards = ['Ah', 'Kd', '7c', '2h', '9s']

    print("=" * 60)
    print("Hand-conditioned matrix construction")
    print("=" * 60)

    A, p1_seqs, p2_seqs = build_A(board_cards)

    print(f"\nNonzero entries: {A.nnz} / {A.shape[0] * A.shape[1]}")

    print("\nBuilding F...")
    F = build_F(A)

    skew_err = (F + F.T).max()
    print(f"Skew-symmetry error: {skew_err}")