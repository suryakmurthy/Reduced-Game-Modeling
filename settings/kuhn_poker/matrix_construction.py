import itertools
import numpy as np

# --------------------------------------------------
# Utility: generate cards
# --------------------------------------------------

def generate_cards(n):
    # Use integers 0..n-1 (higher = stronger)
    return list(range(n))


# --------------------------------------------------
# Strategy encoding / decoding
# Each card has 2 bits:
#   (a_c, b_c)
#   a_c: initial action (0=check, 1=bet)
#   b_c: response (0=fold, 1=call)
# Total strategies = 4^n
# --------------------------------------------------

def decode_strategy(index, n):
    strategy = {}
    for c in range(n):
        a = (index >> (2*c)) & 1
        b = (index >> (2*c + 1)) & 1
        strategy[c] = (a, b)
    return strategy


def all_strategies(n):
    total = 4 ** n
    return [decode_strategy(i, n) for i in range(total)]


# --------------------------------------------------
# Deal generation (ordered pairs, no replacement)
# --------------------------------------------------

def generate_deals(cards):
    deals = []
    for c1 in cards:
        for c2 in cards:
            if c1 != c2:
                deals.append((c1, c2))
    return deals


# --------------------------------------------------
# Game simulation (single deal, deterministic)
# Returns payoff for Player 1
# --------------------------------------------------

def simulate(s1, s2, c1, c2):
    # Player 1 acts
    a1, b1 = s1[c1]
    a2, b2 = s2[c2]

    # Case 1: Player 1 checks
    if a1 == 0:
        # Player 2 decision after check
        if a2 == 0:
            # check-check → showdown (pot = 1)
            return showdown(c1, c2, pot=1)
        else:
            # Player 2 bets
            # Player 1 responds
            if b1 == 0:
                # fold → Player 2 wins 1
                return -1
            else:
                # call → showdown (pot = 2)
                return showdown(c1, c2, pot=2)

    # Case 2: Player 1 bets
    else:
        # Player 2 responds
        if b2 == 0:
            # fold → Player 1 wins 1
            return +1
        else:
            # call → showdown (pot = 2)
            return showdown(c1, c2, pot=2)


def showdown(c1, c2, pot):
    if c1 > c2:
        return pot
    else:
        return -pot


# --------------------------------------------------
# Expected payoff when s1 is Player 1
# --------------------------------------------------

def expected_payoff_P1(s1, s2, deals):
    total = 0.0
    prob = 1.0 / len(deals)

    for c1, c2 in deals:
        total += prob * simulate(s1, s2, c1, c2)

    return total


# --------------------------------------------------
# Symmetrized payoff (this is the key)
# --------------------------------------------------

def symmetrized_payoff(s1, s2, deals):
    u12 = expected_payoff_P1(s1, s2, deals)
    u21 = expected_payoff_P1(s2, s1, deals)
    return 0.5 * (u12 - u21)

def check_duplicate(strategies, n):
    for strat_idx_1 in range(len(strategies)):
        for strat_idx_2 in range(len(strategies)):
            if strat_idx_1 == strat_idx_2:
                continue
            strat_1 = strategies[strat_idx_1]
            strat_2 = strategies[strat_idx_2]
            flag = True
            for i in range(n):
                print(i, strat_1[i], strat_2[i])
                if strat_1[i] != strat_2[i]:
                    flag = False
            # print(strat_idx_1, strat_idx_2)
            if flag:
                return True
    return False


# --------------------------------------------------
# Build full payoff matrix
# --------------------------------------------------

def build_payoff_matrix(n):
    cards = generate_cards(n)
    deals = generate_deals(cards)

    strategies = all_strategies(n)
    print(check_duplicate(strategies, n))
    m = len(strategies)
    
    A = np.zeros((m, m))

    for i, s1 in enumerate(strategies):
        for j, s2 in enumerate(strategies):
            A[i, j] = symmetrized_payoff(s1, s2, deals)

    return A


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":
    n = 3  # change this!

    print(f"Building Kuhn Poker matrix for n={n} cards...")
    A = build_payoff_matrix(n)
    np.save(f"A_{n}", A)
    print("Matrix shape:", A.shape)

    # Check skew-symmetry
    print("Skew-symmetric:", np.allclose(A, -A.T, atol=1e-9))

    # Check diagonal
    print("Zero diagonal:", np.allclose(np.diag(A), 0))

    # Optional: print a small part
    print("Top-left corner of matrix:")
    print(A[:5, :5])