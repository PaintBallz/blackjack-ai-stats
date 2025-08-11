from dataclasses import dataclass, replace
from collections import Counter
from typing import Dict, Tuple, List, Optional, Iterable
import random, math, secrets, os
import matplotlib.pyplot as plt

# =========================
# Blackjack core
# =========================

RANKS = ['2','3','4','5','6','7','8','9','10','J','Q','K','A']

def make_shoe(decks: int = 6) -> Counter:
    """Standard 52-card deck counts per rank, times `decks`."""
    shoe = Counter()
    for r in RANKS:
        shoe[r] = 4 * decks  # each rank has 4 per deck
    return shoe

def card_value(rank: str) -> int:
    if rank == 'A': return 11
    if rank in ['10','J','Q','K']: return 10
    return int(rank)

def hand_value(cards: Tuple[str, ...]) -> Tuple[int, bool]:
    total = 0
    aces = 0
    for r in cards:
        if r == 'A':
            total += 11
            aces += 1
        elif r in {'10', 'J', 'Q', 'K'}:
            total += 10
        else:
            total += int(r)
    while total > 21 and aces:
        total -= 10
        aces -= 1
    is_soft = (aces > 0)
    return total, is_soft

def is_blackjack(cards: Tuple[str, ...]) -> bool:
    return len(cards) == 2 and 'A' in cards and any(c in {'10','J','Q','K'} for c in cards)

A_HIT = "HIT"
A_STAND = "STAND"
A_DOUBLE = "DOUBLE"

@dataclass(frozen=True)
class BJState:
    to_move: str
    player_cards: Tuple[str, ...]
    dealer_cards: Tuple[str, ...]      # (upcard, None) during play; final tuple after resolution
    shoe: Tuple[Tuple[str,int], ...]   # immutable snapshot of remaining shoe
    base_bet: int                      # static bet (configurable in run_comparison)
    bet_mult: int                      # 1 normally, 2 after DOUBLE
    can_double: bool
    resolved: bool

    def shoe_counter(self) -> Counter:
        return Counter(dict(self.shoe))

class Blackjack:
    """Blackjack environment. Dealer stands on all 17 (S17 by default)."""
    def __init__(self, decks: int = 6, dealer_hits_soft_17: bool = False):
        self.decks = decks
        self.H17 = dealer_hits_soft_17
        self._round_hole: Optional[str] = None
        self._round_dealer_hits: Tuple[str, ...] = ()
        self._round_final_dealer: Optional[Tuple[str, ...]] = None

    # --- dealing & shoes ---
    def _draw_from_shoe(self, shoe: Counter, rank: Optional[str] = None) -> Tuple[str, Counter]:
        """Draw specific rank (if provided) or sample proportional to counts."""
        if rank is None:
            total = sum(shoe.values())
            r = random.randrange(total)
            cum = 0
            for k, v in shoe.items():
                cum += v
                if r < cum:
                    rank = k
                    break
        if rank is None or shoe[rank] <= 0:
            raise RuntimeError("Invalid draw")
        shoe2 = shoe.copy()
        shoe2[rank] -= 1
        if shoe2[rank] == 0:
            del shoe2[rank]
        return rank, shoe2

    def _precompute_dealer_hand(self, upcard: str, shoe_after_dealer: Counter):
        dealer = [upcard, self._round_hole]
        shoe = shoe_after_dealer.copy()
        # Dealer hits while total < 17; stands on 17+
        while True:
            dv, _ = hand_value(tuple(dealer))
            if dv >= 17:
                break
            rank, shoe = self._draw_from_shoe(shoe)
            dealer.append(rank)
        self._round_dealer_hits = tuple(dealer[2:])
        self._round_final_dealer = tuple(dealer)

    # --- rules interface ---
    def actions(self, s: BJState) -> Iterable[str]:
        if self.is_terminal(s) or s.to_move != 'Player':
            return []
        acts = [A_HIT, A_STAND]
        if s.can_double:
            acts.append(A_DOUBLE)
        return acts

    def is_terminal(self, s: BJState) -> bool:
        return s.resolved

    def _maybe_resolve_naturals(self, s: BJState) -> BJState:
        p_nat = is_blackjack(s.player_cards)
        hole = s.dealer_cards[1] if len(s.dealer_cards) > 1 else None
        if hole is None:
            hole = self._round_hole
        d_nat = is_blackjack((s.dealer_cards[0], hole)) if hole is not None else False
        if p_nat or d_nat:
            return replace(s, resolved=True, to_move='Dealer')
        return s

    def result(self, s: BJState, move: str) -> BJState:
        if move not in self.actions(s):
            raise ValueError("Illegal move")
        shoe = s.shoe_counter()
        player = list(s.player_cards)

        if move == A_HIT:
            rank, shoe = self._draw_from_shoe(shoe)
            player.append(rank)
            pv, _ = hand_value(tuple(player))
            if pv > 21:
                return replace(s, player_cards=tuple(player),
                               shoe=tuple(sorted(shoe.items())),
                               to_move='Dealer', resolved=True, can_double=False)
            else:
                return replace(s, player_cards=tuple(player),
                               shoe=tuple(sorted(shoe.items())),
                               to_move='Player', can_double=False)

        if move == A_DOUBLE:
            rank, shoe = self._draw_from_shoe(shoe)
            player.append(rank)
            s2 = replace(s, player_cards=tuple(player),
                         shoe=tuple(sorted(shoe.items())),
                         bet_mult=2, to_move='Dealer', can_double=False)
            return self._dealer_play(s2)

        if move == A_STAND:
            return self._dealer_play(replace(s, to_move='Dealer', can_double=False))

        raise RuntimeError("Unreachable")

    def _dealer_play(self, s: BJState) -> BJState:
        if s.resolved:
            return s
        final = self._round_final_dealer
        if final is None:  # safety fallback
            final = (s.dealer_cards[0], self._round_hole)
        return replace(s, dealer_cards=final, resolved=True)

    # --- payouts ---
    def utility_ev(self, s: BJState) -> float:
        """
        Returns chip delta for the player:
        - Naturals: +1.5 * base_bet (no double), or -base_bet if dealer natural only.
        - Otherwise: +/- (base_bet * bet_mult) or 0 on push.
        """
        base = s.base_bet
        mult = s.bet_mult
        pv, _ = hand_value(s.player_cards)

        # Dealer tuple using committed hole if needed
        if len(s.dealer_cards) > 1 and s.dealer_cards[1] is not None:
            dealer_tuple = s.dealer_cards
        else:
            dealer_tuple = (s.dealer_cards[0], self._round_hole)

        dv, _ = hand_value(dealer_tuple)
        p_nat = is_blackjack(s.player_cards)
        d_nat = is_blackjack(dealer_tuple)

        if p_nat or d_nat:
            if p_nat and not d_nat: return 1.5 * base
            if d_nat and not p_nat: return -1.0 * base
            return 0.0

        wager = base * mult
        if pv > 21: return -wager
        if dv > 21: return +wager
        if pv > dv: return +wager
        if pv < dv: return -wager
        return 0.0

    def utility_win(self, s: BJState) -> float:
        """+1 on win, 0 on push, -1 on loss."""
        delta = self.utility_ev(s)
        return 1.0 if delta > 0 else (-1.0 if delta < 0 else 0.0)

# =========================
# Expectiminimax (maximize win probability)
# =========================

def expectiminimax_win(game: Blackjack, state: BJState, depth_limit: int = 6) -> Tuple[float, Optional[str]]:
    cache: Dict[Tuple[BJState, int], Tuple[float, Optional[str]]] = {}
    NEG_INF = -1e9

    def chance_children_hit(s: BJState) -> List[Tuple[float, BJState]]:
        shoe = s.shoe_counter()
        total = sum(shoe.values())
        out: List[Tuple[float, BJState]] = []
        for r, cnt in shoe.items():
            p = cnt / total
            new_shoe = shoe.copy()
            new_shoe[r] -= 1
            if new_shoe[r] == 0: del new_shoe[r]
            new_cards = tuple(list(s.player_cards) + [r])
            pv, _ = hand_value(new_cards)
            if pv > 21:
                ns = replace(s, player_cards=new_cards,
                             shoe=tuple(sorted(new_shoe.items())),
                             to_move='Dealer', resolved=True, can_double=False)
            else:
                ns = replace(s, player_cards=new_cards,
                             shoe=tuple(sorted(new_shoe.items())),
                             to_move='Player', can_double=False)
            out.append((p, ns))
        return out

    def eval_ev(s: BJState, depth: int) -> Tuple[float, Optional[str]]:
        key = (s, depth)
        if key in cache:
            return cache[key]

        # Cutoff heuristic at player node
        if game.is_terminal(s) or (depth <= 0 and s.to_move == 'Player'):
            if not game.is_terminal(s) and s.to_move == 'Player':
                stand_win = game.utility_win(game._dealer_play(replace(s, to_move='Dealer')))
                hit_exp = 0.0
                for p, ns in chance_children_hit(s):
                    if not ns.resolved and ns.to_move == 'Player':
                        hit_exp += p * game.utility_win(game._dealer_play(replace(ns, to_move='Dealer')))
                    else:
                        hit_exp += p * game.utility_win(ns)
                val = max(stand_win, hit_exp)
                cache[key] = (val, None)
                return cache[key]
            val = game.utility_win(s)
            cache[key] = (val, None)
            return cache[key]

        if s.to_move == 'Player':
            best = NEG_INF
            best_a: Optional[str] = None
            for a in game.actions(s):
                if a == A_HIT:
                    expv = 0.0
                    for p, ns in chance_children_hit(s):
                        v, _ = eval_ev(ns, depth)  # chance doesn't reduce depth
                        expv += p * v
                    val = expv
                else:
                    ns = game.result(s, a)
                    v, _ = eval_ev(ns, depth - 1)
                    val = v
                if val > best:
                    best, best_a = val, a
            cache[key] = (best, best_a)
            return cache[key]

        # Dealer turn: deterministic (precomputed)
        ns = game._dealer_play(s)
        v, _ = eval_ev(ns, depth)
        cache[key] = (v, None)
        return cache[key]

    return eval_ev(state, depth_limit)

# =========================
# Monte Carlo Tree Search (two distinct variants)
# =========================

class MCTSNode:
    __slots__ = ("state","parent","children","N","W","untried")
    def __init__(self, state: BJState, parent: Optional['MCTSNode'], actions: List[str]):
        self.state = state
        self.parent = parent
        self.children: Dict[str, 'MCTSNode'] = {}
        self.N = 0
        self.W = 0.0
        self.untried = list(actions)

def stochastic_step_rng(game: Blackjack, s: BJState, action: str, rng: random.Random) -> BJState:
    """Apply action; sample draws from shoe using a provided RNG."""
    if action == A_HIT:
        shoe = s.shoe_counter()
        total = sum(shoe.values())
        r = rng.randrange(total)
        cum = 0
        rank = None
        for k, v in shoe.items():
            cum += v
            if r < cum:
                rank = k
                break
        player = list(s.player_cards)
        player.append(rank)
        shoe[rank] -= 1
        if shoe[rank] == 0: del shoe[rank]
        pv, _ = hand_value(tuple(player))
        if pv > 21:
            return replace(s, player_cards=tuple(player),
                           shoe=tuple(sorted(shoe.items())),
                           to_move='Dealer', resolved=True, can_double=False)
        else:
            return replace(s, player_cards=tuple(player),
                           shoe=tuple(sorted(shoe.items())),
                           to_move='Player', can_double=False)
    elif action in (A_STAND, A_DOUBLE):
        return game.result(s, action)
    else:
        raise ValueError("Unknown action")

# -------- Distinct rollout policies --------

def rollout_profit(game: Blackjack, s: BJState, rng: random.Random) -> float:
    """Profit-focused rollout: occasionally DOUBLE on 9–11; thresholdy HIT/ STAND."""
    state = s
    while not game.is_terminal(state):
        if state.to_move == 'Dealer':
            state = game._dealer_play(state)
            break
        pv, _ = hand_value(state.player_cards)
        if state.can_double and 9 <= pv <= 11 and rng.random() < 0.30:
            state = stochastic_step_rng(game, state, A_DOUBLE, rng)
        elif pv <= 11:
            state = stochastic_step_rng(game, state, A_HIT, rng)
        elif 12 <= pv <= 16:
            up = state.dealer_cards[0]
            upv = 11 if up == 'A' else card_value(up)
            if up == 'A' or upv >= 7:
                state = stochastic_step_rng(game, state, A_HIT, rng)
            else:
                state = stochastic_step_rng(game, state, A_STAND, rng)
        else:
            state = stochastic_step_rng(game, state, A_STAND, rng)
    return game.utility_ev(state)

def rollout_win(game: Blackjack, s: BJState, rng: random.Random) -> float:
    """Win-probability rollout: never DOUBLE; bias to avoid busts."""
    state = s
    while not game.is_terminal(state):
        if state.to_move == 'Dealer':
            state = game._dealer_play(state)
            break
        pv, _ = hand_value(state.player_cards)
        if pv <= 11:
            state = stochastic_step_rng(game, state, A_HIT, rng)
        elif 12 <= pv <= 16:
            up = state.dealer_cards[0]
            upv = 11 if up == 'A' else card_value(up)
            if up == 'A' or upv >= 7:
                state = stochastic_step_rng(game, state, A_HIT, rng)
            else:
                state = stochastic_step_rng(game, state, A_STAND, rng)
        else:
            state = stochastic_step_rng(game, state, A_STAND, rng)
    return game.utility_win(state)

# -------- Core UCT engine (parametrized) --------

def mcts_core(game: Blackjack,
              root_state: BJState,
              iters: int,
              reward_rollout_fn,              # (game, state, rng) -> scalar
              C: float = math.sqrt(2),
              rng: Optional[random.Random] = None) -> str:
    """
    Generic UCT with pluggable reward+rollout and independent RNG.
    """
    rng = rng or random.Random(secrets.randbits(64))
    root = MCTSNode(root_state, None, list(game.actions(root_state)))

    def ucb(node: MCTSNode, child: MCTSNode) -> float:
        if child.N == 0: return float('inf')
        return (child.W / child.N) + C * math.sqrt(math.log(node.N + 1) / child.N)

    for _ in range(iters):
        # 1) Selection
        node = root
        state = node.state

        # Auto-play dealer if needed
        if state.to_move == 'Dealer' and not game.is_terminal(state):
            state = game._dealer_play(state)
            node = MCTSNode(state, node, list(game.actions(state)))

        while not game.is_terminal(state) and not node.untried and node.children:
            a, child = max(node.children.items(), key=lambda kv: ucb(node, kv[1]))
            state = stochastic_step_rng(game, state, a, rng)
            node = child

        # 2) Expansion
        if not game.is_terminal(state) and node.untried:
            a = node.untried.pop()
            next_state = stochastic_step_rng(game, state, a, rng)
            child = MCTSNode(next_state, node, list(game.actions(next_state)))
            node.children[a] = child
            node = child
            state = next_state

        # 3) Simulation
        reward = reward_rollout_fn(game, state, rng)

        # 4) Backprop
        while node is not None:
            node.N += 1
            node.W += reward
            node = node.parent

    if not root.children:
        return A_STAND
    maxN = max(ch.N for ch in root.children.values())
    best_actions = [a for a, ch in root.children.items() if ch.N == maxN]
    return rng.choice(best_actions)

# -------- Public distinct MCTS wrappers --------

def mcts_choose_profit(game: Blackjack, root_state: BJState, iters: int) -> str:
    rng = random.Random(secrets.randbits(64))
    return mcts_core(game, root_state, iters, reward_rollout_fn=rollout_profit, C=math.sqrt(2), rng=rng)

def mcts_choose_win(game: Blackjack, root_state: BJState, iters: int) -> str:
    rng = random.Random(secrets.randbits(64))
    return mcts_core(game, root_state, iters, reward_rollout_fn=rollout_win, C=math.sqrt(2), rng=rng)

# -------- Policies --------

def policy_mcts_profit(game: Blackjack, state: BJState, iters=3000) -> str:
    return mcts_choose_profit(game, state, iters=iters)

def policy_mcts_win(game: Blackjack, state: BJState, iters=3000) -> str:
    return mcts_choose_win(game, state, iters=iters)

def policy_expecti_win(game: Blackjack, state: BJState, depth=6) -> str:
    _, a = expectiminimax_win(game, state, depth_limit=depth)
    return a or A_STAND

# =========================
# Runner utilities
# =========================

def play_full_hand(game: Blackjack, start: BJState, chooser) -> Tuple[BJState, List[str]]:
    s = start
    actions_taken: List[str] = []
    while not game.is_terminal(s):
        if s.to_move == 'Dealer':
            s = game._dealer_play(s)
            break
        a = chooser(game, s)
        actions_taken.append(a)
        s = game.result(s, a)  # unified transition (HIT/DOUBLE/STAND)
    return s, actions_taken

def reconstruct_final_dealer(game: Blackjack) -> Tuple[str, ...]:
    return game._round_final_dealer or ('?', '?')

# =========================
# Plotting (only at the end; no per-round delta plot)
# =========================

def plot_summary(stack_hist: Dict[str, List[float]],
                 save_dir: Optional[str] = None):
    """Create a summary plot after all rounds:
       - Line chart of chip stacks over rounds
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    rounds = len(next(iter(stack_hist.values()), []))
    x = list(range(1, rounds + 1))

    # Stacks over time
    plt.figure()
    for tag, series in stack_hist.items():
        plt.plot(x, series, label=tag)
    plt.title("Chip Stacks Over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Chips")
    plt.legend()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "stacks_over_time.png"), bbox_inches="tight")

    # Show at the end (even if saved)
    plt.show()

# =========================
# Simple console loading bar
# =========================

def progress_bar(current: int, total: int, width: int = 40, prefix: str = "Playing hands") -> None:
    filled = int(width * current / total)
    bar = "█" * filled + "-" * (width - filled)
    print(f"\r{prefix}: |{bar}| {current}/{total}", end="", flush=True)
    if current == total:
        print()  # newline at completion

# =========================
# Tournament: shared dealer, static bet, hidden hole
# =========================

def run_comparison(rounds=10, iters=2500, depth=6, decks=4,
                   starting_chips=1000, base_bet=50,
                   plot=True, save_dir: Optional[str] = None):
    game = Blackjack(decks=decks, dealer_hits_soft_17=False)

    agents = ('MCTS-Profit', 'MCTS-Win', 'Expecti-Win')
    stacks = {a: float(starting_chips) for a in agents}
    rec    = {a: {'win':0,'loss':0,'push':0} for a in agents}

    # histories for end-of-run plotting
    stack_hist = {a: [] for a in agents}  # running stacks after each round

    for rnd in range(1, rounds+1):
        # update loading bar
        progress_bar(rnd - 1, rounds)

        random.seed(secrets.randbits(64))
        base_shoe = make_shoe(decks)

        # Deal dealer once (shared)
        d1, shoe_after = game._draw_from_shoe(base_shoe)
        d2, shoe_after = game._draw_from_shoe(shoe_after)
        game._round_hole = d2
        dealer_public = (d1, None)
        game._precompute_dealer_hand(d1, shoe_after)

        # helper: deal one player from a cloned shoe
        def deal_player_from(cloned_shoe):
            p1, cs = game._draw_from_shoe(cloned_shoe)
            p2, cs = game._draw_from_shoe(cs)
            s = BJState(
                to_move='Player',
                player_cards=(p1, p2),
                dealer_cards=dealer_public,
                shoe=tuple(sorted(cs.items())),
                base_bet=base_bet,
                bet_mult=1,
                can_double=True,
                resolved=False
            )
            return game._maybe_resolve_naturals(s)

        s1 = deal_player_from(shoe_after.copy())  # MCTS-Profit
        s2 = deal_player_from(shoe_after.copy())  # MCTS-Win
        s3 = deal_player_from(shoe_after.copy())  # Expecti-Win

        f1, _ = play_full_hand(game, s1, lambda g, s: policy_mcts_profit(g, s, iters))
        f2, _ = play_full_hand(game, s2, lambda g, s: policy_mcts_win(g, s, iters))
        f3, _ = play_full_hand(game, s3, lambda g, s: policy_expecti_win(g, s, depth))

        def settle(tag, final_state):
            delta = game.utility_ev(final_state)
            if   delta > 0: rec[tag]['win']  += 1
            elif delta < 0: rec[tag]['loss'] += 1
            else:           rec[tag]['push'] += 1
            stacks[tag] += delta

        settle('MCTS-Profit', f1)
        settle('MCTS-Win',    f2)
        settle('Expecti-Win', f3)

        # record histories
        for a in agents:
            stack_hist[a].append(stacks[a])

    # complete loading bar
    progress_bar(rounds, rounds)

    # plots only once, after the tournament
    if plot:
        plot_summary(stack_hist, save_dir=save_dir)

    # final chip stacks only (no per-round prints)
    print("\n==== Final Chip Stacks ====")
    for tag in agents:
        w, l, p = rec[tag]['win'], rec[tag]['loss'], rec[tag]['push']
        print(f"{tag:13s} | Stack={stacks[tag]:.2f} | W-L-P = {w}-{l}-{p}")

if __name__ == "__main__":
    run_comparison(
        rounds=50,
        iters=5000,
        depth=6,
        decks=6,
        starting_chips=1000,
        base_bet=100,
        plot=True,            # show summary plot once at the end
        save_dir="plots"      # set to None to only display; or a folder name to save PNGs
    )
