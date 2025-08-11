import random, math, secrets
from dataclasses import dataclass, replace
from collections import Counter
from typing import Dict, Tuple, List, Optional, Iterable

# =========================
# Blackjack core (with Insurance as an action)
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
A_INSURANCE = "INSURANCE"           # take insurance (half bet)
A_SKIP_INSURANCE = "SKIP_INSURANCE" # decline insurance

@dataclass(frozen=True)
class BJState:
    to_move: str
    player_cards: Tuple[str, ...]
    dealer_cards: Tuple[str, ...]      # (upcard, None) during play; final tuple after resolution
    shoe: Tuple[Tuple[str,int], ...]   # immutable snapshot of remaining shoe
    base_bet: int                      # static bet per round
    bet_mult: int                      # 1 normally, 2 after DOUBLE
    can_double: bool
    resolved: bool
    insurance_bet: int = 0             # 0 if no insurance; otherwise base_bet // 2
    insurance_allowed: bool = False    # whether insurance decision is pending now

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
            dv, soft = hand_value(tuple(dealer))
            # Dealer hits soft 17 only if configured
            if dv > 17 or (dv == 17 and not (soft and self.H17)):
                break
            if dv == 17 and soft and self.H17:
                pass
            if dv < 17 or (dv == 17 and soft and self.H17):
                rank, shoe = self._draw_from_shoe(shoe)
                dealer.append(rank)
                continue
            break
        self._round_dealer_hits = tuple(dealer[2:])
        self._round_final_dealer = tuple(dealer)

    # --- rules interface ---
    def actions(self, s: BJState) -> Iterable[str]:
        if self.is_terminal(s) or s.to_move != 'Player':
            return []
        # If insurance decision is pending, only offer insurance choices
        if s.insurance_allowed:
            return [A_INSURANCE, A_SKIP_INSURANCE]
        acts = [A_HIT, A_STAND]
        if s.can_double:
            acts.append(A_DOUBLE)
        return acts

    def is_terminal(self, s: BJState) -> bool:
        return s.resolved

    def _maybe_resolve_naturals(self, s: BJState) -> BJState:
        # Do not resolve naturals until insurance decision has been made (if applicable)
        if s.insurance_allowed:
            return s
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

        # --- Insurance decision branch ---
        if move == A_INSURANCE:
            s2 = replace(
                s,
                insurance_bet=s.base_bet // 2,
                insurance_allowed=False  # decision made; now we may resolve naturals immediately
            )
            return self._maybe_resolve_naturals(s2)

        if move == A_SKIP_INSURANCE:
            s2 = replace(
                s,
                insurance_bet=0,
                insurance_allowed=False
            )
            return self._maybe_resolve_naturals(s2)

        # --- Regular actions ---
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
        - Insurance (if taken): pays 2:1 if dealer has blackjack; else it's lost.
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

        # Insurance resolution
        ins = 0.0
        if s.insurance_bet:
            ins = (2.0 * s.insurance_bet) if d_nat else (-1.0 * s.insurance_bet)

        # Natural cases (resolved immediately)
        if p_nat or d_nat:
            if p_nat and not d_nat:
                main = 1.5 * base
            elif d_nat and not p_nat:
                main = -1.0 * base
            else:
                main = 0.0  # both have blackjack -> push main
            return main + ins

        # Regular play (no naturals)
        wager = base * mult
        if pv > 21: return -wager + ins
        if dv > 21: return +wager + ins
        if pv > dv: return +wager + ins
        if pv < dv: return -wager + ins
        return 0.0 + ins

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
                available = list(game.actions(s))
                # If insurance decision is pending at cutoff, evaluate both choices
                if A_INSURANCE in available or A_SKIP_INSURANCE in available:
                    best = -1e9
                    for a in available:
                        ns = game.result(s, a)
                        if game.is_terminal(ns):
                            v = game.utility_win(ns)
                        else:
                            stand_win = game.utility_win(game._dealer_play(replace(ns, to_move='Dealer')))
                            hit_exp = 0.0
                            for p, cns in chance_children_hit(ns):
                                if not cns.resolved and cns.to_move == 'Player':
                                    hit_exp += p * game.utility_win(game._dealer_play(replace(cns, to_move='Dealer')))
                                else:
                                    hit_exp += p * game.utility_win(cns)
                            v = max(stand_win, hit_exp)
                        if v > best: best = v
                    cache[key] = (best, None)
                    return cache[key]
                # Otherwise, approximate by stand vs hit from current state
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
    elif action in (A_STAND, A_DOUBLE, A_INSURANCE, A_SKIP_INSURANCE):
        return game.result(s, action)
    else:
        raise ValueError("Unknown action")

# -------- Distinct rollout policies --------

def rollout_profit(game: Blackjack, s: BJState, rng: random.Random) -> float:
    """Profit-focused rollout: occasionally DOUBLE on 9–11; thresholdy HIT/STAND; insurance: even-money only."""
    state = s
    while not game.is_terminal(state):
        if state.to_move == 'Dealer':
            state = game._dealer_play(state)
            break
        # Insurance gate if pending
        acts = list(game.actions(state))
        if A_INSURANCE in acts or A_SKIP_INSURANCE in acts:
            # Even-money only
            if is_blackjack(state.player_cards):
                state = stochastic_step_rng(game, state, A_INSURANCE, rng)
            else:
                state = stochastic_step_rng(game, state, A_SKIP_INSURANCE, rng)
            continue
        pv, soft = hand_value(state.player_cards)
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
    """Win-probability rollout: never DOUBLE; avoid busts; insurance: even-money only."""
    state = s
    while not game.is_terminal(state):
        if state.to_move == 'Dealer':
            state = game._dealer_play(state)
            break
        # Insurance gate if pending
        acts = list(game.actions(state))
        if A_INSURANCE in acts or A_SKIP_INSURANCE in acts:
            if is_blackjack(state.player_cards):
                state = stochastic_step_rng(game, state, A_INSURANCE, rng)
            else:
                state = stochastic_step_rng(game, state, A_SKIP_INSURANCE, rng)
            continue
        pv, soft = hand_value(state.player_cards)
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

def policy_mcts_profit(game: Blackjack, state: BJState, iters=2000) -> str:
    return mcts_choose_profit(game, state, iters=iters)

def policy_mcts_win(game: Blackjack, state: BJState, iters=2000) -> str:
    return mcts_choose_win(game, state, iters=iters)

def policy_expecti_win(game: Blackjack, state: BJState, depth=6) -> str:
    _, a = expectiminimax_win(game, state, depth_limit=depth)
    return a or A_STAND

# =========================
# Runner utilities (non-GUI)
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
        s = game.result(s, a)  # unified transition
    return s, actions_taken

def reconstruct_final_dealer(game: Blackjack) -> Tuple[str, ...]:
    return game._round_final_dealer or ('?', '?')

# =========================
# Tkinter GUI for Human vs 3 AI players
# =========================

import tkinter as tk
from tkinter import ttk

class BlackjackGUI:
    def __init__(self, root,
                 decks=6,
                 base_bet=100,
                 starting_chips=1000,
                 iters=3000,
                 depth=6,
                 h17=False):
        self.root = root
        self.root.title("Blackjack AI Table — Human + 3 Agents")
        self.game = Blackjack(decks=decks, dealer_hits_soft_17=h17)
        self.base_bet = base_bet
        self.starting_chips = float(starting_chips)
        self.iters = iters
        self.depth = depth

        # Players
        self.tags = ["Human", "MCTS-Profit", "MCTS-Win", "Expecti-Win"]
        self.stacks = {t: float(starting_chips) for t in self.tags}
        self.rec = {t: {'win':0,'loss':0,'push':0} for t in self.tags}

        # Round state
        self.current_round = 0
        self.dealer_public: Tuple[str, Optional[str]] = ("?", None)
        self.final_dealer: Tuple[str, ...] = ("?", "?")
        self.hole_committed: Optional[str] = None

        self.start_states: Dict[str, BJState] = {}
        self.final_states: Dict[str, BJState] = {}
        self.actions_taken: Dict[str, List[str]] = {}
        self.settled_this_round: Dict[str, bool] = {}
        self.human_actions: List[str] = []

        # Build UI
        self._build_ui()

    # ---------- UI Layout ----------
    def _build_ui(self):
        self.root.geometry("980x700")
        container = ttk.Frame(self.root, padding=10)
        container.pack(fill="both", expand=True)

        # Header
        header = ttk.Frame(container)
        header.pack(fill="x", pady=(0,10))
        self.round_label = ttk.Label(header, text="Round: 0", font=("Segoe UI", 14, "bold"))
        self.round_label.pack(side="left")
        ttk.Label(header, text=f"Base Bet: {self.base_bet}", font=("Segoe UI", 12)).pack(side="left", padx=20)

        self.next_btn = ttk.Button(header, text="Deal / Next Round", command=self.start_round)
        self.next_btn.pack(side="right")

        # Dealer frame
        dealer_frame = ttk.Labelframe(container, text="Dealer", padding=10)
        dealer_frame.pack(fill="x", pady=6)

        self.dealer_up = ttk.Label(dealer_frame, text="Upcard: ?", font=("Consolas", 14))
        self.dealer_up.pack(side="left", padx=8)

        self.dealer_hole = ttk.Label(dealer_frame, text="Hole: (hidden)", font=("Consolas", 14))
        self.dealer_hole.pack(side="left", padx=8)

        self.dealer_total = ttk.Label(dealer_frame, text="Total: -", font=("Consolas", 14))
        self.dealer_total.pack(side="left", padx=8)

        # Players area
        players_frame = ttk.Frame(container)
        players_frame.pack(fill="both", expand=True, pady=6)

        self.pframes: Dict[str, dict] = {}

        for tag in self.tags:
            frame = ttk.Labelframe(players_frame, text=tag, padding=10)
            frame.pack(fill="x", pady=6)

            top = ttk.Frame(frame)
            top.pack(fill="x")

            self.pframes.setdefault(tag, {})
            self.pframes[tag]['cards'] = ttk.Label(top, text="Cards: []", font=("Consolas", 12))
            self.pframes[tag]['cards'].pack(side="left", padx=5)

            self.pframes[tag]['total'] = ttk.Label(top, text="Total: -", font=("Consolas", 12))
            self.pframes[tag]['total'].pack(side="left", padx=10)

            self.pframes[tag]['actions'] = ttk.Label(top, text="Actions: -", font=("Consolas", 12))
            self.pframes[tag]['actions'].pack(side="left", padx=10)

            self.pframes[tag]['result'] = ttk.Label(top, text="Result: -", font=("Consolas", 12))
            self.pframes[tag]['result'].pack(side="left", padx=10)

            self.pframes[tag]['stack'] = ttk.Label(top, text=f"Stack: {self.stacks[tag]:.2f}", font=("Consolas", 12, "bold"))
            self.pframes[tag]['stack'].pack(side="right", padx=5)

            if tag == "Human":
                btns = ttk.Frame(frame)
                btns.pack(fill="x", pady=(8,0))
                self.btn_ins = ttk.Button(btns, text="Insurance", command=lambda: self.on_human_action(A_INSURANCE))
                self.btn_skip_ins = ttk.Button(btns, text="Skip Insurance", command=lambda: self.on_human_action(A_SKIP_INSURANCE))
                self.btn_hit = ttk.Button(btns, text="Hit", command=lambda: self.on_human_action(A_HIT))
                self.btn_stand = ttk.Button(btns, text="Stand", command=lambda: self.on_human_action(A_STAND))
                self.btn_double = ttk.Button(btns, text="Double", command=lambda: self.on_human_action(A_DOUBLE))
                for b in (self.btn_ins, self.btn_skip_ins, self.btn_hit, self.btn_stand, self.btn_double):
                    b.pack(side="left", padx=5)
            else:
                # No buttons for AI players
                pass

        # Status log
        log_frame = ttk.Labelframe(container, text="Round Log", padding=10)
        log_frame.pack(fill="both", expand=True, pady=6)
        self.log = tk.Text(log_frame, height=10, wrap="word", font=("Consolas", 10))
        self.log.pack(fill="both", expand=True)
        self._set_human_buttons(enabled=False)

    # ---------- Helpers ----------
    def _set_human_buttons(self, enabled: bool, insurance_phase: Optional[bool]=None, can_double: bool=False):
        state = ("!disabled" if enabled else "disabled")
        # Default disable all
        for b in (getattr(self, 'btn_ins', None),
                  getattr(self, 'btn_skip_ins', None),
                  getattr(self, 'btn_hit', None),
                  getattr(self, 'btn_stand', None),
                  getattr(self, 'btn_double', None)):
            if b: b.state([state])

        # If enabled, refine based on phase
        if enabled:
            if insurance_phase is True:
                # Only insurance choices
                self.btn_hit.state(["disabled"])
                self.btn_stand.state(["disabled"])
                self.btn_double.state(["disabled"])
            else:
                # Regular actions
                self.btn_ins.state(["disabled"])
                self.btn_skip_ins.state(["disabled"])
                if not can_double:
                    self.btn_double.state(["disabled"])

    def _append_log(self, msg: str):
        self.log.insert("end", msg + "\n")
        self.log.see("end")

    def _format_cards(self, cards: Tuple[str, ...]) -> str:
        return "[" + ", ".join(cards) + "]"

    # ---------- Round lifecycle ----------
    def start_round(self):
        # Reset UI for new round
        self.current_round += 1
        self.round_label.config(text=f"Round: {self.current_round}")
        self.log.delete("1.0", "end")

        # Disable deal button until round is done
        self.next_btn.state(["disabled"])
        self._append_log(f"=== Dealing Round {self.current_round} ===")

        # Reset per-round
        self.start_states.clear()
        self.final_states.clear()
        self.actions_taken.clear()
        self.settled_this_round = {t: False for t in self.tags}
        self.human_actions = []

        # Clear UI fields
        for tag in self.tags:
            self.pframes[tag]['cards'].config(text="Cards: []")
            self.pframes[tag]['total'].config(text="Total: -")
            self.pframes[tag]['actions'].config(text="Actions: -")
            self.pframes[tag]['result'].config(text="Result: -")
        self.dealer_up.config(text="Upcard: ?")
        self.dealer_hole.config(text="Hole: (hidden)")
        self.dealer_total.config(text="Total: -")

        # Prepare round
        random.seed(secrets.randbits(64))
        base_shoe = make_shoe(self.game.decks)

        # Deal dealer once
        d1, shoe_after = self.game._draw_from_shoe(base_shoe)
        d2, shoe_after = self.game._draw_from_shoe(shoe_after)
        self.game._round_hole = d2
        self.dealer_public = (d1, None)
        self.game._precompute_dealer_hand(d1, shoe_after)  # shared dealer
        self.final_dealer = reconstruct_final_dealer(self.game)

        self.dealer_up.config(text=f"Upcard: {d1}")
        self._append_log(f"Dealer Upcard: {d1}. Hole is hidden.")

        # Deal each player from a cloned shoe
        def deal_player_from(cloned_shoe):
            p1, cs = self.game._draw_from_shoe(cloned_shoe)
            p2, cs = self.game._draw_from_shoe(cs)
            s = BJState(
                to_move='Player',
                player_cards=(p1, p2),
                dealer_cards=self.dealer_public,
                shoe=tuple(sorted(cs.items())),
                base_bet=self.base_bet,
                bet_mult=1,
                can_double=True,
                resolved=False,
                insurance_bet=0,
                insurance_allowed=(self.dealer_public[0] == 'A')
            )
            # If insurance is pending, do NOT resolve naturals yet
            return s if s.insurance_allowed else self.game._maybe_resolve_naturals(s)

        # Create initial states
        shoe_clone = shoe_after.copy()
        self.start_states["Human"] = deal_player_from(shoe_clone.copy())
        self.start_states["MCTS-Profit"] = deal_player_from(shoe_clone.copy())
        self.start_states["MCTS-Win"] = deal_player_from(shoe_clone.copy())
        self.start_states["Expecti-Win"] = deal_player_from(shoe_clone.copy())

        # Show human starting hand
        hstart = self.start_states["Human"]
        hv, _ = hand_value(hstart.player_cards)
        self.pframes["Human"]['cards'].config(text=f"Cards: {self._format_cards(hstart.player_cards)}")
        self.pframes["Human"]['total'].config(text=f"Total: {hv}")
        self.pframes["Human"]['stack'].config(text=f"Stack: {self.stacks['Human']:.2f}")
        self.pframes["Human"]['actions'].config(text="Actions: (your move)")
        self._append_log(f"Human starting hand: {hstart.player_cards} (Total {hv})")

        # Show AIs starting hands (results hidden until human finishes)
        for tag in ("MCTS-Profit","MCTS-Win","Expecti-Win"):
            s0 = self.start_states[tag]
            v0, _ = hand_value(s0.player_cards)
            self.pframes[tag]['cards'].config(text=f"Cards: {self._format_cards(s0.player_cards)}")
            self.pframes[tag]['total'].config(text=f"Total: {v0}")
            self.pframes[tag]['stack'].config(text=f"Stack: {self.stacks[tag]:.2f}")
            self.pframes[tag]['actions'].config(text="Actions: (pending)")
            self.pframes[tag]['result'].config(text="Result: (pending)")
            self._append_log(f"{tag} starting hand: {s0.player_cards} (Total {v0})")

        # Precompute AI outcomes but don't reveal yet
        self._compute_ai_round_outcomes()

        # Enable human controls depending on insurance phase
        self._refresh_human_controls()

    def _compute_ai_round_outcomes(self):
        # Compute final states & actions for AI players (not shown until human resolves)
        # MCTS-Profit
        s1 = self.start_states["MCTS-Profit"]
        f1, a1 = play_full_hand(self.game, s1, lambda g, s: policy_mcts_profit(g, s, self.iters))
        self.final_states["MCTS-Profit"] = f1
        self.actions_taken["MCTS-Profit"] = a1

        # MCTS-Win
        s2 = self.start_states["MCTS-Win"]
        f2, a2 = play_full_hand(self.game, s2, lambda g, s: policy_mcts_win(g, s, self.iters))
        self.final_states["MCTS-Win"] = f2
        self.actions_taken["MCTS-Win"] = a2

        # Expectiminimax-Win
        s3 = self.start_states["Expecti-Win"]
        f3, a3 = play_full_hand(self.game, s3, lambda g, s: policy_expecti_win(g, s, self.depth))
        self.final_states["Expecti-Win"] = f3
        self.actions_taken["Expecti-Win"] = a3

    def _refresh_human_controls(self):
        s = self.start_states["Human"]
        if self.game.is_terminal(s):
            # Naturals resolved (e.g., blackjack situation after insurance decision)
            self._finalize_round_if_done(human_final=s, human_actions=self.human_actions)
            return
        if s.insurance_allowed:
            self._set_human_buttons(True, insurance_phase=True)
        else:
            can_double = s.can_double
            self._set_human_buttons(True, insurance_phase=False, can_double=can_double)

    # ---------- Human actions ----------
    def on_human_action(self, action: str):
        s = self.start_states["Human"]
        # Validate
        legal = list(self.game.actions(s))
        if action not in legal:
            self._append_log(f"(Illegal or unavailable action: {action})")
            return
        self.human_actions.append(action)

        # Apply
        ns = self.game.result(s, action)
        self.start_states["Human"] = ns

        # Update human UI
        v, _ = hand_value(ns.player_cards)
        self.pframes["Human"]['cards'].config(text=f"Cards: {self._format_cards(ns.player_cards)}")
        self.pframes["Human"]['total'].config(text=f"Total: {v}")
        self.pframes["Human"]['actions'].config(text=f"Actions: {self.human_actions}")

        if ns.insurance_allowed:
            self._append_log("Insurance decision pending...")
        else:
            self._append_log(f"Human action: {action}. Now Total={v}.")

        # If terminal or dealer's turn, resolve human and reveal everything
        if self.game.is_terminal(ns) or ns.to_move == 'Dealer':
            # Ensure dealer plays (in case not already)
            nf = self.game._dealer_play(ns)
            self.start_states["Human"] = nf  # store final in same dict for simplicity
            self._finalize_round_if_done(human_final=nf, human_actions=self.human_actions)
        else:
            # Continue; refresh button set
            self._refresh_human_controls()

    # ---------- Finalization ----------
    def _finalize_round_if_done(self, human_final: BJState, human_actions: List[str]):
        # Reveal dealer
        self.dealer_hole.config(text=f"Hole: {self.game._round_hole}")
        dv, _ = hand_value(self.final_dealer)
        self.dealer_total.config(text=f"Total: {dv}")
        self._append_log(f"Dealer final: {self.final_dealer} (Total {dv})")

        # Settle Human
        if not self.settled_this_round["Human"]:
            delta = self.game.utility_ev(human_final)
            res = "WIN" if delta > 0 else ("LOSS" if delta < 0 else "PUSH")
            self.stacks["Human"] += delta
            self.rec["Human"][res.lower()] += 1
            self.pframes["Human"]['result'].config(text=f"Result: {res} ({delta:+.2f})")
            self.pframes["Human"]['stack'].config(text=f"Stack: {self.stacks['Human']:.2f}")
            self.settled_this_round["Human"] = True
            self._append_log(f"Human final: {human_final.player_cards} -> {res} {delta:+.2f} | Stack={self.stacks['Human']:.2f}")

        # Reveal & settle AI players
        for tag in ("MCTS-Profit","MCTS-Win","Expecti-Win"):
            if not self.settled_this_round[tag]:
                f = self.final_states[tag]
                a = self.actions_taken[tag]
                v, _ = hand_value(f.player_cards)
                delta = self.game.utility_ev(f)
                res = "WIN" if delta > 0 else ("LOSS" if delta < 0 else "PUSH")
                self.stacks[tag] += delta
                self.rec[tag][res.lower()] += 1
                self.pframes[tag]['actions'].config(text=f"Actions: {a}")
                self.pframes[tag]['result'].config(text=f"Result: {res} ({delta:+.2f})")
                self.pframes[tag]['stack'].config(text=f"Stack: {self.stacks[tag]:.2f}")
                self._append_log(f"{tag} final: {f.player_cards} (Total {v}) -> {res} {delta:+.2f} | Stack={self.stacks[tag]:.2f}")
                self.settled_this_round[tag] = True

        # Enable next round, disable human buttons
        self._set_human_buttons(False)
        self.next_btn.state(["!disabled"])

# =========================
# Launch GUI
# =========================

if __name__ == "__main__":
    root = tk.Tk()
    # TTK theme tweaks (optional)
    try:
        from tkinter import ttk
        style = ttk.Style()
        # Use a platform-appropriate theme if available
        style.theme_use(style.theme_use())
    except Exception:
        pass

    app = BlackjackGUI(
        root,
        decks=6,
        base_bet=100,
        starting_chips=1000,
        iters=3000,
        depth=6,
        h17=False  # Dealer stands on all 17
    )
    root.mainloop()
