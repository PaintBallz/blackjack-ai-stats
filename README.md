# Blackjack AI (Stats)

This repo contains a **single-file** Blackjack engine plus three AI agents:

- **MCTS-Profit** – Monte Carlo Tree Search maximizing **expected chips (EV)**
- **MCTS-Win** – Monte Carlo Tree Search maximizing **win probability**
- **Expectiminimax-Win** – Expectiminimax maximizing **win probability**

The program now runs **silently** (no per-round logs), shows a **console loading bar** while playing all hands, then prints **final chip stacks & W-L-P** and renders **one plot** of chip stacks over time at the end.

---

## Features

- **Standard Blackjack rules**
  - Dealer **stands on all 17** (S17).
  - **Aces** count as 11, then demote to 1 as needed to avoid busting.
  - **K/Q/J/10** count as 10.
  - Actions: **HIT**, **STAND**, **DOUBLE** (no splits).
  - **Naturals** (player blackjack) pay **3:2** on the **base bet only** (double does not apply).
  - If the **dealer busts** and the **player ≤ 21**, the player **wins automatically**.

- **Shared dealer per round**
  - All agents in a round face the **same dealer hand** (upcard + committed hole + precomputed hit sequence).
  - The hole card is **hidden** from agents during play.

- **Two distinct MCTS agents**
  - **Profit** rollout is slightly aggressive (may **DOUBLE** on 9–11).
  - **Win** rollout is conservative (never **DOUBLE**); avoids bust risk.
  - Each uses an **independent RNG** so tie situations don’t mirror decisions.

- **Expectiminimax agent**
  - Chance nodes on **HIT** expand using card probabilities from the shoe.
  - Objective is the **win indicator** (+1 / 0 / −1).

- **Configurable tournament**
  - `rounds` (hands to play)
  - `iters` (MCTS iterations per decision)
  - `depth` (Expectiminimax depth)
  - `decks` (shoe size)
  - `starting_chips` (per agent)
  - `base_bet` (static bet; **DOUBLE** uses 2× `base_bet`)
  - `plot` (show end-of-run plot)
  - `save_dir` (optional folder to save PNGs)

- **Clean output**
  - **Console loading bar** during play.
  - **One summary plot** at the end: *Chip Stacks Over Rounds*.
  - **Final chip stacks** and **W-L-P** printed in the console.

---

## Files

- **Single file** (e.g., `blackjack_ai.py`) containing:
  - Core engine (`Blackjack`, `BJState`, rules & payouts)
  - **MCTS-Profit** and **MCTS-Win** implementations
  - **Expectiminimax-Win**
  - Tournament runner `run_comparison(...)`

---

## Requirements

- **Python 3.9+**
- **matplotlib** (for the end-of-run plot)

Install the dependency:

```bash
pip install matplotlib
```

## How to Run
  1. Download the code file.
  2. Run:
  ```bash
  python blackjack_ai.py
  ```

## Configuration

Edit the call at the bottom of the file:

```bash
if __name__ == "__main__":
    run_comparison(
        rounds=25,              # number of hands to play
        iters=3000,             # MCTS iterations per decision
        depth=6,                # Expectiminimax search depth
        decks=6,                # number of decks in the shoe
        starting_chips=1000,    # starting chip count
        base_bet=100            # static bet per hand; DOUBLE uses 2× base_bet
    )
```
## Notes
- The dealer hand is precomputed once per round so every agent faces the same final dealer outcome—apple-to-apples comparisons.