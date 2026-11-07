"""
agent.py - YOUR poker bot. This one file is a complete web server.

Run it:
    python agent.py                  # listens on port 5008
    PORT=5010 python agent.py        # or pick a port

Play a match against a baseline (three windows):
    python agent.py                                 # your bot on port 5008
    python baseline-bots/baseline_random.py         # opponent on port 5009
    python judge_engine.py                          # judge: both halves (uses those two ports)

Check that the tournament can run it:
    python -m pytest -q              # tests in test_agent.py
    ./tools/check_build.sh           # builds and smoke-tests it like the tournament

TODO:
    1. Assign your agent a unique PARTICIPANT and AGENT_NAME below.
    2. Implement the three functions between "YOUR CODE HERE" and "END OF YOUR CODE".

Everything under "SERVER PLUMBING" talks to the judge. You normally never
need to change it, and if you break it the judge can't talk to your bot.
"""

import os
import traceback

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from evaluator import card_rank, card_suit, cards_to_str, omaha_score, warmup  # handy helpers, see evaluator.py
from game import Action, AgentConfig, HandResult, Observation, Street

# Build omaha_score's lookup table now, while the server starts, instead of
# during your first decision (where it would eat into your time bank).
# Do any other slow setup (loading a model, ...) here too: the server must
# answer within 15 seconds of starting.
warmup()

# rename as your team/bot's name
PARTICIPANT = "YAK"
AGENT_NAME = "YAK-Winner"


# ===== YOUR CODE HERE =====

import random
import time
from itertools import combinations
from math import comb

import numpy as np

from evaluator import get_table


_BIN = np.array([[comb(n, k) for k in range(6)] for n in range(36)], dtype=np.int64)
_B2 = _BIN[:, 1:3].copy()
_B3 = _BIN[:, 1:4].copy()
_N_TRIPS = comb(36, 3)
_PI = np.array(list(combinations(range(5), 2)), dtype=np.int64)
_TI = {k: np.array(list(combinations(range(k), 3)), dtype=np.int64) for k in (3, 4, 5)}


def _build_table() -> np.ndarray:
    table = get_table()
    five = np.array(list(combinations(range(36), 5)), dtype=np.int64)
    masks = (np.int64(1) << five).sum(axis=1)
    scores = np.fromiter((table[m] for m in masks.tolist()), dtype=np.int64, count=len(masks))
    ranks = np.zeros(comb(36, 5), dtype=np.uint16)
    ranks[_BIN[five, np.arange(1, 6)].sum(axis=1)] = np.searchsorted(np.unique(scores), scores) + 1

    pairs = np.array(list(combinations(range(36), 2)), dtype=np.int64)
    trips = np.array(list(combinations(range(36), 3)), dtype=np.int64)
    pair_by_id = np.zeros_like(pairs)
    pair_by_id[_BIN[pairs, np.arange(1, 3)].sum(axis=1)] = pairs
    trip_by_id = np.zeros_like(trips)
    trip_by_id[_BIN[trips, np.arange(1, 4)].sum(axis=1)] = trips

    big = np.zeros((len(pairs), _N_TRIPS), dtype=np.uint16)
    cols = np.arange(1, 6)
    for p in range(len(pairs)):
        a, b = pair_by_id[p]
        ok = (trip_by_id != a).all(axis=1) & (trip_by_id != b).all(axis=1)
        cards = np.concatenate([np.broadcast_to(pair_by_id[p], (int(ok.sum()), 2)), trip_by_id[ok]], axis=1)
        cards.sort(axis=1)
        big[p, ok] = ranks[_BIN[cards, cols].sum(axis=1)]
    return big.ravel()


_BIG = _build_table()


def _best_scores(holes: np.ndarray, boards: np.ndarray) -> np.ndarray:
    hp = holes[:, _PI]
    bt = boards[:, _TI[boards.shape[1]]]
    pids = _B2[hp[..., 0], 0] + _B2[hp[..., 1], 1]
    tids = _B3[bt[..., 0], 0] + _B3[bt[..., 1], 1] + _B3[bt[..., 2], 2]
    return _BIG[(pids * _N_TRIPS)[:, :, None] + tids[:, None, :]].reshape(len(holes), -1).max(axis=1)


def _simulate(hole, board, n: int, rng: np.random.Generator):
    hole = sorted(hole)
    board = sorted(board)
    known = set(hole) | set(board)
    rem = np.array([c for c in range(36) if c not in known], dtype=np.int64)
    picks = rem[rng.random((n, len(rem))).argsort(axis=1)[:, :10 - len(board)]]
    opp = np.sort(picks[:, :5], axis=1)
    vis = np.broadcast_to(np.array(board, dtype=np.int64), (n, len(board)))
    full = np.sort(np.concatenate([vis, picks[:, 5:]], axis=1), axis=1)
    mine = _best_scores(np.broadcast_to(np.array(hole, dtype=np.int64), (n, 5)), full)
    theirs = _best_scores(opp, full)
    result = (mine > theirs) + 0.5 * (mine == theirs)
    if len(board) < 3:
        return result, None
    now = theirs if len(board) == 5 else _best_scores(opp, vis)
    pct = np.empty(n)
    pct[now.argsort(kind="stable")] = (np.arange(n) + 1) / n
    return result, pct


def _weigh(sim, opp_bias: float) -> float:
    result, pct = sim
    if pct is None or opp_bias <= 0.01:
        return float(result.mean())
    w = pct ** opp_bias
    return float((result * w).sum() / w.sum())


_SUIT_PATTERNS = ((4, 1), (3, 2), (3, 1, 1), (2, 2, 1), (2, 1, 1, 1))
_WINDOWS = tuple(frozenset(range(s, s + 5)) for s in range(5)) + (frozenset({8, 0, 1, 2, 3}),)
_PF_COEF = (
    -0.02387, -0.01549, -0.00580, 0.00562, 0.02263, 0.02015, 0.01898, 0.01973, 0.01920,
    0.01780, 0.05605, 0.03756, 0.07518, 0.05796,
    0.10263, 0.04161, -0.06496,
    -0.00098, -0.00096, 0.01559,
    -0.00132, 0.00092, 0.07017,
    0.03068, 0.01924,
    0.01223,
)


def _preflop_features(hole) -> list:
    rc = [0] * 9
    by_suit: dict = {}
    for c in hole:
        rc[c >> 2] += 1
        by_suit.setdefault(c & 3, []).append(c >> 2)
    pattern = tuple(sorted((len(v) for v in by_suit.values()), reverse=True))
    tops = [max(v) for v in by_suit.values() if len(v) >= 2]
    distinct = {c >> 2 for c in hole}
    windows = [len(distinct & w) for w in _WINDOWS]
    pair_ranks = [r for r in range(9) if rc[r] >= 2]
    return (rc + [float(pattern == p) for p in _SUIT_PATTERNS]
            + [rc.count(2), rc.count(3), rc.count(4)]
            + [tops.count(8), tops.count(7), sum(tops) / 8.0]
            + [sum(x >= 3 for x in windows), sum(x >= 4 for x in windows), len(distinct)]
            + [sum(pair_ranks) / 8.0, float(rc[8] >= 2), 1.0])


def _preflop_equity(hole) -> float:
    return sum(a * b for a, b in zip(_preflop_features(hole), _PF_COEF))


RAISES = (Action.RAISE_HALF, Action.RAISE_POT)

TIME_BUDGET = 6.0
OVERHEAD_PER_CALL = 0.001
MIN_SAMPLES, MAX_SAMPLES = 40, 400
VALUE_HALF = {Street.PREFLOP: 0.555, Street.FLOP: 0.60, Street.TURN: 0.63, Street.RIVER: 0.66}
VALUE_POT = {Street.PREFLOP: 0.60, Street.FLOP: 0.70, Street.TURN: 0.74, Street.RIVER: 0.80}
CALL_MARGIN = {Street.PREFLOP: 0.02, Street.FLOP: 0.04, Street.TURN: 0.03, Street.RIVER: 0.0}


def new_half(config: AgentConfig, memory: dict) -> None:
    memory.update(
        num_hands=config.num_hands,
        time_used=0.0,
        calls=0,
        mc_runs=0,
        sec_per_sample=5e-6,
        sim=None,
        rng=np.random.default_rng(2026),
        prng=random.Random(2026),
        opp_actions=0,
        opp_raises=0,
        opp_faced=0,
        opp_folds=0,
        pending_raise=False,
        seen=(-1, 0),
    )


def _opp_stats(memory: dict) -> tuple[float, float, float]:
    aggr = (memory["opp_raises"] + 1) / (memory["opp_actions"] + 3)
    fold_rate = (memory["opp_folds"] + 1) / (memory["opp_faced"] + 3)
    scale = min(1.0, max(0.0, (0.6 - aggr) / 0.35))
    return aggr, fold_rate, scale


def _track(obs: Observation, memory: dict) -> None:
    hand, seen = memory["seen"]
    if hand != obs.hand_index:
        seen = 0
    elif memory["pending_raise"]:
        memory["opp_faced"] += 1
        memory["pending_raise"] = False
    for a in obs.action_history[seen:]:
        if not a.is_me:
            memory["opp_actions"] += 1
            if a.action in RAISES:
                memory["opp_raises"] += 1
    memory["seen"] = (obs.hand_index, len(obs.action_history))


def _num_samples(obs: Observation, memory: dict) -> int:
    hands_done = max(1, obs.hand_index)
    hands_left = max(1, memory["num_hands"] - obs.hand_index)
    calls_per_hand = max(2.0, memory["calls"] / hands_done)
    runs_per_hand = max(0.8, memory["mc_runs"] / hands_done)
    spare = TIME_BUDGET - memory["time_used"] - hands_left * calls_per_hand * OVERHEAD_PER_CALL
    if spare <= 0:
        return MIN_SAMPLES
    n = int(spare / (hands_left * runs_per_hand) / memory["sec_per_sample"])
    return max(MIN_SAMPLES, min(MAX_SAMPLES, n))


def _simulation(obs: Observation, memory: dict):
    cached = memory["sim"]
    if cached is not None and cached[0] == obs.hand_index and cached[1] == obs.street:
        return cached[2]
    n = _num_samples(obs, memory)
    t = time.perf_counter()
    sim = _simulate(obs.hole_cards, obs.board, n, memory["rng"])
    memory["sec_per_sample"] = 0.8 * memory["sec_per_sample"] + 0.2 * (time.perf_counter() - t) / n
    memory["mc_runs"] += 1
    memory["sim"] = (obs.hand_index, obs.street, sim)
    return sim


def _pick(obs: Observation, wanted: Action) -> Action:
    legal = obs.legal_actions
    if wanted in legal:
        return wanted
    if wanted in RAISES:
        for a in (Action.RAISE_POT, Action.RAISE_HALF):
            if a in legal:
                return a
    return Action.CHECK_CALL


def _decide(obs: Observation, memory: dict) -> Action:
    street = obs.street
    aggr, fold_rate, scale = _opp_stats(memory)

    opp_r_street = sum(1 for a in obs.action_history if not a.is_me and a.street == street and a.action in RAISES)
    opp_r_before = sum(1 for a in obs.action_history if not a.is_me and a.street < street and a.action in RAISES)

    if street == Street.PREFLOP:
        r = opp_r_street
        eq = _preflop_equity(obs.hole_cards) - scale * (0.035 * r + 0.015 * r * r)
    else:
        bias = scale * (2.0 * opp_r_street + 0.6 * opp_r_before)
        eq = _weigh(_simulation(obs, memory), bias)

    thin = 0.04 if (memory["opp_faced"] >= 10 and fold_rate < 0.15) else 0.0
    v_half = VALUE_HALF[street] - thin
    v_pot = VALUE_POT[street] - thin

    if obs.to_call == 0:
        if eq >= v_pot:
            return _pick(obs, Action.RAISE_POT)
        if eq >= v_half:
            return _pick(obs, Action.RAISE_HALF)
        if street != Street.PREFLOP and memory["opp_faced"] >= 15 and fold_rate > 0.4:
            p = 0.35 if eq < 0.35 else (0.25 if street == Street.FLOP and eq > 0.45 else 0.0)
            if p and memory["prng"].random() < p:
                return _pick(obs, Action.RAISE_HALF)
        return Action.CHECK_CALL

    pot_odds = obs.to_call / (obs.pot + obs.to_call)
    all_in_call = obs.to_call >= obs.my_stack
    margin = 0.0 if all_in_call else CALL_MARGIN[street]

    if eq >= v_pot + 0.03 * obs.raises_this_street and any(a in obs.legal_actions for a in RAISES):
        return _pick(obs, Action.RAISE_POT)
    if eq >= pot_odds + margin:
        return Action.CHECK_CALL
    return Action.FOLD


def choose_action(obs: Observation, memory: dict) -> Action:
    t = time.perf_counter()
    try:
        if "time_used" not in memory:
            new_half(AgentConfig(), memory)
        _track(obs, memory)
        action = _decide(obs, memory)
        if action not in obs.legal_actions:
            action = Action.CHECK_CALL
    except Exception:
        action = Action.CHECK_CALL
    if action in RAISES:
        memory["pending_raise"] = True
    memory["calls"] = memory.get("calls", 0) + 1
    memory["time_used"] = memory.get("time_used", 0.0) + time.perf_counter() - t + OVERHEAD_PER_CALL
    return action


def on_hand_end(result: HandResult, memory: dict) -> None:
    try:
        if memory.get("pending_raise"):
            memory["opp_faced"] = memory.get("opp_faced", 0) + 1
            if not result.went_to_showdown and result.my_net > 0:
                memory["opp_folds"] = memory.get("opp_folds", 0) + 1
            memory["pending_raise"] = False
    except Exception:
        pass

# ===== END OF YOUR CODE =====


# ===== SERVER PLUMBING (you normally don't need to edit below) =====
# The judge calls these routes:
#   GET  /           -> who you are (also used as the health check)
#   POST /new-half   -> start a fresh half: memory is wiped
#   POST /act        -> the judge sends any finished hands plus an Observation;
#                       on_hand_end() runs for each hand, then you answer an action
#   POST /hand-end   -> the judge sends the last finished hands of a half
#
# Why "async def"? The routes run on one thread and call your functions
# directly, so your functions never run twice at once. Your own functions
# stay normal "def" (don't make them async).
#
# Keep your per-half state in `memory`. In the tournament every half runs in
# a fresh server process, so everything starts over. When you test locally,
# though, one process can play both halves and only `memory` is wiped, NOT
# other global variables. Using `memory` makes your bot behave the same way
# in both places.

app = FastAPI()
MEMORY: dict = {}                                   # your memory for this half
STATUS = {"started": False}                         # has /new-half been called?
TOKEN = os.environ.get("AGENT_TOKEN") or None       # set by the tournament


async def _read_request(request: Request, path: str) -> tuple[dict, JSONResponse | None]:
    """Check the token (and that the half has started) BEFORE reading the body.

    Checking first means a request without the right token costs your bot
    almost nothing, so nobody else can slow it down by sending junk.
    Returns (body, None) when the request is fine, or ({}, error response).
    """
    if TOKEN and request.headers.get("x-agent-token") != TOKEN:
        return {}, JSONResponse({"error": "bad or missing X-Agent-Token"}, status_code=401)
    if not STATUS["started"] and path != "/new-half":
        return {}, JSONResponse({"error": "call /new-half first"}, status_code=409)
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return {}, JSONResponse({"error": "the request body must be a JSON object"}, status_code=400)
    return body, None


def _error_info(exc: Exception) -> dict:
    """Describe an exception from your code (call this inside `except`)."""
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _crash(exc: Exception, hand_end_errors: list | None = None) -> JSONResponse:
    """Report an exception from your code to the judge (with traceback)."""
    return JSONResponse({**_error_info(exc), "hand_end_errors": hand_end_errors or []},
                        status_code=500)


def _deliver_results(raw_results) -> list:
    """Call on_hand_end() for each finished hand the judge sent.
    Returns a list of error reports (one per hand that crashed)."""
    errors = []
    for raw in raw_results or []:
        try:
            on_hand_end(HandResult.from_dict(raw), MEMORY)
        except Exception as exc:
            errors.append(_error_info(exc))
    return errors


@app.get("/")
async def info():
    """Health check: tells the judge who this bot is."""
    return {"participant": PARTICIPANT, "agent_name": AGENT_NAME}


@app.post("/new-half")
async def route_new_half(request: Request):
    """Wipe memory and call your new_half(). The body is {"config": {...}}."""
    body, refused = await _read_request(request, "/new-half")
    if refused:
        return refused
    MEMORY.clear()
    STATUS["started"] = True
    try:
        new_half(AgentConfig.from_dict(body.get("config", {})), MEMORY)
    except Exception as exc:
        return _crash(exc)
    return {"ok": True}


@app.post("/act")
async def route_act(request: Request):
    """Deliver finished hands, then turn the JSON into an Observation, call
    choose_action(), and send the answer back."""
    body, refused = await _read_request(request, "/act")
    if refused:
        return refused
    hand_end_errors = _deliver_results(body.get("hand_results"))
    try:
        obs = Observation.from_dict(body["observation"])
        action = choose_action(obs, MEMORY)
    except Exception as exc:
        return _crash(exc, hand_end_errors)
    # Send the action's name; anything strange is passed on for the judge to reject.
    if isinstance(action, Action):
        answer = action.name
    elif action is None or isinstance(action, (str, int, float, bool)):
        answer = action
    else:
        answer = repr(action)
    return {"action": answer, "hand_end_errors": hand_end_errors}


@app.post("/hand-end")
async def route_hand_end(request: Request):
    """Call on_hand_end() for the last finished hands of a half."""
    body, refused = await _read_request(request, "/hand-end")
    if refused:
        return refused
    return {"ok": True, "hand_end_errors": _deliver_results(body.get("results"))}


if __name__ == "__main__":
    # The tournament sets PORT as an environment variable, so each bot is compliant to our evaluation platform
    # Locally it defaults to 5008 for your own testing convenience
    port = int(os.environ.get("PORT", "5008"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)
