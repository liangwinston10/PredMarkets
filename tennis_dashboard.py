"""
Tennis Betting Dashboard
Streamlit app combining tournament schedule, Kalshi markets, and model edge analysis.
Run with: streamlit run tennis_dashboard.py
"""

import sys
import os

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Ensure tools/ is on the path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from tools.tennis_schedule import get_upcoming_schedule, get_live_tournaments
from tools.kalshi_markets  import (
    get_tennis_markets, search_player_markets, enrich_market,
    parse_implied_prob, parse_volume, API_KEY,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tennis Edge Dashboard",
    page_icon="🎾",
    layout="wide",
)

st.title("🎾 Tennis Edge Dashboard")

# ── Cached model loaders (expensive HTTP fetches, refresh hourly) ──────────────
@st.cache_resource(ttl=3600, show_spinner="Loading ATP model data...")
def get_atp_engine_and_stats():
    from tools.tennis_predict import load_engine, load_stats
    engine = load_engine()
    stats  = load_stats(engine)
    return engine, stats


@st.cache_resource(ttl=3600, show_spinner="Loading WTA model data...")
def get_wta_engine_and_stats():
    from tools.wta_predict import load_engine, load_stats
    engine = load_engine()
    stats  = load_stats(engine)
    return engine, stats


def _cached_form(player_name: str):
    """Fetch recent form for a player. Module-level ESPN cache persists between Streamlit reruns."""
    from tools.player_form import _load_cache, fetch_recent_form
    _load_cache()   # no-op after first call (checks _cache is not None)
    return fetch_recent_form(player_name)


def _form_blend_dash(comp_p1: float, form1, form2):
    """80/20 blend of composite toward recent win-rate ratio. Returns None if form unavailable."""
    if form1 is None or form2 is None:
        return None
    wr1 = form1.get("win_rate_30d") if form1.get("n_30d", 0) >= 3 else form1.get("win_rate_60d")
    wr2 = form2.get("win_rate_30d") if form2.get("n_30d", 0) >= 3 else form2.get("win_rate_60d")
    if wr1 is None or wr2 is None:
        return None
    c1 = max(0.10, min(0.90, wr1))
    c2 = max(0.10, min(0.90, wr2))
    return max(0.001, min(0.999, 0.80 * comp_p1 + 0.20 * (c1 / (c1 + c2))))


# ── Live tournament banner ─────────────────────────────────────────────────────
live_now = get_live_tournaments()
if live_now:
    names = ", ".join(t["name"] for t in live_now)
    st.success(f"**Live now:** {names}")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📅 Schedule", "📊 Kalshi Markets", "⚡ Edge Analysis"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SCHEDULE
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Upcoming ATP/WTA 1000 + Grand Slams")

    col_f1, col_f2, col_f3 = st.columns([2, 2, 2])
    with col_f1:
        tour_filter = st.radio("Tour", ["All", "ATP", "WTA", "ATP/WTA"], horizontal=True)
    with col_f2:
        days_ahead = st.slider("Look-ahead (days)", min_value=14, max_value=365, value=120, step=14)
    with col_f3:
        show_completed = st.checkbox("Show completed tournaments", value=False)

    df_sched = get_upcoming_schedule(days_ahead=days_ahead, include_completed=show_completed)

    if tour_filter != "All":
        df_sched = df_sched[df_sched["tour"].str.contains(tour_filter, case=False, na=False)]

    if df_sched.empty:
        st.info("No tournaments found for the selected filters.")
    else:
        # Format for display
        display = df_sched[[
            "name", "tour", "start_date", "end_date",
            "surface", "category", "location", "draw_size",
            "prize_money_usd", "status",
        ]].copy()
        display["start_date"] = display["start_date"].dt.strftime("%b %d")
        display["end_date"]   = display["end_date"].dt.strftime("%b %d, %Y")
        display["prize_money_usd"] = display["prize_money_usd"].apply(lambda x: f"${x:,.0f}")
        display = display.rename(columns={
            "name":            "Tournament",
            "tour":            "Tour",
            "start_date":      "Starts",
            "end_date":        "Ends",
            "surface":         "Surface",
            "category":        "Category",
            "location":        "Location",
            "draw_size":       "Draw",
            "prize_money_usd": "Prize Money",
            "status":          "Status",
        })

        def highlight_status(row):
            if row["Status"] == "Live":
                return ["background-color: #1a4a1a"] * len(row)
            return [""] * len(row)

        styled = display.style.apply(highlight_status, axis=1)
        st.dataframe(styled, width='stretch', hide_index=True)
        st.caption(f"Showing {len(df_sched)} events. 2026 calendar — dates may shift slightly as season progresses.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — KALSHI MARKETS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Kalshi Tennis Prediction Markets")

    if not API_KEY:
        st.info(
            "Running in **unauthenticated** mode. "
            "Add `KALSHI_API_KEY=your_key` to `.env` to unlock trade history and full market depth.",
            icon="ℹ️",
        )

    col_s1, col_s2, col_s3 = st.columns([3, 1, 1])
    with col_s1:
        search_query = st.text_input("Search by player or tournament name", placeholder="e.g. Sinner, Wimbledon...")
    with col_s2:
        mkt_status = st.selectbox("Status", ["open", "closed", "all"])
    with col_s3:
        refresh_markets = st.button("Refresh", use_container_width=False)

    # Fetch markets (cached unless user hits Refresh)
    cache_key = f"kalshi_{mkt_status}_{search_query}"
    if refresh_markets or cache_key not in st.session_state:
        if search_query.strip():
            markets, err = search_player_markets(search_query, status=mkt_status)
        else:
            markets, err = get_tennis_markets(status=mkt_status)
        st.session_state[cache_key] = (markets, err)
    else:
        markets, err = st.session_state[cache_key]

    if err:
        st.warning(f"Kalshi API: {err}")
        markets = []

    if not markets:
        if not err:
            st.info("No markets found. Try a different search or check back when a tournament is active.")
    else:
        enriched = [enrich_market(m) for m in markets]

        # ── Auto-edge: H2H only, fav side (prob>50%), tournament filter ──────────
        st.subheader("⚡ Auto Edge — H2H Match Markets")

        import re, difflib as _difflib
        from tools.tennis_predict import run_prediction as _atp_predict
        from tools.wta_predict    import run_prediction as _wta_predict

        def _extract_tourney(mk: dict) -> str:
            rules = mk.get("rules_primary", "")
            hit = re.search(r'in the \d{4} (.+?) (?:Round|Final|Semi|Quarter|Match|after)', rules)
            return hit.group(1).strip() if hit else "Other"

        def _surface(mk: dict) -> str:
            keys = " ".join([mk.get("rules_primary",""), mk.get("event_ticker",""), mk.get("ticker","")]).upper()
            for kw in ["WIMBLEDON","WIMB","WMEN","WWOM","GRASS","QUEENS","HALLE","EASTBOURNE"]:
                if kw in keys: return "Grass"
            for kw in ["MONTE CARLO","MONTE-CARLO","MADRID","ROME","ITALIAN","ROLAND","CLAY",
                       "BARCELONA","HAMBURG","ESTORIL","BUCHAREST","MARRAKECH","LINZ"]:
                if kw in keys: return "Clay"
            return "Hard"

        def _best_of(mk: dict) -> int:
            keys = (mk.get("rules_primary","") + mk.get("event_ticker","")).upper()
            return 5 if any(kw in keys for kw in ["GRAND SLAM","ROLAND","WIMBLEDON","US OPEN","AUSTRALIAN"]) else 3

        def _is_wta(mk: dict) -> bool:
            return "WTA" in (mk.get("ticker","") + mk.get("title","")).upper()

        def _find_player(name: str, known: list):
            if not name or not known: return None
            hits = _difflib.get_close_matches(name, known, n=1, cutoff=0.45)
            if hits: return hits[0]
            last = name.split()[-1].lower()
            hits = [p for p in known if p.lower().endswith(last)]
            return hits[0] if hits else None

        # Only H2H match markets (title contains "vs"), favourite side only (prob > 50%)
        h2h_all = [
            m for m in enriched
            if m.get("yes_sub_title")
            and " vs " in m.get("title", "")
            and (m.get("implied_prob") or 0) > 0.5
        ]

        if not h2h_all:
            st.info("No H2H match markets with prob >50% found yet.")
        else:
            # Tournament filter (multi-select, all selected by default)
            tourneys = sorted(set(_extract_tourney(m) for m in h2h_all))
            tourney_pick = st.multiselect(
                "Filter by tournament", options=tourneys,
                default=tourneys, key="ae_tourney"
            )
            h2h = [m for m in h2h_all if _extract_tourney(m) in tourney_pick] if tourney_pick else h2h_all

            # Load engines (already cached)
            atp_eng, atp_sts, wta_eng, wta_sts = None, None, None, None
            _eng_err = []
            try: atp_eng, atp_sts = get_atp_engine_and_stats()
            except Exception as _e: _eng_err.append(f"ATP engine: {_e}")
            try: wta_eng, wta_sts = get_wta_engine_and_stats()
            except Exception as _e: _eng_err.append(f"WTA engine: {_e}")
            if _eng_err:
                st.warning(" | ".join(_eng_err))

            import sys as _sys
            _sz_dir = os.path.join(REPO_ROOT, "tennis_model")
            if _sz_dir not in _sys.path:
                _sys.path.insert(0, _sz_dir)
            from sizing import size_day

            def _infer_round(mk: dict) -> str:
                rules = mk.get("rules_primary", "").upper()
                if "ROUND OF 128" in rules: return "R128"
                if "ROUND OF 64"  in rules: return "R64"
                if "ROUND OF 32"  in rules: return "R32"
                if "ROUND OF 16"  in rules: return "R16"
                if "QUARTER"      in rules: return "QF"
                if "SEMI"         in rules: return "SF"
                if "FINAL"        in rules: return "F"
                return "R32"

            # ── Bankroll input ─────────────────────────────────────────────────
            bankroll = st.number_input(
                "Bankroll ($)", min_value=10.0, max_value=1_000_000.0,
                value=float(st.session_state.get("bankroll", 1000.0)),
                step=100.0, format="%.2f", key="bankroll_input",
            )
            st.session_state["bankroll"] = bankroll

            # ── Build edge rows ────────────────────────────────────────────────
            edge_rows    = []
            sizing_feed  = []   # VALUE bets — buy Yes
            reverse_feed = []   # REVERSE bets — buy No
            _sized_events = set()  # deduplicate: each event_ticker in at most one feed

            # Build event→[markets] index for opponent lookup
            _ev_idx: dict = {}
            for _em in enriched:
                _etk = _em.get("event_ticker", "")
                if _etk:
                    _ev_idx.setdefault(_etk, []).append(_em)

            for m in h2h:
                fav   = m.get("yes_sub_title", "")
                title = m.get("title", "")
                # Best source: companion market in same event where yes_sub_title != fav
                _ev_tk = m.get("event_ticker", "")
                _comp  = next((e for e in _ev_idx.get(_ev_tk, [])
                               if e.get("yes_sub_title", "").strip() != fav.strip()
                               and e.get("yes_sub_title", "")), None)
                if _comp:
                    dog = _comp["yes_sub_title"]
                else:
                    # Fallback: regex on title (gives last name only)
                    _vs = re.search(r'the (.+?) vs (.+?) (?::|match)', title, re.IGNORECASE)
                    if _vs:
                        _a, _b = _vs.group(1).strip(), _vs.group(2).strip()
                        dog = _b if fav.split()[-1].lower() in _a.lower() else _a
                        if dog == fav:
                            dog = _b if _a == fav else _a
                    else:
                        dog = m.get("no_sub_title", "")

                mkt_prob = m.get("implied_prob")
                surf     = _surface(m)
                bo       = _best_of(m)
                wta      = _is_wta(m)
                tourney  = _extract_tourney(m)
                rnd      = _infer_round(m)

                eng = wta_eng if wta else atp_eng
                sts = wta_sts if wta else atp_sts

                comp_val         = None
                edge_val         = None
                model_str        = "—"
                edge_str         = "—"
                sim_str          = "—"
                sim_edge_str     = "—"
                form_adj_str     = "—"
                form_adj_edge_str = "—"
                signal           = "—"
                _form_fav        = None
                _form_dog        = None

                if eng and sts and mkt_prob is not None:
                    known = sts.get("known_players", [])
                    p1 = _find_player(fav, known)
                    p2 = _find_player(dog, known)
                    if p1 and p2 and p1 != p2:
                        try:
                            res       = _wta_predict(p1, p2, surf, mkt_prob, eng, sts) if wta else \
                                        _atp_predict(p1, p2, surf, bo, mkt_prob, eng, sts)
                            comp_val  = res["comp_p1"]
                            sim_val   = res.get("sim_p1")
                            edge_val  = comp_val - mkt_prob
                            sim_edge  = (sim_val - mkt_prob) if sim_val is not None else None
                            model_str = f"{comp_val*100:.1f}%"
                            sim_str   = f"{sim_val*100:.1f}%" if sim_val is not None else "—"
                            edge_str  = f"{edge_val*100:+.1f}%"
                            sim_edge_str = f"{sim_edge*100:+.1f}%" if sim_edge is not None else "—"

                            # Form adjustment (ESPN velocity, 20% weight)
                            _form_fav = _cached_form(p1)
                            _form_dog = _cached_form(p2)
                            _fadj_val = _form_blend_dash(comp_val, _form_fav, _form_dog)
                            if _fadj_val is not None:
                                form_adj_str = f"{_fadj_val*100:.1f}%"
                                form_adj_edge_str = f"{(_fadj_val - mkt_prob)*100:+.1f}%"

                            _fadj_edge = (_fadj_val - mkt_prob) if _fadj_val is not None else None
                            if _fadj_edge is not None and ((edge_val > 0 and _fadj_edge < 0) or (edge_val < 0 and _fadj_edge > 0)):
                                signal = "CONFLICT"
                            elif comp_val < 0.5:
                                signal = "REVERSE"
                            elif edge_val > 0:
                                signal = "VALUE"
                            else:
                                signal = "~"
                            if signal != "REVERSE" and edge_val is not None and edge_val >= 0.04 \
                                    and _ev_tk not in _sized_events:
                                _sized_events.add(_ev_tk)
                                sizing_feed.append({
                                    "match_id":   f"{fav} vs {dog}",
                                    "favourite":  fav,
                                    "underdog":   dog,
                                    "tourney":    tourney,
                                    "surface":    surf,
                                    "round":      rnd,
                                    "p_model":    comp_val,
                                    "p_market":   mkt_prob,
                                    "vol":        int(parse_volume(m)),
                                    "form_fav":   _form_fav,
                                    "form_dog":   _form_dog,
                                    "sim_val":    sim_val,
                                    "fadj_val":   _fadj_val,
                                })
                            if signal == "REVERSE" and comp_val is not None and mkt_prob is not None \
                                    and _ev_tk not in _sized_events:
                                _sized_events.add(_ev_tk)
                                reverse_feed.append({
                                    "match_id":   f"{dog} vs {fav} (No)",
                                    "favourite":  dog,
                                    "underdog":   fav,
                                    "tourney":    tourney,
                                    "surface":    surf,
                                    "round":      rnd,
                                    "p_model":    1 - comp_val,
                                    "p_market":   1 - mkt_prob,
                                    "vol":        int(parse_volume(m)),
                                    "form_fav":   _form_dog,
                                    "form_dog":   _form_fav,
                                    "sim_val":    (1 - sim_val) if sim_val is not None else None,
                                    "fadj_val":   (1 - _fadj_val) if _fadj_val is not None else None,
                                })
                        except Exception as ex:
                            edge_str = f"err: {ex}"

                edge_rows.append({
                    "Tournament":      tourney,
                    "Favourite":       fav,
                    "Underdog":        dog,
                    "Surface":         surf,
                    "Mkt":             f"{mkt_prob*100:.0f}%" if mkt_prob else "—",
                    "Comp":            model_str,
                    "Comp Edge":       edge_str,
                    "Sim":             sim_str,
                    "Sim Edge":        sim_edge_str,
                    "Form-adj":        form_adj_str,
                    "Form-adj Edge":   form_adj_edge_str,
                    "Signal":          signal,
                    "Vol":             int(parse_volume(m)),
                })

            # ── All markets table ──────────────────────────────────────────────
            show_value_only = st.checkbox("Show signals only (VALUE / REVERSE / CONFLICT)", value=False, key="ae_val_only")
            df_edge = pd.DataFrame(edge_rows).sort_values("Vol", ascending=False)
            if show_value_only:
                df_edge = df_edge[df_edge["Signal"].isin(["VALUE", "REVERSE", "CONFLICT"])]

            def _hl_edge(row):
                if row["Signal"] == "VALUE":
                    return ["background-color: #00c853; color: #000000"] * len(row)
                if row["Signal"] == "REVERSE":
                    return ["background-color: #fff176; color: #000000"] * len(row)
                if row["Signal"] == "CONFLICT":
                    return ["background-color: #ff1744; color: #000000"] * len(row)
                return [""] * len(row)

            st.dataframe(df_edge.style.apply(_hl_edge, axis=1), width='stretch', hide_index=True)

            # ── Bet sizing table (VALUE + positive edge) ──────────────────────
            if sizing_feed:
                st.subheader("💰 Bet Sizing")
                st.caption(f"Bankroll: ${bankroll:,.2f}  |  ⅛ Kelly  |  Edge-proportional cap  |  Daily exposure cap by round")

                sized = size_day(sizing_feed, bankroll=bankroll, round_stage=sizing_feed[0]["round"])

                bet_rows = []
                total_stake = 0.0
                for r in sized:
                    stake = r["stake"]
                    total_stake += stake
                    pct   = r["stake_pct"]
                    edge  = r["edge"]
                    # Kalshi payout: buy Yes at p, win (1-p)/p per $ staked if correct
                    p_mkt = r["p_market"]
                    payout = stake * (1 - p_mkt) / p_mkt if (stake > 0 and p_mkt > 0) else 0.0

                    _sv  = r.get("sim_val")
                    _fv  = r.get("fadj_val")
                    bet_rows.append({
                        "Favourite":    r["favourite"],
                        "Underdog":     r["underdog"],
                        "Tournament":   r["tourney"],
                        "Surface":      r["surface"],
                        "Mkt":          f"{p_mkt*100:.0f}%",
                        "Comp":         f"{r['p_model']*100:.1f}%",
                        "Sim":          f"{_sv*100:.1f}%" if _sv is not None else "—",
                        "Form-adj":     f"{_fv*100:.1f}%" if _fv is not None else "—",
                        "Edge":         f"{edge*100:.1f}%",
                        "Stake ($)":    f"${stake:,.2f}" if stake > 0 else "—",
                        "Payout ($)":   f"${payout:,.2f}" if payout > 0 else "—",
                        "Signal":       r["signal"],
                        "Vol":          r["vol"],
                        "_signal_ord":  0 if r["signal"] == "BET" else 1,
                        "_edge_val":    edge,
                    })

                df_bets = (
                    pd.DataFrame(bet_rows)
                    .sort_values(["_signal_ord", "_edge_val"], ascending=[True, False])
                    .drop(columns=["_signal_ord", "_edge_val"])
                )

                # Highlight BET rows
                def _hl(row):
                    if row["Signal"] == "BET":
                        return ["background-color: #00c853; color: #000000"] * len(row)
                    if row["Signal"] == "CAP_BOUND":
                        return ["background-color: #ff6d00; color: #000000"] * len(row)
                    if row["Signal"] == "CONFLICT":
                        return ["background-color: #b71c1c; color: #ffffff"] * len(row)
                    return [""] * len(row)

                st.dataframe(df_bets.style.apply(_hl, axis=1), width='stretch', hide_index=True)

                ec1, ec2, ec3 = st.columns(3)
                with ec1:
                    bets_placed = sum(1 for r in sized if r["signal"] == "BET")
                    st.metric("Actionable bets", bets_placed)
                with ec2:
                    st.metric("Total stake", f"${total_stake:,.2f}")
                with ec3:
                    st.metric("% of bankroll", f"{total_stake/bankroll*100:.1f}%")

                # ── Velocity breakdown for green-zone BET rows ─────────────────
                bet_rows_with_form = [r for r in sized if r["signal"] == "BET"
                                      and (r.get("form_fav") or r.get("form_dog"))]
                if bet_rows_with_form:
                    st.markdown("#### Recent Form — Green Zone")
                    for r in bet_rows_with_form:
                        f_fav = r.get("form_fav")
                        f_dog = r.get("form_dog")
                        with st.expander(f"{r['favourite']} vs {r['underdog']}  ·  {r['surface']}  ·  {r['tourney']}"):
                            fc_l, fc_r = st.columns(2)
                            for col, pname, fd in [(fc_l, r["favourite"], f_fav), (fc_r, r["underdog"], f_dog)]:
                                with col:
                                    st.markdown(f"**{pname}**")
                                    if fd:
                                        results_seq = fd.get("results", [])[:7]
                                        badges = "  ".join(
                                            f":green[**W**]" if x == "W" else f":red[**L**]"
                                            for x in results_seq
                                        )
                                        st.markdown(badges)
                                        wr30 = fd.get("win_rate_30d")
                                        wr60 = fd.get("win_rate_60d")
                                        n30  = fd.get("n_30d", 0)
                                        n60  = fd.get("n_60d", 0)
                                        streak = fd.get("streak", "—")
                                        days   = fd.get("days_since_last", "?")
                                        wr30_str = f"{wr30*100:.0f}% ({n30}m)" if wr30 is not None else "—"
                                        wr60_str = f"{wr60*100:.0f}% ({n60}m)" if wr60 is not None else "—"
                                        st.caption(
                                            f"30d: **{wr30_str}**  ·  60d: {wr60_str}  "
                                            f"·  Streak: **{streak}**  ·  Last match: {days}d ago"
                                        )
                                    else:
                                        st.caption("No form data")

            if not sizing_feed:
                st.info("No positive-edge H2H markets found yet. Check back when markets are active.")

            # ── Reverse bet sizing (REVERSE signal — buy No on market fav) ────
            if reverse_feed:
                st.subheader("🟡 Reverse Sizing")
                st.caption(f"Bankroll: ${bankroll:,.2f}  |  1/16 Kelly (conservative)  |  Buying No on market favourite")

                rev_sized = size_day(reverse_feed, bankroll=bankroll,
                                     round_stage=reverse_feed[0]["round"])
                # Apply 1/16 Kelly conservatively by halving the ⅛ Kelly stakes
                for _r in rev_sized:
                    _r["stake"] = round(_r["stake"] * 0.5, 2)
                    _r["stake_pct"] = _r["stake_pct"] * 0.5

                rev_rows = []
                rev_total = 0.0
                for r in rev_sized:
                    stake = r["stake"]
                    rev_total += stake
                    p_mkt = r["p_market"]   # No price = 1 - original mkt_prob
                    payout = stake * (1 - p_mkt) / p_mkt if (stake > 0 and p_mkt > 0) else 0.0
                    _sv  = r.get("sim_val")
                    _fv  = r.get("fadj_val")
                    rev_rows.append({
                        "Bet On":       r["favourite"],
                        "Against":      r["underdog"],
                        "Tournament":   r["tourney"],
                        "Surface":      r["surface"],
                        "No Price":     f"{p_mkt*100:.0f}%",
                        "Comp (dog)":   f"{r['p_model']*100:.1f}%",
                        "Sim (dog)":    f"{_sv*100:.1f}%" if _sv is not None else "—",
                        "Form-adj (dog)": f"{_fv*100:.1f}%" if _fv is not None else "—",
                        "Edge":         f"{r['edge']*100:.1f}%",
                        "Stake ($)":    f"${stake:,.2f}" if stake > 0 else "—",
                        "Payout ($)":   f"${payout:,.2f}" if payout > 0 else "—",
                        "Signal":       r["signal"],
                        "Vol":          r["vol"],
                        "_signal_ord":  0 if r["signal"] == "BET" else 1,
                        "_edge_val":    r["edge"],
                    })

                df_rev = (
                    pd.DataFrame(rev_rows)
                    .sort_values(["_signal_ord", "_edge_val"], ascending=[True, False])
                    .drop(columns=["_signal_ord", "_edge_val"])
                )

                def _hl_rev(row):
                    if row["Signal"] == "BET":
                        return ["background-color: #fff176; color: #000000"] * len(row)
                    if row["Signal"] == "CAP_BOUND":
                        return ["background-color: #ff6d00; color: #000000"] * len(row)
                    return [""] * len(row)

                st.dataframe(df_rev.style.apply(_hl_rev, axis=1), width='stretch', hide_index=True)

                rc1, rc2, rc3 = st.columns(3)
                with rc1:
                    st.metric("Actionable reverses", sum(1 for r in rev_sized if r["signal"] == "BET"))
                with rc2:
                    st.metric("Total stake", f"${rev_total:,.2f}")
                with rc3:
                    st.metric("% of bankroll", f"{rev_total/bankroll*100:.1f}%")

                rev_bet_with_form = [r for r in rev_sized if r["signal"] == "BET"
                                     and (r.get("form_fav") or r.get("form_dog"))]
                if rev_bet_with_form:
                    st.markdown("#### Recent Form — Reverse Zone")
                    for r in rev_bet_with_form:
                        with st.expander(f"{r['favourite']} (No on {r['underdog']})  ·  {r['surface']}  ·  {r['tourney']}"):
                            fc_l, fc_r = st.columns(2)
                            for col, pname, fd in [(fc_l, r["favourite"], r.get("form_fav")),
                                                    (fc_r, r["underdog"],  r.get("form_dog"))]:
                                with col:
                                    st.markdown(f"**{pname}**")
                                    if fd:
                                        badges = "  ".join(":green[**W**]" if x == "W" else ":red[**L**]"
                                                           for x in fd.get("results", [])[:7])
                                        st.markdown(badges)
                                        wr30 = fd.get("win_rate_30d"); n30 = fd.get("n_30d", 0)
                                        wr60 = fd.get("win_rate_60d"); n60 = fd.get("n_60d", 0)
                                        wr30_s = f"{wr30*100:.0f}% ({n30}m)" if wr30 is not None else "—"
                                        wr60_s = f"{wr60*100:.0f}% ({n60}m)" if wr60 is not None else "—"
                                        st.caption(f"30d: **{wr30_s}**  ·  60d: {wr60_s}  "
                                                   f"·  Streak: **{fd.get('streak','—')}**  "
                                                   f"·  Last: {fd.get('days_since_last','?')}d ago")
                                    else:
                                        st.caption("No form data")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EDGE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Model Edge Analysis")

    tour_choice = st.radio("Tour", ["ATP", "WTA"], horizontal=True)

    # Load the appropriate engine (lazy, cached)
    engine_loaded = False
    load_error    = None
    try:
        if tour_choice == "ATP":
            engine, stats = get_atp_engine_and_stats()
        else:
            engine, stats = get_wta_engine_and_stats()
        engine_loaded = True
    except Exception as e:
        load_error = str(e)

    if load_error:
        st.error(f"Failed to load model: {load_error}")
        st.stop()

    known_players = stats["known_players"]

    # ── Input form ─────────────────────────────────────────────────────────────
    with st.form("prediction_form"):
        fc1, fc2 = st.columns(2)

        with fc1:
            p1_search = st.text_input("Search Player 1", placeholder="Type to filter...")
            p1_candidates = (
                [p for p in known_players if p1_search.lower() in p.lower()]
                if p1_search else known_players
            )
            p1_name = st.selectbox("Player 1", p1_candidates, key="p1_select") if p1_candidates else None

        with fc2:
            p2_search = st.text_input("Search Player 2", placeholder="Type to filter...")
            p2_candidates = (
                [p for p in known_players if p2_search.lower() in p.lower()]
                if p2_search else known_players
            )
            p2_name = st.selectbox("Player 2", p2_candidates, key="p2_select") if p2_candidates else None

        fc3, fc4, fc5 = st.columns([2, 2, 3])

        with fc3:
            surface = st.selectbox("Surface", ["Hard", "Clay", "Grass"])

        with fc4:
            if tour_choice == "ATP":
                best_of = st.radio("Best of", [3, 5], horizontal=True)
            else:
                st.markdown("**Best of**")
                st.markdown("Bo3 *(WTA)*")
                best_of = 3

        with fc5:
            use_market = st.checkbox("Include market odds")
            prefilled  = st.session_state.get("prefill_market_prob")
            if use_market:
                default_val = float(prefilled) if prefilled else 0.50
                market_p1 = st.number_input(
                    "Market implied prob for Player 1 (0–1)",
                    min_value=0.01, max_value=0.99,
                    value=default_val, step=0.01,
                    format="%.2f",
                )
                if prefilled and st.session_state.get("prefill_ticker"):
                    st.caption(f"Prefilled from Kalshi: {st.session_state['prefill_ticker']}")
            else:
                market_p1 = None

        submitted = st.form_submit_button("Run Prediction", width='stretch', type="primary")

    # ── Results ────────────────────────────────────────────────────────────────
    if submitted:
        if not p1_name or not p2_name:
            st.warning("Please select both players.")
        elif p1_name == p2_name:
            st.warning("Player 1 and Player 2 must be different.")
        else:
            with st.spinner("Running model..."):
                try:
                    if tour_choice == "ATP":
                        from tools.tennis_predict import run_prediction
                        result = run_prediction(p1_name, p2_name, surface, best_of, market_p1, engine, stats)
                    else:
                        from tools.wta_predict import run_prediction
                        result = run_prediction(p1_name, p2_name, surface, market_p1, engine, stats)
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    st.stop()

            comp_p1  = result["comp_p1"]
            comp_p2  = 1.0 - comp_p1
            sim_p1   = result.get("sim_p1")
            elo_diff = result["elo_diff_abs"]

            # Fetch form and compute form-adj
            _form1_t3 = _cached_form(p1_name)
            _form2_t3 = _cached_form(p2_name)
            form_adj_t3 = _form_blend_dash(comp_p1, _form1_t3, _form2_t3)

            # ── Three-indicator summary ────────────────────────────────────────
            st.divider()
            ind_cols = st.columns(3)
            _indicators = [
                ("Composite",    comp_p1, 1 - comp_p1),
                ("Simulation",   sim_p1,  1 - sim_p1 if sim_p1 else None),
                ("Form-adj",     form_adj_t3, 1 - form_adj_t3 if form_adj_t3 else None),
            ]
            for col, (label, v1, v2) in zip(ind_cols, _indicators):
                with col:
                    if v1 is not None:
                        delta = None
                        if market_p1 is not None:
                            _e = v1 - market_p1
                            delta = f"{_e*100:+.1f}% edge"
                        st.metric(
                            label,
                            f"{v1*100:.1f}% / {v2*100:.1f}%",
                            delta=delta,
                            delta_color="normal" if (delta and abs(v1 - market_p1) >= 0.04) else "off",
                        )
                    else:
                        st.metric(label, "—")

            # ── Edge signal (comp-based) ───────────────────────────────────────
            if market_p1 is not None:
                edge = comp_p1 - market_p1
                market_p2 = 1.0 - market_p1
                if edge > 0.04:
                    st.success(
                        f"**VALUE — Model favors {p1_name}**  |  "
                        f"Comp: {comp_p1*100:.1f}%  vs  Market: {market_p1*100:.1f}%  |  "
                        f"Edge: {edge*100:+.1f}%"
                    )
                elif edge < -0.04:
                    st.error(
                        f"**VALUE — Model favors {p2_name}**  |  "
                        f"Comp: {comp_p2*100:.1f}%  vs  Market: {market_p2*100:.1f}%  |  "
                        f"Edge: {abs(edge)*100:.1f}%"
                    )
                else:
                    st.info(
                        f"Model and market roughly agree  |  "
                        f"Comp: {comp_p1*100:.1f}%  vs  Market: {market_p1*100:.1f}%  |  "
                        f"Edge: {edge*100:+.1f}%"
                    )

            # ── Recent Form breakdown ─────────────────────────────────────────
            if _form1_t3 or _form2_t3:
                st.subheader("Recent Form")
                fcols = st.columns(2)
                for col, pname, fd in [(fcols[0], p1_name, _form1_t3), (fcols[1], p2_name, _form2_t3)]:
                    with col:
                        st.markdown(f"**{pname}**")
                        if fd:
                            results_seq = fd.get("results", [])[:7]
                            badges = "  ".join(
                                ":green[**W**]" if x == "W" else ":red[**L**]"
                                for x in results_seq
                            )
                            st.markdown(badges)
                            wr30 = fd.get("win_rate_30d")
                            wr60 = fd.get("win_rate_60d")
                            n30  = fd.get("n_30d", 0)
                            n60  = fd.get("n_60d", 0)
                            wr30_s = f"{wr30*100:.0f}% ({n30}m)" if wr30 is not None else "—"
                            wr60_s = f"{wr60*100:.0f}% ({n60}m)" if wr60 is not None else "—"
                            st.caption(
                                f"30d: **{wr30_s}**  "
                                f"·  60d: {wr60_s}  "
                                f"·  Streak: **{fd.get('streak','—')}**  "
                                f"·  Last: {fd.get('days_since_last','?')}d ago"
                            )
                        else:
                            st.caption("No form data available")

            # ── Component breakdown chart ──────────────────────────────────────
            st.subheader("Model Components")
            components = {
                "Elo":         (result["elo_p1"],      1 - result["elo_p1"]),
                "Surface Elo": (result["surf_elo_p1"], 1 - result["surf_elo_p1"]),
                "Rank":        (result["rank_p1"],     1 - result["rank_p1"]),
                "Return (RGW)":(result["rgw1"],        result["rgw2"]),
                "Serve (SGW)": (result["sgw1"],        result["sgw2"]),
                "Ace Rate":    (result["ace1"],        result["ace2"]),
            }

            labels = list(components.keys())
            vals_p1 = [v[0] * 100 for v in components.values()]
            vals_p2 = [v[1] * 100 for v in components.values()]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name=p1_name,
                y=labels,
                x=vals_p1,
                orientation="h",
                marker_color="#1f77b4",
                text=[f"{v:.1f}%" for v in vals_p1],
                textposition="inside",
            ))
            fig.add_trace(go.Bar(
                name=p2_name,
                y=labels,
                x=vals_p2,
                orientation="h",
                marker_color="#ff7f0e",
                text=[f"{v:.1f}%" for v in vals_p2],
                textposition="inside",
            ))
            fig.update_layout(
                barmode="group",
                height=340,
                margin=dict(l=10, r=10, t=20, b=20),
                xaxis_title="Value (%)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
            )
            fig.update_xaxes(gridcolor="rgba(255,255,255,0.1)")
            st.plotly_chart(fig, width='stretch')

            # ── Market comparison bar (if market odds provided) ────────────────
            if market_p1 is not None:
                st.subheader("Model vs Market")
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    name="Model",
                    x=[p1_name, p2_name],
                    y=[comp_p1 * 100, comp_p2 * 100],
                    marker_color=["#1f77b4", "#ff7f0e"],
                    text=[f"{comp_p1*100:.1f}%", f"{comp_p2*100:.1f}%"],
                    textposition="outside",
                ))
                fig2.add_trace(go.Bar(
                    name="Market",
                    x=[p1_name, p2_name],
                    y=[market_p1 * 100, (1 - market_p1) * 100],
                    marker_color=["rgba(31,119,180,0.4)", "rgba(255,127,14,0.4)"],
                    text=[f"{market_p1*100:.1f}%", f"{(1-market_p1)*100:.1f}%"],
                    textposition="outside",
                ))
                fig2.update_layout(
                    barmode="group",
                    height=280,
                    margin=dict(l=10, r=10, t=20, b=20),
                    yaxis_title="Win Probability (%)",
                    yaxis_range=[0, 110],
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                )
                fig2.update_yaxes(gridcolor="rgba(255,255,255,0.1)")
                st.plotly_chart(fig2, width='stretch')

            # ── Calibration details ────────────────────────────────────────────
            with st.expander("Calibration details"):
                    # live/wta are already imported (via tennis_predict/wta_predict sys.path insert)
                import sys as _sys
                _model_dir = os.path.join(REPO_ROOT, "tennis_model")
                if _model_dir not in _sys.path:
                    _sys.path.insert(0, _model_dir)
                import live as _live_mod
                calib = _live_mod.elo_diff_calib(elo_diff)
                if calib:
                    band, rate, elo_bias, comp_bias = calib
                    st.markdown(f"""
| Field | Value |
|---|---|
| Elo diff (abs) | {elo_diff:.0f} |
| Elo diff band | {band} |
| Historical fav win rate | {rate*100:.1f}% |
| Elo model bias | {elo_bias:+.3f} |
| Composite model bias | {comp_bias:+.3f} |
| Fatigue (P1 / P2) | {result['fat1']:.3f} / {result['fat2']:.3f} |
| Surface | {surface} | Best of | {best_of} |
                    """)
                    if tour_choice == "WTA":
                        st.caption("Calibration data: OOS 2023–2024 WTA (n=5,499)")
                    else:
                        st.caption("Calibration data: OOS 2024+ ATP (n=6,157)")
