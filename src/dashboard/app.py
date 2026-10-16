"""IndieGuard dashboard (Review 2). Run from the repository root:

    streamlit run src/dashboard/app.py

Pages: market explorer, rules and complaint themes, patch impact, launch-risk scorer, about. Reads the small tables in
data/dashboard/ (python -m src.dashboard.build_data) and reports/results/. Associations, not causes.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt
import pandas as pd
import streamlit as st

from src.dashboard import data

st.set_page_config(page_title="IndieGuard", page_icon=None, layout="wide")
TIER_COLORS = alt.Scale(domain=["Struggling", "Solid", "Strong"], range=["#C44E52", "#8C8C8C", "#55A868"])


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


# ------------------------------------------------------------------ page 1
def market_explorer() -> None:
    st.title("Market explorer")
    st.caption("1,861 indie roguelike and roguelite games released 2022 to 2025. Review shares come from Steam's own totals at the snapshot date.")
    g = data.games()
    with st.sidebar:
        st.subheader("Filters")
        years = st.multiselect("Release year", sorted(g.release_year.dropna().astype(int).unique()), default=[])
        bands = st.multiselect("Price band", ["free", "<$5", "$5-10", "$10-20", "$20+"], default=[])
        min_rev = st.slider("Minimum reviews", 0, 500, 20, step=10)
        all_tags = sorted({t for ts in g.tag_list for t in ts if t})
        tags = st.multiselect("Must have tags", all_tags, default=[])
    f = g[g.steam_total_reviews >= min_rev]
    if years:
        f = f[f.release_year.isin(years)]
    if bands:
        f = f[f.price_band.isin(bands)]
    for t in tags:
        f = f[f.tag_list.apply(lambda ts, t=t: t in ts)]
    tiered = f[f.tier.notna()]
    c = st.columns(4)
    c[0].metric("Games", f"{len(f):,}")
    c[1].metric("Median negative share", pct(f.neg_share.median()) if len(f) else "n/a")
    c[2].metric("Struggling (30%+ negative)", pct((tiered.tier == "Struggling").mean()) if len(tiered) else "n/a")
    c[3].metric("Strong (10% or less)", pct((tiered.tier == "Strong").mean()) if len(tiered) else "n/a")
    if f.empty:
        st.info("No games match these filters.")
        return
    left, right = st.columns(2)
    left.altair_chart(alt.Chart(f.dropna(subset=["neg_share"])).mark_bar(color="#4C72B0").encode(
        x=alt.X("neg_share:Q", bin=alt.Bin(maxbins=40), title="share of negative reviews", axis=alt.Axis(format="%")), y=alt.Y("count()", title="games")
    ).properties(title="Negative share per game", height=280), width="stretch")
    mix = tiered.groupby(["price_band", "tier"]).size().reset_index(name="games")
    mix["share"] = mix.games / mix.groupby("price_band").games.transform("sum")
    right.altair_chart(alt.Chart(mix).mark_bar().encode(
        x=alt.X("price_band:N", sort=["free", "<$5", "$5-10", "$10-20", "$20+"], title="price band"),
        y=alt.Y("share:Q", axis=alt.Axis(format="%"), title="share of games"), color=alt.Color("tier:N", scale=TIER_COLORS, title="tier"),
        tooltip=["price_band", "tier", "games"]).properties(title="Success tier by price band", height=280), width="stretch")

    st.subheader("Tags with the lowest and highest negative share")
    ex = f.dropna(subset=["neg_share"]).explode("tag_list").rename(columns={"tag_list": "tag"})
    ex = ex[~ex.tag.isin(["", "Indie"])]
    st_ = ex.groupby("tag").agg(games=("appid", "size"), neg_share=("neg_share", "mean")).query("games >= 15").sort_values("neg_share")
    if len(st_) >= 6:
        show = pd.concat([st_.head(8), st_.tail(8)]).reset_index()
        st.altair_chart(alt.Chart(show).mark_bar().encode(
            y=alt.Y("tag:N", sort="x", title=None), x=alt.X("neg_share:Q", axis=alt.Axis(format="%"), title="mean negative share"),
            color=alt.condition(alt.datum.neg_share > st_.neg_share.median(), alt.value("#C44E52"), alt.value("#55A868")),
            tooltip=["tag", "games", alt.Tooltip("neg_share:Q", format=".1%")]).properties(height=380), width="stretch")
        st.caption("Tags on at least 15 games in the current filter. Associations; small groups are noisy.")
    else:
        st.info("Too few games per tag for this filter.")

    st.subheader("Reviews over time")
    m = data.monthly()
    base = alt.Chart(m).encode(x=alt.X("month:T", title=None))
    st.altair_chart(alt.layer(base.mark_area(color="#9db4d4", opacity=.6).encode(y=alt.Y("reviews:Q", title="reviews per month")),
                              base.mark_line(color="#C44E52").encode(y=alt.Y("neg_share:Q", axis=alt.Axis(format="%", title="share negative")))
                              ).resolve_scale(y="independent").properties(height=240), width="stretch")
    st.caption("Review volume follows how well games do, so it is an outcome, not a launch signal.")

    st.subheader("Games")
    st.dataframe(f.sort_values("steam_total_reviews", ascending=False)[["name", "release_year", "price_usd", "steam_total_reviews", "neg_share", "tier", "tags"]]
                 .rename(columns={"steam_total_reviews": "reviews"}), column_config={"neg_share": st.column_config.NumberColumn("negative share", format="percent")},
                 width="stretch", hide_index=True, height=320)


# ------------------------------------------------------------------ page 2
def rules_and_complaints() -> None:
    st.title("Tag rules and complaint themes")
    r_tab, c_tab, h_tab = st.tabs(["Association rules", "Complaint themes", "Refund window"])
    with r_tab:
        st.write("FP-Growth over launch-time tags, price band and Early Access, mined on 1,191 training games. "
                 "Each rule is tested against the Strong tier (35% of training games) with FDR control, then checked once on the 299 test games.")
        r = data.rules().copy()
        c = st.columns(3)
        stab = c[0].multiselect("Stability on test games", ["stable", "unstable", "too few test games"], default=["stable"])
        min_lift = c[1].slider("Minimum lift", 1.2, 2.4, 1.5, step=0.05)
        query = c[2].text_input("Contains tag or price", "")
        r = r[r.stability.isin(stab) & (r.lift >= min_lift)]
        if query:
            r = r[r.antecedents_str.str.contains(query, case=False)]
        st.caption(f"{len(r)} of {len(data.rules())} rules. No rule predicts the Struggling tier: risk is not a matter of tags alone.")
        if len(r):
            st.altair_chart(alt.Chart(r).mark_circle(size=70).encode(
                x=alt.X("n_antecedent:Q", title="training games with the combination"), y=alt.Y("lift:Q", scale=alt.Scale(zero=False)),
                color=alt.Color("stability:N"), tooltip=["antecedents_str", "confidence", "lift", "n_antecedent", "test_n_antecedent", "test_confidence"]
            ).properties(height=300), width="stretch")
            st.dataframe(r.sort_values("lift", ascending=False)[["antecedents_str", "n_antecedent", "confidence", "lift", "q_value", "test_n_antecedent", "test_confidence", "stability"]]
                         .rename(columns={"antecedents_str": "combination", "n_antecedent": "train games", "test_n_antecedent": "test games"}),
                         width="stretch", hide_index=True, height=340)
        st.caption("Associations in observational data, with small supports (about 30 to 60 training games). Hypotheses, not recipes.")
    with c_tab:
        st.write("Eight complaint themes found by NMF topic modelling on negative English reviews.")
        p = data.topic_prevalence()
        groups = {"Negative (<2h, Refund Window)": "negative, under 2 h", "Negative (>=2h, Post-Window)": "negative, 2 h or more",
                  "Negative Reviews": "all negative", "Positive Reviews": "positive"}
        p = p[p.group.isin(groups)].assign(cohort=lambda x: x.group.map(groups))
        st.altair_chart(alt.Chart(p).mark_bar().encode(
            y=alt.Y("label:N", title=None, sort=alt.SortField("topic_id"), axis=alt.Axis(labelLimit=320)), x=alt.X("dominant_topic_share:Q", axis=alt.Axis(format="%"), title="share of reviews where this is the main theme"),
            yOffset="cohort:N", color=alt.Color("cohort:N", title=None), tooltip=["label", "cohort", alt.Tooltip("dominant_topic_share:Q", format=".1%")]
        ).properties(height=420), width="stretch")
        st.dataframe(data.topic_terms()[["topic_id", "label", "top_terms"]], width="stretch", hide_index=True)
        st.caption("Early (under 2 h) negatives lean to crashes and controls; later ones to boss balance and repetition.")
    with h_tab:
        s = data.refund_summary()
        c = st.columns(3)
        c[0].metric("Negative, under 2 h of play", pct(s.neg_share_inside_window))
        c[1].metric("Negative, 2 h or more", pct(s.neg_share_outside_window))
        c[2].metric("Share of all negatives written under 2 h", pct(s.inside_window_share_of_negatives))
        h = data.refund_heatmap()
        st.altair_chart(alt.Chart(h).mark_rect().encode(
            x=alt.X("price_band:N", sort=["free", "<$5", "$5-10", "$10-20", "$20+"], title="price band"),
            y=alt.Y("playtime:N", sort=["<2 h", "2-10 h", "10-50 h", "50 h+"], title="playtime at review"),
            color=alt.Color("neg_share:Q", scale=alt.Scale(scheme="reds"), title="negative share"),
            tooltip=["playtime", "price_band", alt.Tooltip("neg_share:Q", format=".1%"), "reviews"]).properties(height=260), width="stretch")
        st.caption("Inside the refund window, negativity climbs with price; later it barely moves.")


# ------------------------------------------------------------------ page 3
def patch_impact() -> None:
    st.title("Patch impact")
    t = data.patch_tables()
    s, ctl = t["summary"].iloc[0], t["control"].iloc[0]
    c = st.columns(4)
    c[0].metric("Patch comparisons", f"{int(s.eligible_patch_events):,}")
    c[1].metric("Games", f"{int(s.games):,}")
    c[2].metric("Change in positive share", f"{s.mean_game_level_positive_pct_change:+.2f} pp")
    c[3].metric("95% interval", f"[{s.bootstrap_95_ci_low:+.2f}, {s.bootstrap_95_ci_high:+.2f}]")
    st.write("Positive-review share in the 30 days before versus after a patch announcement (game-level mean, bootstrap over games).")
    left, right = st.columns(2)
    la = t["launch"].copy()
    la["launch_group"] = la.launch_group.replace({"<=30 days (including pre-launch)": "up to 30 d", "31-180 days": "31-180 d", ">180 days": "over 180 d"})
    left.altair_chart(alt.Chart(la).mark_bar(color="#4C72B0").encode(
        x=alt.X("launch_group:N", sort=["up to 30 d", "31-180 d", "over 180 d"], title="days since launch", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("mean_positive_pct_change:Q", title="mean change (pp)"), tooltip=["launch_group", "patch_comparisons", "mean_positive_pct_change"]
    ).properties(title="By time since launch", height=280), width="stretch")
    be = t["before"].copy()
    right.altair_chart(alt.Chart(be).mark_bar(color="#C44E52").encode(
        x=alt.X("before_positive_stratum:N", sort=["<70%", "70-90%", ">90%"], title="positive share before the patch", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("mean_positive_pct_change:Q", title="mean change (pp)"), tooltip=["before_positive_stratum", "patch_comparisons", "mean_positive_pct_change"]
    ).properties(title="By pre-patch positivity", height=280), width="stretch")
    st.info(f"Matched no-patch controls ({int(ctl.games_with_controls)} games, {int(ctl.matched_patch_windows)} windows): patch minus control "
            f"{ctl.mean_patch_minus_control_pp:+.2f} pp, 95% interval [{ctl.paired_bootstrap_95_ci_low_pp:+.2f}, {ctl.paired_bootstrap_95_ci_high_pp:+.2f}]. "
            "The interval includes zero: there is no clear patch effect once launch timing and regression to the mean are handled.")


# ------------------------------------------------------------------ page 4
def risk_scorer() -> None:
    st.title("Launch-risk scorer")
    st.warning("Predicts the review-reception tier (Struggling, Solid, Strong) from launch-time features only. About 0.43 macro-F1 in cross-validation against 0.22 "
               "for guessing, and Struggling games are hard to find. Scores are class-scaled, not calibrated probabilities. Not a sales forecast.")
    from src.dashboard.risk import predict_raw_game
    tags, cats = data.tag_vocabulary()
    raw = data.raw_games()
    mode = st.radio("Input", ["Describe a game", "Load an existing game"], horizontal=True)
    game = None
    if mode == "Load an existing game":
        name = st.selectbox("Game", sorted(raw.name.astype(str).unique()))
        row = raw[raw.name.astype(str) == name].iloc[0]
        game = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
        for k in ("price_usd", "n_dlc", "n_supported_languages"):
            game[k] = 0 if game.get(k) is None else game[k]
    else:
        with st.form("game"):
            a, b, c = st.columns(3)
            price = a.number_input("Price (USD, 0 for free)", 0.0, 100.0, 9.99, step=1.0)
            launch = b.date_input("Launch date", value=pd.Timestamp("2026-06-01"))
            early = c.checkbox("Launches in Early Access", value=False)
            d, e, f = st.columns(3)
            dlc = d.number_input("DLC at launch", 0, 50, 0)
            langs = e.number_input("Supported languages", 1, 40, 4)
            ach = f.number_input("Steam achievements", 0, 500, 30)
            g1, g2, g3 = st.columns(3)
            mac, linux = g1.checkbox("macOS", False), g2.checkbox("Linux", False)
            ctrl = g3.selectbox("Controller support", ["none", "partial", "full"], index=2)
            self_pub = st.checkbox("Self-published", True)
            chosen = st.multiselect("Store tags (up to 20)", tags, default=["Roguelike", "Pixel Graphics", "2D"], max_selections=20)
            cchosen = st.multiselect("Store categories", cats, default=[c for c in cats if c in ("Steam Achievements", "Steam Cloud")])
            st.form_submit_button("Score this game")
        game = {"price_usd": float(price), "is_free": bool(price == 0), "launch_date": str(launch),
                "release_date": str(pd.Timestamp(launch) + pd.Timedelta(days=30 if early else 0)), "launch_before_window": False,
                "n_dlc": int(dlc), "n_supported_languages": int(langs), "n_achievements": int(ach), "platform_mac": bool(mac), "platform_linux": bool(linux),
                "controller_support": ctrl, "developers": "dev", "publishers": "dev" if self_pub else "publisher",
                "tags": ";".join(chosen), "categories": ";".join(cchosen)}
    try:
        res = predict_raw_game(game)
    except Exception as e:                     # shown to the user, never silently swallowed
        st.error(f"Could not score this game: {e}")
        return
    st.subheader(f"Predicted tier: {res['tier']}")
    sc = pd.DataFrame({"tier": list(res["scores"]), "score": list(res["scores"].values())})
    st.altair_chart(alt.Chart(sc).mark_bar().encode(x=alt.X("score:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])), y=alt.Y("tier:N", sort=["Struggling", "Solid", "Strong"], title=None),
                                                    color=alt.Color("tier:N", scale=TIER_COLORS, legend=None), tooltip=["tier", alt.Tooltip("score:Q", format=".1%")]).properties(height=140), width="stretch")
    if mode == "Load an existing game":
        actual = data.games().set_index("name").tier.get(name)
        st.caption(f"Recorded tier for this game: {actual if isinstance(actual, str) else 'not tiered (under 20 reviews)'}. Games in the training split were seen by the model.")


# ------------------------------------------------------------------ page 5
def about() -> None:
    st.title("About IndieGuard")
    st.write("IndieGuard helps a small indie studio decide what to fix before Steam's refund window closes, which feature combinations to back, and when to patch.")
    st.markdown(
        "- **Data:** 1,861 indie roguelike and roguelite games (2022 to 2025) and 1.65M reviews, collected from public Steam endpoints on 1 to 2 October 2026.\n"
        "- **Split:** by game (80/20, 5 folds inside train), so no game is on both sides. Hyperparameters tuned by grid search on the shared folds; the test set is scored once.\n"
        "- **Findings:** reviews under 2 hours of play are 21% negative against 9% later; metadata alone predicts game tiers weakly; the review text is far more informative; "
        "no clear patch effect once launch timing and regression to the mean are handled.\n"
        "- **Limits:** associations, not causes; store attributes are snapshots; five very large games are capped; about 60% of reviews are not in English.\n"
        "- **Code and reports:** github.com/Shan713/IndieGuard")


pages = [st.Page(market_explorer, title="Market explorer", default=True), st.Page(rules_and_complaints, title="Tag rules and complaints"),
         st.Page(patch_impact, title="Patch impact"), st.Page(risk_scorer, title="Launch-risk scorer"), st.Page(about, title="About")]
st.navigation(pages).run()
