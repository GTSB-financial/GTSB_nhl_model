import requests
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, date
from collections import defaultdict
import json
import math
from bs4 import BeautifulSoup
import pytz

# ------------------------------------------
# SIMPLE IN-MEMORY CACHE FOR O/U SCRAPER
# ------------------------------------------
_OU_CACHE = None
_OU_CACHE_TIMESTAMP = None

# -------------------------------
# MAP ODDS API NAMES → NHL ABBR
# -------------------------------
ODDS_TEAM_NAME_TO_ABBR = {
    "anaheim ducks": "ANA",
    "arizona coyotes": "ARI",
    "utah mammoth": "UTA",
    "utah": "UTA",
    "utah mammoths": "UTA",
    "boston bruins": "BOS",
    "buffalo sabres": "BUF",
    "carolina hurricanes": "CAR",
    "columbus blue jackets": "CBJ",
    "calgary flames": "CGY",
    "chicago blackhawks": "CHI",
    "colorado avalanche": "COL",
    "dallas stars": "DAL",
    "detroit red wings": "DET",
    "edmonton oilers": "EDM",
    "florida panthers": "FLA",
    "los angeles kings": "LAK",
    "minnesota wild": "MIN",
    "montreal canadiens": "MTL",
    "montréal canadiens": "MTL",
    "new jersey devils": "NJD",
    "nashville predators": "NSH",
    "new york islanders": "NYI",
    "new york rangers": "NYR",
    "ottawa senators": "OTT",
    "philadelphia flyers": "PHI",
    "pittsburgh penguins": "PIT",
    "seattle kraken": "SEA",
    "san jose sharks": "SJS",
    "st louis blues": "STL",
    "tampa bay lightning": "TBL",
    "toronto maple leafs": "TOR",
    "vancouver canucks": "VAN",
    "vegas golden knights": "VGK",
    "winnipeg jets": "WPG",
    "washington capitals": "WSH",
    "ducks": "ANA",
    "coyotes": "ARI",
    "bruins": "BOS",
    "sabres": "BUF",
    "flames": "CGY",
    "hurricanes": "CAR",
    "blackhawks": "CHI",
    "avalanche": "COL",
    "blue jackets": "CBJ",
    "stars": "DAL",
    "red wings": "DET",
    "oilers": "EDM",
    "panthers": "FLA",
    "kings": "LAK",
    "wild": "MIN",
    "canadiens": "MTL",
    "predators": "NSH",
    "devils": "NJD",
    "islanders": "NYI",
    "rangers": "NYR",
    "senators": "OTT",
    "flyers": "PHI",
    "penguins": "PIT",
    "sharks": "SJS",
    "kraken": "SEA",
    "blues": "STL",
    "lightning": "TBL",
    "maple leafs": "TOR",
    "canucks": "VAN",
    "golden knights": "VGK",
    "capitals": "WSH",
    "jets": "WPG",
    "Mammoth": "UTA",
    "mammoth": "UTA",
    "mammoths": "UTA",
    "Mammoths": "UTA",
}
TEAM_LOGO_URL = {
    "ANA": "https://cdn.nhle.com/logos/nhl/svg/ANA_light.svg",
    "ARI": "https://cdn.nhle.com/logos/nhl/svg/ARI_light.svg",   # legacy
    "UTA": "https://cdn.nhle.com/logos/nhl/svg/UTA_light.svg",   # Utah Mammoth / Utah HC

    "BOS": "https://cdn.nhle.com/logos/nhl/svg/BOS_light.svg",
    "BUF": "https://cdn.nhle.com/logos/nhl/svg/BUF_light.svg",
    "CGY": "https://cdn.nhle.com/logos/nhl/svg/CGY_light.svg",
    "CAR": "https://cdn.nhle.com/logos/nhl/svg/CAR_light.svg",
    "CHI": "https://cdn.nhle.com/logos/nhl/svg/CHI_light.svg",
    "COL": "https://cdn.nhle.com/logos/nhl/svg/COL_light.svg",
    "CBJ": "https://cdn.nhle.com/logos/nhl/svg/CBJ_light.svg",
    "DAL": "https://cdn.nhle.com/logos/nhl/svg/DAL_light.svg",
    "DET": "https://cdn.nhle.com/logos/nhl/svg/DET_light.svg",
    "EDM": "https://cdn.nhle.com/logos/nhl/svg/EDM_light.svg",
    "FLA": "https://cdn.nhle.com/logos/nhl/svg/FLA_light.svg",
    "LAK": "https://cdn.nhle.com/logos/nhl/svg/LAK_light.svg",
    "MIN": "https://cdn.nhle.com/logos/nhl/svg/MIN_light.svg",
    "MTL": "https://cdn.nhle.com/logos/nhl/svg/MTL_light.svg",
    "NSH": "https://cdn.nhle.com/logos/nhl/svg/NSH_light.svg",
    "NJD": "https://cdn.nhle.com/logos/nhl/svg/NJD_light.svg",
    "NYI": "https://cdn.nhle.com/logos/nhl/svg/NYI_light.svg",
    "NYR": "https://cdn.nhle.com/logos/nhl/svg/NYR_light.svg",
    "OTT": "https://cdn.nhle.com/logos/nhl/svg/OTT_light.svg",
    "PHI": "https://cdn.nhle.com/logos/nhl/svg/PHI_light.svg",
    "PIT": "https://cdn.nhle.com/logos/nhl/svg/PIT_light.svg",
    "SJS": "https://cdn.nhle.com/logos/nhl/svg/SJS_light.svg",
    "SEA": "https://cdn.nhle.com/logos/nhl/svg/SEA_light.svg",
    "STL": "https://cdn.nhle.com/logos/nhl/svg/STL_light.svg",
    "TBL": "https://cdn.nhle.com/logos/nhl/svg/TBL_light.svg",
    "TOR": "https://cdn.nhle.com/logos/nhl/svg/TOR_light.svg",
    "VAN": "https://cdn.nhle.com/logos/nhl/svg/VAN_light.svg",
    "VGK": "https://cdn.nhle.com/logos/nhl/svg/VGK_light.svg",
    "WSH": "https://cdn.nhle.com/logos/nhl/svg/WSH_light.svg",
    "WPG": "https://cdn.nhle.com/logos/nhl/svg/WPG_light.svg",
}
TEAM_TO_SAO_SLUG = {
    "ANA": "ducks",
    "ARI": "coyotes",
    "BOS": "bruins",
    "BUF": "sabres",
    "CAR": "hurricanes",
    "CBJ": "blue-jackets",
    "CGY": "flames",
    "CHI": "blackhawks",
    "COL": "avalanche",
    "DAL": "stars",
    "DET": "red-wings",
    "EDM": "oilers",
    "FLA": "panthers",
    "LAK": "kings",
    "MIN": "wild",
    "MTL": "canadiens",
    "NJD": "devils",
    "NSH": "predators",
    "NYI": "islanders",
    "NYR": "rangers",
    "OTT": "senators",
    "PHI": "flyers",
    "PIT": "penguins",
    "SEA": "kraken",
    "SJS": "sharks",
    "STL": "blues",
    "TBL": "lightning",
    "TOR": "maple-leafs",
    "VAN": "canucks",
    "VGK": "golden-knights",
    "WPG": "jets",
    "WSH": "capitals",
    "UTA": "mammoth"
}


API_KEY = "9ed7fa4214b159f2d1744e260607548a"

# ---------------------------------------------------------
# TODAY STRING
# ---------------------------------------------------------
def today_ymd():
    return datetime.today().strftime("%Y-%m-%d")
def decimal_to_prob(decimal_odds):
    """Convert decimal odds (e.g. 1.91) to implied probability (0–1)."""
    try:
        if decimal_odds is None:
            return None
        d = float(decimal_odds)
        if d <= 1.0:
            return None
        return 1.0 / d
    except:
        return None

@st.cache_data(ttl=21600)  # cache for 30 minutes
def get_team_game_log_apiweb(team_abbr):
    """
    Fetch per-game GF, GA, SF, SA for a team using NHL API-Web.
    Step 1: get schedule (game IDs)
    Step 2: for each FINAL game, fetch boxscore and pull score/sog.
    """
    season = "20252026"
    schedule_url = f"https://api-web.nhle.com/v1/club-schedule-season/{team_abbr}/{season}"

    try:
        schedule = requests.get(schedule_url).json()
    except Exception as e:
        print(f"API ERROR: cannot fetch schedule for {team_abbr}: {e}")
        return []

    games = []

    final_games = [
        g for g in schedule.get("games", [])
        if g.get("gameState") == "FINAL"
    ][-10:]   # ⬅️ key change: only last 10

    for g in final_games:
        game_id = g["id"]

        box_url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore"
        try:
            box = requests.get(box_url).json()
        except Exception as e:
            print(f"BOX ERROR for game {game_id}: {e}")
            continue

        try:
            home = box["homeTeam"]
            away = box["awayTeam"]

            is_home = (home["abbrev"] == team_abbr)

            if is_home:
                gf = home["score"]
                ga = away["score"]
                sf = home.get("sog", 0)
                sa = away.get("sog", 0)
            else:
                gf = away["score"]
                ga = home["score"]
                sf = away.get("sog", 0)
                sa = home.get("sog", 0)

            opp_abbr = away["abbrev"] if is_home else home["abbrev"]

            games.append({
                "GF": gf,
                "GA": ga,
                "SF": sf,
                "SA": sa,
                "OPP": opp_abbr,
                "LOC": "vs" if is_home else "@"
            })


        except Exception as e:
            print(f"Boxscore missing fields for game {game_id}: {e}")
            continue

    return games



# ---------------------------------------------------------
# LOAD TODAY'S NHL GAMES
# ---------------------------------------------------------
@st.cache_data(ttl=300)
def get_games_today():


    """Fetch today's NHL games safely (avoid 429 rate limit crashes)."""
    today = datetime.today().strftime("%Y-%m-%d")
    url = f"https://api-web.nhle.com/v1/schedule/{today}"

    # ✅ Try once
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
    except:
        # ✅ Optional: brief retry after small delay
        try:
            time.sleep(0.5)
            r = requests.get(url, timeout=10)
            r.raise_for_status()
        except:
            return []   # ✅ graceful fallback instead of crashing

    data = r.json()
    games = []

    for block in data.get("gameWeek", []):
        if block.get("date") != today:
            continue

        for g in block.get("games", []):
            games.append({
                "away_abbr": g["awayTeam"]["abbrev"],
                "home_abbr": g["homeTeam"]["abbrev"],
                "away_name": g["awayTeam"]["commonName"]["default"],
                "home_name": g["homeTeam"]["commonName"]["default"],
            })

    return games

# ---------------------------------------------------------
# ODDS (DRAFTKINGS ONLY)
# ---------------------------------------------------------
@st.cache_data(ttl=300)  # cache for 5 minutes
def get_odds_draftkings_only():
    url = "https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds"
    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": "totals"
    }

    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


# Odds API team-name → NHL abbreviation
TEAM_MAP = {
    "ottawa senators":"OTT","boston bruins":"BOS","st louis blues":"STL","buffalo sabres":"BUF",
    "montreal canadiens":"MTL","new jersey devils":"NJD","minnesota wild":"MIN","carolina hurricanes":"CAR",
    "washington capitals":"WSH","pittsburgh penguins":"PIT","philadelphia flyers":"PHI","nashville predators":"NSH",
    "anaheim ducks":"ANA","dallas stars":"DAL","tampa bay lightning":"TBL","vegas golden knights":"VGK",
    "florida panthers":"FLA","los angeles kings":"LAK","new york rangers":"NYR","detroit red wings":"DET",
    "new york islanders":"NYI","calgary flames":"CGY","chicago blackhawks":"CHI","winnipeg jets":"WPG",
    "san jose sharks":"SJS","vancouver canucks":"VAN","edmonton oilers":"EDM","colorado avalanche":"COL",
    "utah mammoth":"UTA","utah":"UTA"
}

def build_odds_index(payload):
    idx = {}

    for event in payload:
        # ONLY DraftKings
        dk = None
        for b in event.get("bookmakers", []):
            if b.get("key") == "draftkings":
                dk = b
                break
        if not dk:
            continue

        away_raw = (event.get("away_team") or "").lower()
        home_raw = (event.get("home_team") or "").lower()

        away_abbr = ODDS_TEAM_NAME_TO_ABBR.get(away_raw)
        home_abbr = ODDS_TEAM_NAME_TO_ABBR.get(home_raw)

        if not away_abbr or not home_abbr:
            continue

        key = away_abbr + "@" + home_abbr


        # extract totals
        for m in dk.get("markets", []):
            if m.get("key") != "totals":
                continue

            line = None
            over = None
            under = None

            for o in m.get("outcomes", []):
                if line is None and "point" in o:
                    line = float(o["point"])
                if o["name"].lower() == "over":
                    over = o["price"]
                if o["name"].lower() == "under":
                    under = o["price"]

            if line is None:
                continue

            idx[key] = {
                "line": line,
                "over": over,
                "under": under
            }

    return idx

def decimal_to_american(decimal_odds):
    """Convert decimal odds (ex: 1.91) → American odds (-110 or +110)."""
    if decimal_odds in ("", None):
        return ""

    try:
        d = float(decimal_odds)
    except:
        return ""

    if d >= 2.0:
        # Positive odds
        american = int((d - 1) * 100)
        return f"+{american}"
    else:
        # Negative odds
        american = int(-100 / (d - 1))
        return str(american)


def compute_league_shooting_pct(team_stats):
    """Compute league-average shooting percentage from NST stats."""
    total_goals = 0
    total_shots = 0

    for t, s in team_stats.items():
        gf = s.get("GF/G")
        sf = s.get("SF/G")

        if isinstance(gf, (int, float)) and isinstance(sf, (int, float)):
            total_goals += gf
            total_shots += sf

    # avoid zero division — fallback historical 9.6%
    if total_shots == 0:
        return 0.096

    return total_goals / total_shots

# ---------------------------------------------------------
# MODEL PREDICTION (simple GF/GA model)
# ---------------------------------------------------------
def blend_recent(season, recent, weight=0.35):
    """Blend season-long stats with last10 stats."""
    if recent is None:
        return season
    return season * (1 - weight) + recent * weight


def predict_total(a_stats, h_stats):
    """Blend recency GF/GA with season-long xG."""
    
    # GF/GA model using recency-adjusted values
    gfga = (
        (a_stats["GF_adj"] + h_stats["GA_adj"]) / 2 +
        (h_stats["GF_adj"] + a_stats["GA_adj"]) / 2
    )

    # xG season-level model
    xg = (
        (a_stats["xGF_adj"] + h_stats["xGA_adj"]) / 2 +
        (h_stats["xGF_adj"] + a_stats["xGA_adj"]) / 2
    )

    return 0.55 * gfga + 0.45 * xg
   
def normal_cdf(x, mu, sigma):
    """Standard normal CDF for N(mu, sigma^2) using math.erf."""
    z = (x - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))

def win_prob_normal(pred_total, line, side, sigma=2.00):
    """
    Probability that total goes OVER or UNDER the line,
    assuming total goals ~ Normal(pred_total, sigma^2).
    """
    if side.lower() == "over":
        # P(Total > line) = 1 - CDF(line)
        return 1.0 - normal_cdf(line, pred_total, sigma)
    elif side.lower() == "under":
        # P(Total < line) = CDF(line)
        return normal_cdf(line, pred_total, sigma)
    else:
        return 0.0

# ---------------------------------------------------------
# TEAM STATS FROM NATURAL STAT TRICK (ALL SITUATIONS)
# ---------------------------------------------------------
@st.cache_data(ttl=21600)  # 6 hours
def compute_team_stats_from_nst():
    """
    Pulls all-situations team stats from NaturalStatTrick.
    Includes GF/G, GA/G, xGF/G, xGA/G, SF/G, SA/G, Pace.
    """
    import pandas as pd

    url = (
        "https://www.naturalstattrick.com/teamtable.php?"
        "fromseason=20252026&thruseason=20252026&stype=2&sit=all"
    )

    try:
        tables = pd.read_html(url)
    except Exception as e:
        print("NST ERROR:", e)
        return {}

    if not tables:
        print("NST ERROR: No tables returned.")
        return {}

    df = tables[0]
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    stats = {}

    for _, row in df.iterrows():
        team = row["team"].upper()
        gp = row.get("gp", 0)
        if gp == 0:
            continue

        gf = row.get("gf", 0)
        ga = row.get("ga", 0)
        xgf = row.get("xgf", 0)
        xga = row.get("xga", 0)
        sf = row.get("sf", 0)
        sa = row.get("sa", 0)

        stats[team] = {
            "games": gp,
            "GF/G": gf / gp,
            "GA/G": ga / gp,
            "xGF/G": xgf / gp,
            "xGA/G": xga / gp,
            "SF/G": sf / gp,
            "SA/G": sa / gp,
            "Pace (SF+SA)": (sf + sa) / gp,
        }
    return stats

def compute_last10_stats_apiweb(team_games):
    """Compute last-10 averages for GF, GA, SF, SA, and Pace."""
    
    if len(team_games) < 10:
        last = team_games
    else:
        last = team_games[-10:]

    gf = sum(g["GF"] for g in last) / len(last)
    ga = sum(g["GA"] for g in last) / len(last)
    sf = sum(g["SF"] for g in last) / len(last)
    sa = sum(g["SA"] for g in last) / len(last)
    pace = sum((g["SF"] + g["SA"]) for g in last) / len(last)

    return {
        "GF/G_last10": gf,
        "GA/G_last10": ga,
        "SF/G_last10": sf,
        "SA/G_last10": sa,
        "Pace_last10": pace,
    }

def format_team_stats(stats):
    """Return a copy of team stats with floats rounded to 2 decimals."""
    out = {}
    for k, v in stats.items():
        if isinstance(v, float):
            out[k] = round(v, 2)
        else:
            out[k] = v
    return out

def highlight_confidence(val):
    try:
        num = float(val)
    except:
        return ""

    # HIGH EDGE (>= 1.0) → GREEN
    if abs(num) >= 1.0:
        return "background-color: #6aff6a;"   # bright green

    # MEDIUM EDGE (>= 0.5) → YELLOW
    elif abs(num) >= 0.5:
        return "background-color: #fff75a;"   # bright yellow

    # LOW EDGE (< 0.5) → RED
    else:
        return "background-color: #ff6a6a;"   # bright red
    
def highlight_ou(row):
    """Highlight Over/Under Odds cells based on model pick AND confidence level."""
    
    pick = row.get("Model Pick", "")
    conf = row.get("Confidence", "").upper()

    styles = [""] * len(row)

    # Detect confidence level from text
    if "HIGH" in conf:
        color = "#7CFC90"  # bright green
    elif "MEDIUM" in conf:
        color = "#FFF176"  # yellow
    elif "LOW" in conf:
        color = "#E0E0E0"  # light gray
    else:
        color = ""

    # No valid confidence → no highlight
    if color == "":
        return styles

    # Determine which column to highlight
    if pick == "OVER" and "Over Odds" in row.index:
        idx = row.index.get_loc("Over Odds")
        styles[idx] = f"background-color: {color}; font-weight: bold;"

    elif pick == "UNDER" and "Under Odds" in row.index:
        idx = row.index.get_loc("Under Odds")
        styles[idx] = f"background-color: {color}; font-weight: bold;"

    return styles

def highlight_ev(val):
    try:
        num = float(val)
    except:
        return ""

    if num >= 10:
        return "background-color: #00cc00; color: black;"   # bright green
    elif num >= 5:
        return "background-color: #66ff66; color: black;"   # soft green
    elif num < 0:
        return "background-color: #ff9999; color: black;"   # red
    else:
        return ""

def compute_ev(pred_total, sportsbook_line, dec_price, side, sigma=1.35):
    """
    Returns EV as a numeric value (percent).
    side = "OVER" or "UNDER"
    """
    try:
        dec_price = float(dec_price)
    except:
        return None  # missing odds

    # profit for 1-unit stake
    payout = dec_price - 1.0

    # probability model
    win_prob = win_prob_normal(pred_total, sportsbook_line, side, sigma)

    if win_prob is None:
        return None

    # EV per unit staked (converted to percent)
    ev_raw = win_prob * payout - (1 - win_prob)
    ev_percent = ev_raw * 100
    return ev_percent

def compute_home_away_splits(team_games):
    """Compute season-level home/away GF, GA, SF, SA, Pace."""
    
    home = [g for g in team_games if g["is_home"]]
    away = [g for g in team_games if not g["is_home"]]

    def avg(lst, key):
        return sum(g[key] for g in lst) / len(lst) if lst else None

    splits = {
        "GF/G_home": avg(home, "GF"),
        "GA/G_home": avg(home, "GA"),
        "SF/G_home": avg(home, "SF"),
        "SA/G_home": avg(home, "SA"),
        "Pace_home": avg(home, "SF") + avg(home, "SA") if home else None,

        "GF/G_away": avg(away, "GF"),
        "GA/G_away": avg(away, "GA"),
        "SF/G_away": avg(away, "SF"),
        "SA/G_away": avg(away, "SA"),
        "Pace_away": avg(away, "SF") + avg(away, "SA") if away else None,
    }

    return splits

@st.cache_data(ttl=86400)
def fetch_team_over_under():
    """
    NEW VERSION — Scrapes ScoresAndOdds using the updated HTML:
    Team names are in <a data-abbr="Penguins (5)">, and O/U data is
    stored in <tr data-overs="0.4286" data-unders="0.5714">.
    """

    import time
    from bs4 import BeautifulSoup
    global _OU_CACHE, _OU_CACHE_TIMESTAMP

    # Cache: 10 minutes
    if _OU_CACHE is not None and _OU_CACHE_TIMESTAMP is not None:
        if time.time() - _OU_CACHE_TIMESTAMP < 600:
            return _OU_CACHE

    url = "https://www.scoresandodds.com/nhl/teams"

    try:
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")

        table = soup.find("table")
        if not table:
            print("O/U scrape failed: table not found")
            return {}

        tbody = table.find("tbody")
        if not tbody:
            print("O/U scrape failed: tbody not found")
            return {}

        results = {}

        # Each team is one <tr>
        for tr in tbody.find_all("tr"):
            # Example: <tr data-overs="0.4286" data-unders="0.5714">
            over_pct = tr.get("data-overs")
            under_pct = tr.get("data-unders")

            if over_pct is None:
                continue  # Not a valid team row

            # Extract team name from the <a data-abbr="Penguins (5)">
            name_tag = tr.find("a", {"data-abbr": True})
            if not name_tag:
                continue

            raw_name = name_tag.get("data-abbr").lower().strip()

            # Remove things like "(5)"
            clean_name = raw_name.split("(")[0].strip()

            # Map to NHL abbreviation
            abbr = ODDS_TEAM_NAME_TO_ABBR.get(clean_name)
            if not abbr:
                # print("No mapping for:", clean_name)
                continue

            try:
                over_pct = float(over_pct)
            except:
                over_pct = None

            results[abbr] = over_pct

        # Store in cache
        _OU_CACHE = results
        _OU_CACHE_TIMESTAMP = time.time()
        return results

    except Exception as e:
        print("O/U scrape failed:", e)
        return {}





def render_two_row_table(df):

    html = """
    <style>

    /* --- FONT --- */
    @import url('https://fonts.googleapis.com/css2?family=Lato:wght@300;400;700&display=swap');

    /* --- TABLE CONTAINER --- */
    table.nhl {
        border-collapse: collapse;
        width: 100%;
        font-size: 15px;
        font-family: 'Lato', sans-serif;

        /* Recommended combo additions */
        border: 3px solid #444;          /* outer frame */
        border-radius: 10px;             /* rounded corners */
        overflow: hidden;                /* ensures corners clip */
        box-shadow: 0 2px 6px rgba(0,0,0,0.12);
    }

    /* --- HEADER STYLE --- */
    table.nhl thead th {
        background: #f7f7f7;             /* subtle tint */
        font-weight: 700;
        border-bottom: 3px solid #444 !important;   /* bold separator */
        padding: 8px 10px !important;
        border-left: 1px solid #ccc !important;     /* thin header verticals */
        border-right: 1px solid #ccc !important;
    }

    /* --- BASE CELL STYLE (modern + compact) --- */
    table.nhl td {
        padding: 7px 10px !important;
        line-height: 1.25;
        border: 1px solid #d9d9d9 !important;       /* softened thin borders */
    }

    /* --- STACKED DIV CLEANUP --- */
    table.nhl td div {
        border: none !important;
        margin: 2px 0 !important;
        padding: 0 !important;
        line-height: 1.2 !important;
    }

        /* --- THIN INTERNAL BORDER BETWEEN AWAY/HOME ROWS --- */
    tr.away-row td,
    tr.away-row td[rowspan] {
        border-bottom: 1px solid #e6e6e6 !important;
    }

    tr.home-row td,
    tr.home-row td[rowspan] {
        border-top: none !important;
        border-bottom: 1px solid #e6e6e6 !important;
    }

    /* --- THICK MATCHUP SEPARATOR (bottom of home row) --- */
    tr.game-sep td {
        border-bottom: 4px solid #444 !important;
    }

    /* --- EV COLOR CODING --- */
    .ev-pos  { background: #c8f7c5; color: #003300; font-weight: 600; }
    .ev-mid  { background: #fff2b3; color: #664d00; font-weight: 600; }
    .ev-neg  { background: #ffcccc; color: #660000; font-weight: 600; }

    /* --- EDGE COLOR CODING --- */
    .edge-low  { background: #ffcccc; color: #660000; font-weight: 600; }
    .edge-med  { background: #fff2b3; color: #664d00; font-weight: 600; }
    .edge-high { background: #c8f7c5; color: #003300; font-weight: 600; }

    /* ============================================================
    FINAL BORDER FIX — RESTORES CORRECT ROWSPAN + MATCHUP LINES
    ============================================================ */

    /* 1) Default thin border for ALL merged cells (rowspan cells) */
    td[rowspan] {
        border-bottom: 1px solid #e6e6e6 !important;
    }

    /* 2) Thin internal border between Away/Home for non-merged cells */
    tr.away-row td:not([rowspan]) {
        border-bottom: 1px solid #e6e6e6 !important;
    }

    tr.home-row td:not([rowspan]) {
        border-top: none !important;
        border-bottom: 1px solid #e6e6e6 !important;
    }

    /* 3) Bold separator BETWEEN matchups */
    tr.away-row.game-sep td[rowspan],
    tr.home-row.game-sep td {
        border-bottom: 4px solid #444 !important;
    }

    /* Sort arrows */
    .sort-arrow {
        font-size: 12px;
        margin-left: 6px;
        color: #666;
    }

    th[data-sort="asc"] .sort-arrow {
        content: "▲";
    }

    th[data-sort="desc"] .sort-arrow {
        content: "▼";
    }

    /* Ensures logo + text stay on one line */
    td.team-cell {
        display: flex;
        align-items: center;        /* vertically align logo & text */
        gap: 6px;                   /* spacing between logo and text */
    }

    
    </style>

    <table class="nhl">
        <thead>
            <tr>
                <th>Team</th>
                <th>Team Over %</th>
                <th rowspan="2" onclick="sortMatchups(2)" data-sort="none">
                    Over% <span class="sort-arrow"></span>
                <th rowspan="2">Line</th>
                <th>O/U Odds</th>
                <th rowspan="2">Proj</th>
                <th rowspan="2">Model<br>Pick</th>
                <th rowspan="2" onclick="sortMatchups(7)" data-sort="none">
                    Edge <span class="sort-arrow"></span>
                </th>

                <th>EV O/U</th>
            </tr>
        </thead>
        <tbody>
    """

    def get_logo(abbr: str) -> str:
        """Return logo URL for a team abbr, or empty string if missing."""
        return TEAM_LOGO_URL.get(abbr, "")
    
    # EV formatting
    def fmt_ev(val):
        try:
            f = float(val)
            if f < 0:
                css = "ev-neg"
            elif f < 5:
                css = "ev-mid"
            else:
                css = "ev-pos"
            return f"<td class='{css}'>{f:+.2f}%</td>"
        except:
            return f"<td>{val}</td>"

    # EDGE formatting
    def fmt_edge(val):
        try:
            f = float(val)
            if abs(f) < 0.25:
                css = "edge-low"
            elif abs(f) < 0.50:
                css = "edge-med"
            else:
                css = "edge-high"

            if f > 0:
                return f"<td class='{css}' rowspan='2'>+{f}</td>"
            else:
                return f"<td class='{css}' rowspan='2'>{f}</td>"
        except:
            return f"<td rowspan='2'>{val}</td>"

    # Render rows
    for _, row in df.iterrows():

        ev_o = fmt_ev(row["EV Over"])
        ev_u = fmt_ev(row["EV Under"])
        edge_cell = fmt_edge(row["Edge"])

        # AWAY
        html += f"""
        <tr class="away-row game-sep">
            <td class="team-cell">
                <img src="{get_logo(row['Away'])}"
                    style="height:22px; vertical-align:middle;">
                {row['Away']}
            </td>

            <td>{row['Away Over %']}%</td>

            <td rowspan="2">{row['Game Over %']}%</td>
            <td rowspan="2">{row['Line']}</td>

            <td>O {row['Over']}</td>

            <td rowspan="2">{row['Proj']}</td>
            <td rowspan="2">{row['Model Pick']}</td>

            {edge_cell}

            {ev_o}
        </tr>
        """

        # HOME
        html += f"""
        <tr class="home-row game-sep">
            <td class="team-cell">
                <img src="{get_logo(row['Home'])}"
                    style="height:22px; vertical-align:middle;">
                {row['Home']}
            </td>


            <td>{row['Home Over %']}%</td>

            <td>U {row['Under']}</td>

            {ev_u}
        </tr>
        """

    html += "</tbody></table>"
    html += """
    <script>

    function sortMatchups(colIndex) {
        const table = document.querySelector("table.nhl");
        const tbody = table.querySelector("tbody");
        const headers = table.querySelectorAll("thead th");

        // 1. Build array of matchup pairs
        const rows = Array.from(tbody.querySelectorAll("tr"));
        let matchups = [];
        for (let i = 0; i < rows.length; i++) {
            if (rows[i].classList.contains("away-row")) {
                matchups.push([rows[i], rows[i+1]]);
                i++;
            }
        }

        // 2. Determine sorting direction
        const th = headers[colIndex];
        let currentState = th.getAttribute("data-sort");
        let newState = (currentState === "asc") ? "desc" : "asc";

        // Reset all arrows except this one
        headers.forEach(h => {
            h.setAttribute("data-sort", "none");
            let span = h.querySelector(".sort-arrow");
            if (span) span.innerHTML = "";
        });

        // Set new state for this column
        th.setAttribute("data-sort", newState);
        let arrow = th.querySelector(".sort-arrow");
        if (arrow) arrow.innerHTML = (newState === "asc") ? "▲" : "▼";

        // 3. Extract column values
        function getCellValue(row, index) {
            let cell = row.children[index];
            if (!cell) return 0;

            let text = cell.innerText
                .replace("%", "")
                .replace("+", "")
                .replace("−", "-"); // safety for unicode minus

            let val = parseFloat(text);
            return isNaN(val) ? 0 : Math.abs(val);
        }


        // 4. Sort matchup pairs
        matchups.sort((a, b) => {
            let aVal = getCellValue(a[0], colIndex);
            let bVal = getCellValue(b[0], colIndex);
            return (newState === "asc") ? (aVal - bVal) : (bVal - aVal);
        });

        // 5. Re-render sorted rows
        tbody.innerHTML = "";
        matchups.forEach(pair => {
            tbody.appendChild(pair[0]);
            tbody.appendChild(pair[1]);
        });
    }

    </script>

    """

    return html
  
@st.cache_data(ttl=3600)
def fetch_last5_games_scoresandodds(team_slug, max_games=5):
    """
    Returns last 5 games from ScoresAndOdds with:
    opponent, score, W/L, O/U
    """
    import requests
    from bs4 import BeautifulSoup

    url = f"https://www.scoresandodds.com/nhl/teams/{team_slug}"
    headers = {"User-Agent": "Mozilla/5.0"}

    r = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(r.text, "lxml")

    container = soup.select_one("div#this ul.table-list.active")
    if not container:
        return []

    games = []

    rows = container.select("li > div.table-list-row")

    for row in rows:
        score_span = row.select_one("span.table-list-score")
        if not score_span:
            continue

        score_txt = score_span.get_text(strip=True)

        try:
            g1, g2 = map(int, score_txt.split("-"))
            total_goals = g1 + g2
        except:
            continue

        team_span = row.select_one("span.table-list-team.win, span.table-list-team.loss")
        opp_span  = row.select_one("span.table-list-team.opp")
        ou_span   = row.select_one("span.table-list-odds.ou")

        if not team_span or not opp_span or not ou_span:
            continue

        result = "W" if "win" in team_span.get("class", []) else "L"
        opp = opp_span.get("data-abbr", "").upper()

        # Parse total line (always shown as over)
        ou_txt = ou_span.get_text(strip=True).lower()
        try:
            line = float(ou_txt.replace("o", "").replace("u", ""))
            ou_result = "O" if total_goals > line else "U"
        except:
            ou_result = None

        games.append({
            "result": result,
            "score": score_txt,
            "opp": opp,
            "ou": ou_result
        })

        if len(games) >= max_games:
            break

    return games





# ---------------------------------------------------------
# STREAMLIT MAIN
# ---------------------------------------------------------
def main():
 
    from datetime import datetime
    import time

    # Title (same as before)
    st.title("GTSB Winners Only Board")

    # Force timezone to Central Time
    central = pytz.timezone("America/Chicago")
    now_ct = datetime.now(central)

    today_str = now_ct.strftime("%B %d, %Y")
    updated_str = now_ct.strftime("%Y-%m-%d %I:%M:%S %p")


    # Larger header + timestamp underneath
    st.markdown(
        f"""
        <div style="margin-top:-10px;">
            <h2 style="margin-bottom:0px;">{today_str}</h2>
            <p style="font-size:16px; color:gray; margin-top:0px;">Updated: {updated_str}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


    # Load games
    games = get_games_today()
    # Get a unique set of teams playing today
    teams_in_today = set()

    for g in games:
        teams_in_today.add(g["away_abbr"])
        teams_in_today.add(g["home_abbr"])


    # -------------------------
    # LOAD NST STATS (FULL NAMES)
    # -------------------------
    team_stats = compute_team_stats_from_nst()

    # -------------------------
    # CONVERT FULL NAMES → ABBREVIATIONS
    # -------------------------
    converted = {}

    for full_name, stats in team_stats.items():
        key_norm = full_name.lower()
        abbr = ODDS_TEAM_NAME_TO_ABBR.get(key_norm)
    
        if abbr:
            converted[abbr] = stats
        else:
            print("NST WARNING: No abbreviation found for:", full_name)

    team_stats = converted

    over_data = fetch_team_over_under()

    for abbr, stats in team_stats.items():
        stats["OverPct"] = over_data.get(abbr, None)

    # -------------------------
    # OPTIMIZED LAST-10 LOOKUPS
    # Only fetch last-10 for teams playing today
    # -------------------------
    for abbr, stats in team_stats.items():

        if abbr not in teams_in_today:
            # Still must create last10 fields so the model doesn't break
            stats["GF/G_last10"] = stats["GF/G"]
            stats["GA/G_last10"] = stats["GA/G"]
            stats["SF/G_last10"] = stats["SF/G"]
            stats["SA/G_last10"] = stats["SA/G"]
            stats["Pace_last10"] = stats["Pace (SF+SA)"]
            continue

        # Teams playing today → fetch real last-10
        team_games = get_team_game_log_apiweb(abbr)

        if team_games:
            last10 = compute_last10_stats_apiweb(team_games)
            stats.update(last10)
        else:
            # fallback → use season averages
            stats.update({
                "GF/G_last10": stats["GF/G"],
                "GA/G_last10": stats["GA/G"],
                "SF/G_last10": stats["SF/G"],
                "SA/G_last10": stats["SA/G"],
                "Pace_last10": stats["Pace (SF+SA)"],
            })
    # ---------------------------
    # APPLY RECENCY BLENDING
    # ---------------------------
    RECENCY_WEIGHT = 0.50

    for team, stats in team_stats.items():

        stats["GF_adj"] = blend_recent(stats["GF/G"], stats["GF/G_last10"], RECENCY_WEIGHT)
        stats["GA_adj"] = blend_recent(stats["GA/G"], stats["GA/G_last10"], RECENCY_WEIGHT)

        # xG stays season-level because NHL API doesn't give xG per game (Option A)
        stats["xGF_adj"] = stats["xGF/G"]
        stats["xGA_adj"] = stats["xGA/G"]

        # Blended pace
        stats["Pace_adj"] = blend_recent(stats["Pace (SF+SA)"], stats["Pace_last10"], RECENCY_WEIGHT) 
    
    # Load odds
    st.write("✅ Fetching DraftKings odds…")
    odds_payload = get_odds_draftkings_only()
    odds_idx = build_odds_index(odds_payload)
    
    rows = []

    # -----------------------------
    # BUILD TABLE FOR TODAY'S GAMES
    # -----------------------------
    for g in games:
        away = g["away_abbr"]
        home = g["home_abbr"]

        key = f"{away}@{home}"
        odds = odds_idx.get(key, {})

        away_stats = team_stats.get(away, {"GF/G": 2.8, "GA/G": 2.8})
        home_stats = team_stats.get(home, {"GF/G": 2.8, "GA/G": 2.8})
        
        predicted = round(predict_total(away_stats, home_stats), 2)


        # ---------------------------
        # CONFIDENCE METRIC + MODEL PICK + BOTH EVs
        # ---------------------------
        model_pick = ""
        dist = None
        sportsbook_line = None

        line = odds.get("line", "")
        over_dec = odds.get("over")
        under_dec = odds.get("under")

        # ---------------------------
        # EDGE + model pick direction
        # ---------------------------
        edge = ""

        if line not in ("", None):
            try:
                sportsbook_line = float(line)
                dist = predicted - sportsbook_line   # signed difference

                # MODEL PICK LOGIC
                if dist > 0:
                    model_pick = "OVER"
                elif dist < 0:
                    model_pick = "UNDER"
                else:
                    model_pick = ""

                # EDGE VALUE (numeric only, formatted)
                if dist > 0:
                    edge = f"+{abs(dist):.2f}"
                elif dist < 0:
                    edge = f"-{abs(dist):.2f}"
                else:
                    edge = "0.00"

            except:
                edge = ""
                model_pick = ""
        else:
            model_pick = ""
            edge = ""



        # ---------------------------
        # EV FOR BOTH OVER & UNDER
        # ---------------------------
        ev_over = compute_ev(predicted, sportsbook_line, over_dec, "OVER")
        ev_under = compute_ev(predicted, sportsbook_line, under_dec, "UNDER")

        away_over_pct = team_stats.get(away, {}).get("OverPct")
        home_over_pct = team_stats.get(home, {}).get("OverPct")

        # Format as integer % or blank
        def fmt_pct(x):
            try:
                return round(float(x) * 100)
            except:
                return ""

        away_over_pct_fmt = fmt_pct(away_over_pct)
        home_over_pct_fmt = fmt_pct(home_over_pct)

        # Game Over% = average of the two
        if away_over_pct is not None and home_over_pct is not None:
            game_over_pct = round((away_over_pct + home_over_pct) / 2 * 100)
        else:
            game_over_pct = ""




        rows.append({
            "Away": away,
            "Home": home,

            "Away Over %": away_over_pct_fmt,
            "Home Over %": home_over_pct_fmt,
            "Game Over %": game_over_pct,

            "Line": line,
            "Over": decimal_to_american(over_dec),
            "Under": decimal_to_american(under_dec),
            "Proj": predicted,
            "Model Pick": model_pick,
            "Edge": edge,
            "EV Over": ev_over,
            "EV Under": ev_under,

            # store metrics for choosing daily winners
            "_EV_MAX": max(ev_over if pd.notna(ev_over) else -999,
                        ev_under if pd.notna(ev_under) else -999),
            "_EV_SIDE": "OVER" if (ev_over or -999) >= (ev_under or -999) else "UNDER",
            "_DIST": abs(dist) if dist is not None else -999,
            "_MODEL_PICK": model_pick,
        })








    df = pd.DataFrame(rows)

    # Identify best bet
    best_bet_idx = df["_EV_MAX"].idxmax()
    best_bet_side = df.loc[best_bet_idx, "_EV_SIDE"]

    # Identify best value
    best_value_idx = df["_DIST"].idxmax()
    best_value_side = df.loc[best_value_idx, "_MODEL_PICK"]
    
    # ---------- BEST BET STAR ----------
    current = df.at[best_bet_idx, "Model Pick"]
    if not isinstance(current, str):
        current = ""
    df.at[best_bet_idx, "Model Pick"] = current.rstrip() + " ⭐"

    # ---------- BEST VALUE DIAMOND ----------
    current = df.at[best_value_idx, "Model Pick"]
    if not isinstance(current, str):
        current = ""     # ensures no errors
    df.at[best_value_idx, "Model Pick"] = current.rstrip() + " ◆"

    # Drop helper columns
    df = df.drop(columns=["_EV_MAX", "_EV_SIDE", "_DIST", "_MODEL_PICK"])

    # Ensure numeric types for formatting in HTML
    df["Line"] = pd.to_numeric(df["Line"], errors="ignore")
    df["Proj"] = pd.to_numeric(df["Proj"], errors="ignore")
    df["EV Over"] = pd.to_numeric(df["EV Over"], errors="coerce")
    df["EV Under"] = pd.to_numeric(df["EV Under"], errors="coerce")

    st.subheader("Today's Games")
   
    # ---- ONLY THIS: render HTML table ----
    html_table = render_two_row_table(df)
    
    import streamlit.components.v1 as components
    components.html(html_table, height=900, scrolling=True)
    
    st.markdown("---")
    st.header("🔍 Explain Matchup")

    # Build matchup options
    matchup_map = {
        f"{g['away_abbr']} @ {g['home_abbr']}": g
        for g in games
    }

    # Default to Best Bet matchup if available
    default_matchup = list(matchup_map.keys())[0]
    if "⭐" in df["Model Pick"].astype(str).to_string():
        for i, row in df.iterrows():
            if "⭐" in str(row["Model Pick"]):
                default_matchup = f"{row['Away']} @ {row['Home']}"
                break

    selected_label = st.selectbox(
        "Select a matchup to explain:",
        list(matchup_map.keys()),
        index=list(matchup_map.keys()).index(default_matchup)
    )

    g = matchup_map[selected_label]
    away = g["away_abbr"]
    home = g["home_abbr"]

    away_stats = team_stats.get(away, {})
    home_stats = team_stats.get(home, {})

    def compact_line(label, season, last10, trend_html):
        return (
            f"<div style='line-height:1.25; margin-bottom:2px;'>"
            f"<strong>{label}</strong>: {season} → {last10} "
            f"&nbsp;&nbsp; {trend_html}"
            f"</div>"
        )



    def trend_fmt(season, last10):
        try:
            delta = last10 - season
        except:
            return ""

        delta = round(delta, 2)

        # Neutral
        if abs(delta) < 0.05:
            return "<span style='color:gray;'>— 0.00</span>"

        # Positive
        if delta > 0:
            return f"<span style='color:green; font-weight:600;'>▲ +{delta}</span>"

        # Negative
        return f"<span style='color:red; font-weight:600;'>▼ {delta}</span>"

    def trend(season, last10):
        try:
            return round(last10 - season, 2)
        except:
            return ""

    def fmt(val):
        try:
            return round(val, 2)
        except:
            return ""

    def last5_results(team_abbr):
        slug = TEAM_TO_SAO_SLUG.get(team_abbr)
        if not slug:
            return []

        games = fetch_last5_games_scoresandodds(slug)

        out = []

        for g in games:
            result = g["result"]          # "W" or "L"
            score  = g["score"]           # "3-2"
            opp    = g["opp"]             # "OTT"
            ou     = g.get("ou")           # "O" / "U" / None

            # Color W/L
            if result == "W":
                res_html = "<span style='color:#2E7D32; font-weight:600;'>W</span>"
            else:
                res_html = "<span style='color:#C62828; font-weight:600;'>L</span>"

            # O/U display
            if ou == "O":
                ou_html = " (OVER)"
            elif ou == "U":
                ou_html = " (UNDER)"
            else:
                ou_html = ""


            out.append(f"{res_html} {score} vs {opp}{ou_html}")

        return out





    colA, colB = st.columns(2)

    # ---------------- AWAY TEAM ----------------
    with colA:
        st.subheader(f"{away} (Away)")

        st.markdown("**Season vs Last 10**")

        st.markdown(
            compact_line(
                "GF/G",
                fmt(away_stats.get("GF/G")),
                fmt(away_stats.get("GF/G_last10")),
                trend_fmt(away_stats.get("GF/G"), away_stats.get("GF/G_last10"))
            ),
            unsafe_allow_html=True
        )

        st.markdown(
            compact_line(
                "GA/G",
                fmt(away_stats.get("GA/G")),
                fmt(away_stats.get("GA/G_last10")),
                trend_fmt(away_stats.get("GA/G"), away_stats.get("GA/G_last10"))
            ),
            unsafe_allow_html=True
        )

        st.markdown(
            compact_line(
                "Pace",
                fmt(away_stats.get("Pace (SF+SA)")),
                fmt(away_stats.get("Pace_last10")),
                trend_fmt(
                    away_stats.get("Pace (SF+SA)"),
                    away_stats.get("Pace_last10")
                )
            ),
            unsafe_allow_html=True
        )



        st.markdown("**Last 5 Games**")
        for r in last5_results(away):
            st.markdown(
                f"<div style='line-height:1.2; margin-bottom:1px;'>{r}</div>",
                unsafe_allow_html=True
            )


    # ---------------- HOME TEAM ----------------
    with colB:
        st.subheader(f"{home} (Home)")

        st.markdown("**Season vs Last 10**")

        st.markdown(
            compact_line(
                "GF/G",
                fmt(home_stats.get("GF/G")),
                fmt(home_stats.get("GF/G_last10")),
                trend_fmt(home_stats.get("GF/G"), home_stats.get("GF/G_last10"))
            ),
            unsafe_allow_html=True
        )

        st.markdown(
            compact_line(
                "GA/G",
                fmt(home_stats.get("GA/G")),
                fmt(home_stats.get("GA/G_last10")),
                trend_fmt(home_stats.get("GA/G"), home_stats.get("GA/G_last10"))
            ),
            unsafe_allow_html=True
        )

        st.markdown(
            compact_line(
                "Pace",
                fmt(home_stats.get("Pace (SF+SA)")),
                fmt(home_stats.get("Pace_last10")),
                trend_fmt(
                    home_stats.get("Pace (SF+SA)"),
                    home_stats.get("Pace_last10")
                )
            ),
            unsafe_allow_html=True
        )


        st.markdown("**Last 5 Games**")
        for r in last5_results(home):
            st.markdown(
                f"<div style='line-height:1.2; margin-bottom:1px;'>{r}</div>",
                unsafe_allow_html=True
            )



if __name__ == "__main__":
    main()

