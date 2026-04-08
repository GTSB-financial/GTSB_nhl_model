import requests
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, date, timezone
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
    "ARI": "https://cdn.nhle.com/logos/nhl/svg/ARI_light.svg",
    "UTA": "https://cdn.nhle.com/logos/nhl/svg/UTA_light.svg",
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

API_KEY = "abd6a7659c64fd320752e57fef58691b"

DEFAULT_STATS = {
    "games": 0,
    "GF/G": 2.8,
    "GA/G": 2.8,
    "xGF/G": 2.8,
    "xGA/G": 2.8,
    "SF/G": 29.0,
    "SA/G": 29.0,
    "Pace (SF+SA)": 58.0,
    "GF/G_last10": 2.8,
    "GA/G_last10": 2.8,
    "SF/G_last10": 29.0,
    "SA/G_last10": 29.0,
    "Pace_last10": 58.0,
    "GF_adj": 2.8,
    "GA_adj": 2.8,
    "xGF_adj": 2.8,
    "xGA_adj": 2.8,
    "Pace_adj": 58.0,
    "OverPct": None,
}

def today_ymd():
    return datetime.today().strftime("%Y-%m-%d")

def decimal_to_prob(decimal_odds):
    try:
        if decimal_odds is None:
            return None
        d = float(decimal_odds)
        if d <= 1.0:
            return None
        return 1.0 / d
    except:
        return None

@st.cache_data(ttl=21600)
def get_team_game_log_apiweb(team_abbr):
    season = "20252026"
    schedule_url = f"https://api-web.nhle.com/v1/club-schedule-season/{team_abbr}/{season}"

    try:
        schedule = requests.get(schedule_url, timeout=15).json()
    except Exception as e:
        print(f"API ERROR: cannot fetch schedule for {team_abbr}: {e}")
        return []

    games = []

    final_games = [
        g for g in schedule.get("games", [])
        if g.get("gameState") == "FINAL"
    ][-10:]

    for g in final_games:
        game_id = g["id"]
        box_url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore"
        try:
            box = requests.get(box_url, timeout=15).json()
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

@st.cache_data(ttl=300)
def get_games_today():
    now_utc = datetime.now(timezone.utc)
    today_utc = now_utc.date()
    url = f"https://api-web.nhle.com/v1/schedule/{today_utc.isoformat()}"

    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
    except:
        return []

    data = r.json()
    games = []

    for block in data.get("gameWeek", []):
        if block.get("date") != today_utc.isoformat():
            continue

        for g in block.get("games", []):
            start = g.get("startTimeUTC")
            if not start:
                continue

            start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))

            if start_dt <= now_utc:
                continue

            games.append({
                "away_abbr": g["awayTeam"]["abbrev"],
                "home_abbr": g["homeTeam"]["abbrev"],
                "away_name": g["awayTeam"]["commonName"]["default"],
                "home_name": g["homeTeam"]["commonName"]["default"],
                "startTimeUTC": start,
            })

    return games

@st.cache_data(ttl=300)
def get_odds_draftkings_only():
    url = "https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds"
    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": "totals,h2h"
    }

    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()

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

def build_ml_odds_index(payload):
    idx = {}

    for event in payload:
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

        away_ml_dec = None
        home_ml_dec = None

        for m in dk.get("markets", []):
            if m.get("key") != "h2h":
                continue

            for o in m.get("outcomes", []):
                name_raw = (o.get("name") or "").lower()
                price = o.get("price")

                team_abbr = ODDS_TEAM_NAME_TO_ABBR.get(name_raw)
                if not team_abbr:
                    continue

                if team_abbr == away_abbr:
                    away_ml_dec = price
                elif team_abbr == home_abbr:
                    home_ml_dec = price

        if away_ml_dec is not None or home_ml_dec is not None:
            idx[key] = {
                "away_ml_dec": away_ml_dec,
                "home_ml_dec": home_ml_dec,
            }

    return idx

def decimal_to_american(decimal_odds):
    if decimal_odds in ("", None):
        return ""

    try:
        d = float(decimal_odds)
    except:
        return ""

    if d >= 2.0:
        american = int((d - 1) * 100)
        return f"+{american}"
    else:
        american = int(-100 / (d - 1))
        return str(american)

def prob_to_american(prob):
    try:
        prob = float(prob)
    except:
        return None

    if prob <= 0 or prob >= 1:
        return None

    if prob >= 0.5:
        return int(round(-prob / (1 - prob) * 100))
    else:
        return int(round((1 - prob) / prob * 100))

def american_to_prob(odds):
    try:
        odds = int(odds)
    except:
        return None

    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)

def compute_league_shooting_pct(team_stats):
    total_goals = 0
    total_shots = 0

    for t, s in team_stats.items():
        gf = s.get("GF/G")
        sf = s.get("SF/G")

        if isinstance(gf, (int, float)) and isinstance(sf, (int, float)):
            total_goals += gf
            total_shots += sf

    if total_shots == 0:
        return 0.096

    return total_goals / total_shots

def blend_recent(season, recent, weight=0.35):
    if recent is None:
        return season
    return season * (1 - weight) + recent * weight

def predict_total(a_stats, h_stats):
    a_gf = a_stats.get("GF_adj", a_stats.get("GF/G", 2.8))
    a_ga = a_stats.get("GA_adj", a_stats.get("GA/G", 2.8))
    a_xgf = a_stats.get("xGF_adj", a_stats.get("xGF/G", a_gf))
    a_xga = a_stats.get("xGA_adj", a_stats.get("xGA/G", a_ga))

    h_gf = h_stats.get("GF_adj", h_stats.get("GF/G", 2.8))
    h_ga = h_stats.get("GA_adj", h_stats.get("GA/G", 2.8))
    h_xgf = h_stats.get("xGF_adj", h_stats.get("xGF/G", h_gf))
    h_xga = h_stats.get("xGA_adj", h_stats.get("xGA/G", h_ga))

    gfga = ((a_gf + h_ga) / 2) + ((h_gf + a_ga) / 2)
    xg = ((a_xgf + h_xga) / 2) + ((h_xgf + a_xga) / 2)

    return 0.55 * gfga + 0.45 * xg

def split_team_xg_from_total(pred_total, away_stats, home_stats):
    away_strength = (
        away_stats.get("GF_adj", away_stats.get("GF/G", 2.8)) +
        home_stats.get("GA_adj", home_stats.get("GA/G", 2.8))
    ) / 2

    home_strength = (
        home_stats.get("GF_adj", home_stats.get("GF/G", 2.8)) +
        away_stats.get("GA_adj", away_stats.get("GA/G", 2.8))
    ) / 2

    strength_sum = away_strength + home_strength

    if strength_sum <= 0 or pred_total is None:
        return None, None

    away_xg = pred_total * (away_strength / strength_sum)
    home_xg = pred_total * (home_strength / strength_sum)

    return round(away_xg, 2), round(home_xg, 2)

def pythagorean_win_prob(team_xg, opp_xg, k=2.0):
    try:
        team_xg = float(team_xg)
        opp_xg = float(opp_xg)
    except:
        return None

    if team_xg <= 0 and opp_xg <= 0:
        return None

    denom = (team_xg ** k) + (opp_xg ** k)
    if denom == 0:
        return None

    return team_xg ** k / denom

def normal_cdf(x, mu, sigma):
    z = (x - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))

def win_prob_normal(pred_total, line, side, sigma=2.00):
    if pred_total is None or line is None:
        return None

    if side.lower() == "over":
        return 1.0 - normal_cdf(line, pred_total, sigma)
    elif side.lower() == "under":
        return normal_cdf(line, pred_total, sigma)
    else:
        return 0.0

@st.cache_data(ttl=21600)
def compute_team_stats_from_nst():
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
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    stats = {}

    for _, row in df.iterrows():
        team = str(row["team"]).upper().strip()
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
    if not team_games:
        return {
            "GF/G_last10": None,
            "GA/G_last10": None,
            "SF/G_last10": None,
            "SA/G_last10": None,
            "Pace_last10": None,
        }

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

    if abs(num) >= 1.0:
        return "background-color: #6aff6a;"
    elif abs(num) >= 0.5:
        return "background-color: #fff75a;"
    else:
        return "background-color: #ff6a6a;"

def highlight_ou(row):
    pick = row.get("Model Pick", "")
    conf = row.get("Confidence", "").upper()
    styles = [""] * len(row)

    if "HIGH" in conf:
        color = "#7CFC90"
    elif "MEDIUM" in conf:
        color = "#FFF176"
    elif "LOW" in conf:
        color = "#E0E0E0"
    else:
        color = ""

    if color == "":
        return styles

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
        return "background-color: #00cc00; color: black;"
    elif num >= 5:
        return "background-color: #66ff66; color: black;"
    elif num < 0:
        return "background-color: #ff9999; color: black;"
    else:
        return ""

def compute_ev(pred_total, sportsbook_line, dec_price, side, sigma=1.35):
    try:
        dec_price = float(dec_price)
    except:
        return None

    payout = dec_price - 1.0
    win_prob = win_prob_normal(pred_total, sportsbook_line, side, sigma)

    if win_prob is None:
        return None

    ev_raw = win_prob * payout - (1 - win_prob)
    ev_percent = ev_raw * 100
    return ev_percent

def compute_ev_ml(win_prob, dec_price):
    if win_prob is None or dec_price in (None, ""):
        return None

    try:
        d = float(dec_price)
    except:
        return None

    if d <= 1.0:
        return None

    payout = d - 1.0
    ev_raw = win_prob * payout - (1 - win_prob)
    return ev_raw * 100.0

def compute_home_away_splits(team_games):
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
    import time
    global _OU_CACHE, _OU_CACHE_TIMESTAMP

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

        for tr in tbody.find_all("tr"):
            over_pct = tr.get("data-overs")
            under_pct = tr.get("data-unders")

            if over_pct is None:
                continue

            name_tag = tr.find("a", {"data-abbr": True})
            if not name_tag:
                continue

            raw_name = name_tag.get("data-abbr").lower().strip()
            clean_name = raw_name.split("(")[0].strip()
            abbr = ODDS_TEAM_NAME_TO_ABBR.get(clean_name)
            if not abbr:
                continue

            try:
                over_pct = float(over_pct)
            except:
                over_pct = None

            results[abbr] = over_pct

        _OU_CACHE = results
        _OU_CACHE_TIMESTAMP = time.time()
        return results

    except Exception as e:
        print("O/U scrape failed:", e)
        return {}

def render_two_row_table(df):
    html = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lato:wght@300;400;700&display=swap');

    table.nhl {
        border-collapse: collapse;
        width: 100%;
        font-size: 15px;
        font-family: 'Lato', sans-serif;
        border: 3px solid #444;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 6px rgba(0,0,0,0.12);
    }

    table.nhl thead th {
        background: #f7f7f7;
        font-weight: 700;
        border-bottom: 3px solid #444 !important;
        padding: 6px 6px !important;
        border-left: 1px solid #ccc !important;
        border-right: 1px solid #ccc !important;
    }

    table.nhl td {
        padding: 5px 6px !important;
        line-height: 1.25;
        border: 1px solid #d9d9d9 !important;
    }

    table.nhl td div {
        border: none !important;
        margin: 2px 0 !important;
        padding: 0 !important;
        line-height: 1.2 !important;
    }

    tr.away-row td,
    tr.away-row td[rowspan] {
        border-bottom: 1px solid #e6e6e6 !important;
    }

    tr.home-row td,
    tr.home-row td[rowspan] {
        border-top: none !important;
        border-bottom: 1px solid #e6e6e6 !important;
    }

    tr.game-sep td {
        border-bottom: 4px solid #444 !important;
    }

    .ev-pos  { background: #c8f7c5; color: #003300; font-weight: 600; }
    .ev-mid  { background: #fff2b3; color: #664d00; font-weight: 600; }
    .ev-neg  { background: #ffcccc; color: #660000; font-weight: 600; }

    .edge-low  { background: #ffcccc; color: #660000; font-weight: 600; }
    .edge-med  { background: #fff2b3; color: #664d00; font-weight: 600; }
    .edge-high { background: #c8f7c5; color: #003300; font-weight: 600; }

    td[rowspan] {
        border-bottom: 1px solid #e6e6e6 !important;
    }

    tr.away-row td:not([rowspan]) {
        border-bottom: 1px solid #e6e6e6 !important;
    }

    tr.home-row td:not([rowspan]) {
        border-top: none !important;
        border-bottom: 1px solid #e6e6e6 !important;
    }

    tr.away-row.game-sep td[rowspan],
    tr.home-row.game-sep td {
        border-bottom: 4px solid #444 !important;
    }

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

    td.team-cell {
        display: flex;
        align-items: center;
        gap: 6px;
    }

    th, td {
        white-space: nowrap;
    }

    table.nhl thead th {
        position: sticky;
        top: 0;
        z-index: 5;
    }
    </style>

    <table class="nhl">
        <thead>
          <tr>
            <th>Team</th>
            <th>ML</th>
            <th data-sort="none" onclick="sortMatchups(2)">
              ML<br>Diff <span class="sort-arrow"></span>
            </th>
            <th data-sort="none" onclick="sortMatchups(3)">
              EV <span class="sort-arrow"></span>
            </th>
            <th>Team<br>Ov %</th>
            <th rowspan="2" data-sort="none" onclick="sortMatchups(5)">
              Ov % <span class="sort-arrow"></span>
            </th>
            <th rowspan="2">Line</th>
            <th>O/U Odds</th>
            <th rowspan="2">Proj</th>
            <th rowspan="2">Pick</th>
            <th rowspan="2" data-sort="none" onclick="sortMatchups(10)">
              Edge <span class="sort-arrow"></span>
            </th>
            <th>EV O/U</th>
          </tr>
        </thead>

        <colgroup>
            <col style="width:140px">
            <col style="width:70px">
            <col style="width:65px">
            <col style="width:60px">
            <col style="width:70px">
            <col style="width:60px">
            <col style="width:60px">
            <col style="width:85px">
            <col style="width:55px">
            <col style="width:55px">
            <col style="width:55px">
            <col style="width:60px">
        </colgroup>

        <tbody>
    """

    def get_logo(abbr: str) -> str:
        return TEAM_LOGO_URL.get(abbr, "")

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

    def fmt_ml_ev(val):
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
            return "<td></td>"

    def fmt_ml_goal_diff(val, side):
        try:
            f = float(val)
        except:
            return "<td></td>"

        sort_val = abs(f)

        if abs(f) < 0.05:
            return f"<td data-value='0'></td>"

        show = (f > 0 and side == "HOME") or (f < 0 and side == "AWAY")

        if not show:
            return f"<td data-value='{sort_val}'></td>"

        if sort_val < 0.25:
            css = "edge-low"
        elif sort_val < 0.75:
            css = "edge-med"
        else:
            css = "edge-high"

        return f"<td class='{css}' data-value='{sort_val}'>+{sort_val:.2f}</td>"

    for _, row in df.iterrows():
        ev_o = fmt_ev(row["EV Over"])
        ev_u = fmt_ev(row["EV Under"])
        edge_cell = fmt_edge(row["Edge"])
        ml_ev_away = fmt_ml_ev(row["ML Away EV"])
        ml_ev_home = fmt_ml_ev(row["ML Home EV"])
        ml_goal_away = fmt_ml_goal_diff(row["ML Goal Diff"], "AWAY")
        ml_goal_home = fmt_ml_goal_diff(row["ML Goal Diff"], "HOME")

        ml_ev_sort = max(
            abs(row["ML Away EV"]) if row["ML Away EV"] is not None else 0,
            abs(row["ML Home EV"]) if row["ML Home EV"] is not None else 0,
        )

        ou_ev_sort = max(
            abs(row["EV Over"]) if row["EV Over"] is not None else 0,
            abs(row["EV Under"]) if row["EV Under"] is not None else 0,
        )

        html += f"""
        <tr class="away-row game-sep"
            data-ml-ev="{ml_ev_sort}"
            data-ou-ev="{ou_ev_sort}">

            <td class="team-cell">
                <img src="{get_logo(row['Away'])}" style="height:22px;">
                {row['Away']}
            </td>

            <td>{row['ML Away DK']}</td>
            {ml_goal_away}
            {ml_ev_away}

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

        html += f"""
        <tr class="home-row game-sep">
            <td class="team-cell">
                <img src="{get_logo(row['Home'])}" style="height:22px;">
                {row['Home']}
            </td>

            <td>{row['ML Home DK']}</td>
            {ml_goal_home}
            {ml_ev_home}

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

      const rows = Array.from(tbody.querySelectorAll("tr"));
      let matchups = [];
      for (let i = 0; i < rows.length; i++) {
        if (rows[i].classList.contains("away-row")) {
          matchups.push([rows[i], rows[i + 1]]);
          i++;
        }
      }

      const th = headers[colIndex];
      const current = th.getAttribute("data-sort");
      const direction = current === "asc" ? "desc" : "asc";

      headers.forEach(h => {
        h.setAttribute("data-sort", "none");
        const span = h.querySelector(".sort-arrow");
        if (span) span.innerHTML = "";
      });

      th.setAttribute("data-sort", direction);
      const arrow = th.querySelector(".sort-arrow");
      if (arrow) arrow.innerHTML = direction === "asc" ? "▲" : "▼";

      function getVal(row) {
        const cell = row.children[colIndex];
        if (!cell) return 0;
        const v = cell.getAttribute("data-value");
        if (v !== null) return parseFloat(v) || 0;

        let txt = cell.innerText
          .replace("%", "")
          .replace("+", "")
          .replace("−", "-");

        return Math.abs(parseFloat(txt)) || 0;
      }

      matchups.sort((a, b) => {
        const av = getVal(a[0]);
        const bv = getVal(b[0]);
        return direction === "asc" ? av - bv : bv - av;
      });

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
        opp_span = row.select_one("span.table-list-team.opp")
        ou_span = row.select_one("span.table-list-odds.ou")

        if not team_span or not opp_span or not ou_span:
            continue

        result = "W" if "win" in team_span.get("class", []) else "L"
        opp = opp_span.get("data-abbr", "").upper()

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

def main():
    import streamlit.components.v1 as components

    st.title("GTSB Winners Only Board")

    central = pytz.timezone("America/Chicago")
    now_ct = datetime.now(central)

    today_str = now_ct.strftime("%B %d, %Y")
    updated_str = now_ct.strftime("%Y-%m-%d %I:%M:%S %p")

    st.markdown(
        f"""
        <div style="margin-top:-10px;">
            <h2 style="margin-bottom:0px;">{today_str}</h2>
            <p style="font-size:16px; color:gray; margin-top:0px;">Updated: {updated_str}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    games = get_games_today()

    teams_in_today = set()
    for g in games:
        teams_in_today.add(g["away_abbr"])
        teams_in_today.add(g["home_abbr"])

    # FIXED: use NST output directly
    team_stats = compute_team_stats_from_nst()

    # If NST returns full names instead of abbreviations, convert only when needed
    normalized_team_stats = {}
    for key, stats in team_stats.items():
        raw_key = str(key).strip()
        upper_key = raw_key.upper()
        lower_key = raw_key.lower()

        if len(upper_key) in (3, 4):
            normalized_team_stats[upper_key] = stats
        else:
            abbr = ODDS_TEAM_NAME_TO_ABBR.get(lower_key)
            if abbr:
                normalized_team_stats[abbr] = stats
            else:
                print("NST WARNING: No abbreviation found for:", raw_key)

    team_stats = normalized_team_stats

    over_data = fetch_team_over_under()

    for abbr, stats in team_stats.items():
        stats["OverPct"] = over_data.get(abbr, None)

    for abbr, stats in team_stats.items():
        if abbr not in teams_in_today:
            stats["GF/G_last10"] = stats.get("GF/G", DEFAULT_STATS["GF/G"])
            stats["GA/G_last10"] = stats.get("GA/G", DEFAULT_STATS["GA/G"])
            stats["SF/G_last10"] = stats.get("SF/G", DEFAULT_STATS["SF/G"])
            stats["SA/G_last10"] = stats.get("SA/G", DEFAULT_STATS["SA/G"])
            stats["Pace_last10"] = stats.get("Pace (SF+SA)", DEFAULT_STATS["Pace (SF+SA)"])
            continue

        team_games = get_team_game_log_apiweb(abbr)

        if team_games:
            last10 = compute_last10_stats_apiweb(team_games)
            stats.update(last10)
        else:
            stats.update({
                "GF/G_last10": stats.get("GF/G", DEFAULT_STATS["GF/G"]),
                "GA/G_last10": stats.get("GA/G", DEFAULT_STATS["GA/G"]),
                "SF/G_last10": stats.get("SF/G", DEFAULT_STATS["SF/G"]),
                "SA/G_last10": stats.get("SA/G", DEFAULT_STATS["SA/G"]),
                "Pace_last10": stats.get("Pace (SF+SA)", DEFAULT_STATS["Pace (SF+SA)"]),
            })

    RECENCY_WEIGHT = 0.75

    for team, stats in team_stats.items():
        stats["GF_adj"] = blend_recent(
            stats.get("GF/G", DEFAULT_STATS["GF/G"]),
            stats.get("GF/G_last10", stats.get("GF/G", DEFAULT_STATS["GF/G"])),
            RECENCY_WEIGHT
        )
        stats["GA_adj"] = blend_recent(
            stats.get("GA/G", DEFAULT_STATS["GA/G"]),
            stats.get("GA/G_last10", stats.get("GA/G", DEFAULT_STATS["GA/G"])),
            RECENCY_WEIGHT
        )
        stats["xGF_adj"] = stats.get("xGF/G", DEFAULT_STATS["xGF/G"])
        stats["xGA_adj"] = stats.get("xGA/G", DEFAULT_STATS["xGA/G"])
        stats["Pace_adj"] = blend_recent(
            stats.get("Pace (SF+SA)", DEFAULT_STATS["Pace (SF+SA)"]),
            stats.get("Pace_last10", stats.get("Pace (SF+SA)", DEFAULT_STATS["Pace (SF+SA)"])),
            RECENCY_WEIGHT
        )

    st.write("✅ Fetching DraftKings odds…")
    odds_payload = get_odds_draftkings_only()
    odds_idx = build_odds_index(odds_payload)
    ml_odds_idx = build_ml_odds_index(odds_payload)

    rows = []

    for g in games:
        away = g["away_abbr"]
        home = g["home_abbr"]

        key = f"{away}@{home}"
        odds = odds_idx.get(key, {})
        ml_odds = ml_odds_idx.get(key, {})

        away_stats = team_stats.get(away, DEFAULT_STATS.copy())
        home_stats = team_stats.get(home, DEFAULT_STATS.copy())

        predicted = round(predict_total(away_stats, home_stats), 2)

        away_xg, home_xg = split_team_xg_from_total(
            predicted,
            away_stats,
            home_stats
        )

        ml_goal_diff = (
            round(home_xg - away_xg, 2)
            if away_xg is not None and home_xg is not None
            else None
        )

        away_ml_prob = None
        home_ml_prob = None
        away_ml_fair = None
        home_ml_fair = None

        if away_xg is not None and home_xg is not None:
            away_ml_prob = pythagorean_win_prob(away_xg, home_xg, k=2.0)
            home_ml_prob = pythagorean_win_prob(home_xg, away_xg, k=2.0)

        if away_ml_prob is not None and home_ml_prob is not None:
            away_ml_fair = prob_to_american(away_ml_prob)
            home_ml_fair = prob_to_american(home_ml_prob)

        dk_away_ml_dec = ml_odds.get("away_ml_dec")
        dk_home_ml_dec = ml_odds.get("home_ml_dec")

        dk_away_ml_amer = decimal_to_american(dk_away_ml_dec) if dk_away_ml_dec else ""
        dk_home_ml_amer = decimal_to_american(dk_home_ml_dec) if dk_home_ml_dec else ""

        away_ml_implied = decimal_to_prob(dk_away_ml_dec) if dk_away_ml_dec else None
        home_ml_implied = decimal_to_prob(dk_home_ml_dec) if dk_home_ml_dec else None

        away_ml_edge_pct = None
        home_ml_edge_pct = None

        if away_ml_prob is not None and away_ml_implied is not None:
            away_ml_edge_pct = (away_ml_prob - away_ml_implied) * 100.0

        if home_ml_prob is not None and home_ml_implied is not None:
            home_ml_edge_pct = (home_ml_prob - home_ml_implied) * 100.0

        away_ml_ev = compute_ev_ml(away_ml_prob, dk_away_ml_dec)
        home_ml_ev = compute_ev_ml(home_ml_prob, dk_home_ml_dec)

        ev_a = away_ml_ev if away_ml_ev is not None else -999
        ev_h = home_ml_ev if home_ml_ev is not None else -999

        _ML_EV_MAX = max(ev_a, ev_h)
        _ML_EV_SIDE = "AWAY" if ev_a >= ev_h else "HOME"

        edge_a = abs(away_ml_edge_pct) if away_ml_edge_pct is not None else -999
        edge_h = abs(home_ml_edge_pct) if home_ml_edge_pct is not None else -999

        _ML_EDGE_ABS = max(edge_a, edge_h)

        model_pick = ""
        dist = None
        sportsbook_line = None

        line = odds.get("line", "")
        over_dec = odds.get("over")
        under_dec = odds.get("under")

        edge = ""

        if line not in ("", None):
            try:
                sportsbook_line = float(line)
                dist = predicted - sportsbook_line

                if dist > 0:
                    model_pick = "OVER"
                elif dist < 0:
                    model_pick = "UNDER"
                else:
                    model_pick = ""

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

        ev_over = compute_ev(predicted, sportsbook_line, over_dec, "OVER")
        ev_under = compute_ev(predicted, sportsbook_line, under_dec, "UNDER")

        away_over_pct = team_stats.get(away, {}).get("OverPct")
        home_over_pct = team_stats.get(home, {}).get("OverPct")

        def fmt_pct(x):
            try:
                return round(float(x) * 100)
            except:
                return ""

        away_over_pct_fmt = fmt_pct(away_over_pct)
        home_over_pct_fmt = fmt_pct(home_over_pct)

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
            "_EV_MAX": max(ev_over if pd.notna(ev_over) else -999,
                           ev_under if pd.notna(ev_under) else -999),
            "_EV_SIDE": "OVER" if (ev_over or -999) >= (ev_under or -999) else "UNDER",
            "_DIST": abs(dist) if dist is not None else -999,
            "_MODEL_PICK": model_pick,
            "ML Away xG": away_xg,
            "ML Home xG": home_xg,
            "ML Goal Diff": ml_goal_diff,
            "ML Away Prob": away_ml_prob,
            "ML Home Prob": home_ml_prob,
            "ML Away Fair": away_ml_fair,
            "ML Home Fair": home_ml_fair,
            "ML Away DK": dk_away_ml_amer,
            "ML Home DK": dk_home_ml_amer,
            "ML Away EV": away_ml_ev,
            "ML Home EV": home_ml_ev,
            "_ML_EV_MAX": _ML_EV_MAX,
            "_ML_SIDE": _ML_EV_SIDE,
            "_ML_EDGE_ABS": _ML_EDGE_ABS,
        })

    if not rows:
        st.warning("No pregame NHL matchups found for today, or the feeds did not return usable data.")
        return

    df = pd.DataFrame(rows)

    best_bet_idx = df["_EV_MAX"].idxmax()
    best_value_idx = df["_DIST"].idxmax()

    current = df.at[best_bet_idx, "Model Pick"]
    if not isinstance(current, str):
        current = ""
    df.at[best_bet_idx, "Model Pick"] = current.rstrip() + " ⭐"

    current = df.at[best_value_idx, "Model Pick"]
    if not isinstance(current, str):
        current = ""
    df.at[best_value_idx, "Model Pick"] = current.rstrip() + " ◆"

    df = df.drop(columns=["_EV_MAX", "_EV_SIDE", "_DIST", "_MODEL_PICK"])

    # Ensure numeric types for formatting in HTML
    df["Line"] = pd.to_numeric(df["Line"], errors="coerce")
    df["Proj"] = pd.to_numeric(df["Proj"], errors="coerce")
    df["EV Over"] = pd.to_numeric(df["EV Over"], errors="coerce")
    df["EV Under"] = pd.to_numeric(df["EV Under"], errors="coerce")
    st.subheader("Today's Games")

    html_table = render_two_row_table(df)
    components.html(html_table, height=900, scrolling=True)

    st.markdown("---")
    st.header("🔍 Explain Matchup")

    matchup_map = {
        f"{g['away_abbr']} @ {g['home_abbr']}": g
        for g in games
    }

    if not matchup_map:
        st.info("No matchups available to explain.")
        return

    default_matchup = list(matchup_map.keys())[0]
    if "⭐" in df["Model Pick"].astype(str).to_string():
        for _, row in df.iterrows():
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

    away_stats = team_stats.get(away, DEFAULT_STATS.copy())
    home_stats = team_stats.get(home, DEFAULT_STATS.copy())

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

        if abs(delta) < 0.05:
            return "<span style='color:gray;'>— 0.00</span>"

        if delta > 0:
            return f"<span style='color:green; font-weight:600;'>▲ +{delta}</span>"

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
            result = g["result"]
            score = g["score"]
            opp = g["opp"]
            ou = g.get("ou")

            if result == "W":
                res_html = "<span style='color:#2E7D32; font-weight:600;'>W</span>"
            else:
                res_html = "<span style='color:#C62828; font-weight:600;'>L</span>"

            if ou == "O":
                ou_html = " (OVER)"
            elif ou == "U":
                ou_html = " (UNDER)"
            else:
                ou_html = ""

            out.append(f"{res_html} {score} vs {opp}{ou_html}")

        return out

    colA, colB = st.columns(2)

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
