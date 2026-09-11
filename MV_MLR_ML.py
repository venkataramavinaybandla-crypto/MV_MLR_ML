import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
from sklearn.linear_model import LogisticRegression

ALIASES = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Deccan Chargers': 'Sunrisers Hyderabad',
    'Kings XI Punjab': 'Punjab Kings',
    'Rising Pune Supergiants': 'Rising Pune Supergiant',
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
}

HOME = {
    'Mumbai Indians': ['Mumbai'],
    'Chennai Super Kings': ['Chennai'],
    'Kolkata Knight Riders': ['Kolkata'],
    'Royal Challengers Bengaluru': ['Bangalore', 'Bengaluru'],
    'Delhi Capitals': ['Delhi'],
    'Punjab Kings': ['Chandigarh', 'Mohali'],
    'Rajasthan Royals': ['Jaipur'],
    'Sunrisers Hyderabad': ['Hyderabad'],
    'Rising Pune Supergiant': ['Pune'],
    'Pune Warriors': ['Pune'],
    'Gujarat Lions': ['Rajkot'],
    'Gujarat Titans': ['Ahmedabad'],
    'Kochi Tuskers Kerala': ['Kochi'],
    'Lucknow Super Giants': ['Lucknow'],
}

TEAM_COLORS = {
    'Mumbai Indians': '#0057A6',
    'Chennai Super Kings': '#F2C21C',
    'Kolkata Knight Riders': '#5B2A86',
    'Royal Challengers Bengaluru': '#D9232E',
    'Delhi Capitals': '#1F5BB5',
    'Punjab Kings': '#DC1F2E',
    'Rajasthan Royals': '#3B5AA6',
    'Sunrisers Hyderabad': '#F0641E',
    'Gujarat Titans': '#1B2133',
    'Lucknow Super Giants': '#A3225B',
    'Rising Pune Supergiant': '#6A2E8C',
    'Pune Warriors': '#6A2E8C',
    'Gujarat Lions': '#E0521A',
    'Kochi Tuskers Kerala': '#2E8B57',
}

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'matches.csv')

def is_home(team, city):
    cities = HOME.get(team)
    if cities is None or pd.isna(city):
        return 0
    return int(city in cities)

def build_dataset():
    df = pd.read_csv(DATA_PATH)
    for c in ['team1', 'team2', 'toss_winner', 'winner']:
        df[c] = df[c].replace(ALIASES)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    stats = {}
    h2h = {}
    rows = []
    for _, r in df.iterrows():
        t1, t2, w = r['team1'], r['team2'], r['winner']
        toss_w, toss_d, city = r['toss_winner'], r['toss_decision'], r['city']
        s1 = stats.setdefault(t1, {'w': 0, 'p': 0, 'form': []})
        s2 = stats.setdefault(t2, {'w': 0, 'p': 0, 'form': []})
        key = tuple(sorted([t1, t2]))
        hh = h2h.setdefault(key, {t1: 0, t2: 0, 'p': 0})
        t1_wr = s1['w'] / s1['p'] if s1['p'] else 0.5
        t2_wr = s2['w'] / s2['p'] if s2['p'] else 0.5
        t1_form = np.mean(s1['form'][-5:]) if s1['form'] else 0.5
        t2_form = np.mean(s2['form'][-5:]) if s2['form'] else 0.5
        t1_h2h = hh.get(t1, 0) / hh['p'] if hh['p'] else 0.5
        if pd.notna(w):
            label = 0 if w == t1 else (1 if w == t2 else 2)
        else:
            label = 2
        rows.append({
            't1_wr': t1_wr, 't2_wr': t2_wr, 't1_form': t1_form, 't2_form': t2_form,
            't1_h2h': t1_h2h, 't1_home': is_home(t1, city), 't2_home': is_home(t2, city),
            'toss_t1': int(toss_w == t1), 'toss_bat': int(toss_d == 'bat'), 'label': label,
        })
        for team, s, won in [(t1, s1, w == t1), (t2, s2, w == t2)]:
            s['p'] += 1
            if won:
                s['w'] += 1
            s['form'].append(1 if won else 0)
        if pd.notna(w) and w in (t1, t2):
            hh[w] = hh.get(w, 0) + 1
        hh['p'] += 1
    return pd.DataFrame(rows), stats, h2h, sorted(set(df['team1']) | set(df['team2'])), sorted(df['city'].dropna().unique().tolist())

FEATURES, STATS, H2H, TEAMS, CITIES = build_dataset()
X = FEATURES.drop(columns='label')
y = FEATURES['label']
MODEL = LogisticRegression(max_iter=1000)
MODEL.fit(X, y)

def team_stat(team):
    s = STATS.get(team, {'w': 0, 'p': 0, 'form': []})
    wr = s['w'] / s['p'] if s['p'] else 0.5
    form = np.mean(s['form'][-5:]) if s['form'] else 0.5
    return wr, form

def head_to_head(t1, t2):
    key = tuple(sorted([t1, t2]))
    hh = H2H.get(key, {t1: 0, t2: 0, 'p': 0})
    return hh.get(t1, 0) / hh['p'] if hh['p'] else 0.5

app = Flask(__name__)

SHARED_HEAD = """
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=Barlow+Condensed:wght@500;600;700&display=swap" rel="stylesheet">
<style>
:root{--accent1:#0057A6;--accent2:#D9232E;--bg:#F4EEE2;--ink:#211E1B;--panel:#FFFFFF;}
*{box-sizing:border-box;}
body{margin:0;background:var(--bg);color:var(--ink);font-family:'Barlow Condensed',sans-serif;}
nav{display:flex;align-items:center;gap:32px;padding:18px 6vw;border-bottom:3px solid var(--ink);background:var(--panel);}
nav .brand{font-family:'Fraunces',serif;font-size:1.6rem;margin:0;letter-spacing:-1px;text-decoration:none;color:var(--ink);}
nav a{font-size:1.05rem;text-transform:uppercase;letter-spacing:1.5px;color:#655f56;text-decoration:none;padding:6px 0;border-bottom:2px solid transparent;transition:color .2s,border-color .2s;}
nav a:hover,nav a.active{color:var(--ink);border-bottom-color:var(--ink);}
header{padding:38px 6vw 18px;border-bottom:3px solid var(--ink);}
header h1{font-family:'Fraunces',serif;font-size:2.6rem;margin:0;letter-spacing:-1px;}
header p{margin:6px 0 0;font-size:1.1rem;text-transform:uppercase;letter-spacing:2px;color:#655f56;}
main{max-width:980px;margin:0 auto;padding:40px 6vw 80px;}
.card{background:var(--panel);border:2px solid var(--ink);border-radius:2px;padding:28px 30px;margin-bottom:28px;box-shadow:8px 8px 0 rgba(33,30,27,0.08);}
.card h2{font-family:'Fraunces',serif;font-size:1.4rem;margin-top:0;}
.row{display:flex;gap:24px;flex-wrap:wrap;}
.field{flex:1;min-width:200px;}
label{display:block;font-size:.95rem;text-transform:uppercase;letter-spacing:1.5px;color:#655f56;margin-bottom:6px;}
select{width:100%;padding:12px;font-size:1.1rem;font-family:'Barlow Condensed',sans-serif;border:2px solid var(--ink);border-radius:2px;background:#fff;}
button{margin-top:20px;padding:14px 32px;font-size:1.2rem;font-family:'Fraunces',serif;font-weight:700;background:var(--ink);color:#fff;border:none;border-radius:2px;cursor:pointer;letter-spacing:1px;}
button:hover{background:#3a352f;}
.vs{display:flex;align-items:center;justify-content:center;font-family:'Fraunces',serif;font-size:2rem;color:#b4ab9c;}
#result{display:none;}
.bar-row{margin:18px 0;}
.bar-label{display:flex;justify-content:space-between;font-size:1.05rem;margin-bottom:6px;}
.bar-track{height:26px;background:#eee5d4;border:2px solid var(--ink);border-radius:2px;overflow:hidden;}
.bar-fill{height:100%;width:0;transition:width .8s ease;}
.verdict{font-family:'Fraunces',serif;font-size:1.6rem;text-align:center;padding:18px;margin-top:10px;border:3px solid var(--ink);}
</style>
"""

NAV = """
<nav>
  <a href="/" class="brand">MatchState</a>
  <a href="/" class="{{ 'active' if active == 'home' else '' }}">Home</a>
  <a href="/simulation" class="{{ 'active' if active == 'simulation' else '' }}">Simulation</a>
  <a href="/about" class="{{ 'active' if active == 'about' else '' }}">About Us</a>
</nav>
"""

HOME_PAGE = """
<!doctype html>
<html>
<head>
{{ head|safe }}
<title>MatchState</title>
</head>
<body>
{{ nav|safe }}
<main>
<div class="card">
<h2>Set the fixture</h2>
<div class="row">
<div class="field">
<label>Team One</label>
<select id="team1"></select>
</div>
<div class="vs">VS</div>
<div class="field">
<label>Team Two</label>
<select id="team2"></select>
</div>
</div>
<div class="row" style="margin-top:20px;">
<div class="field">
<label>Venue City</label>
<select id="city"></select>
</div>
<div class="field">
<label>Toss Winner</label>
<select id="toss_winner"></select>
</div>
<div class="field">
<label>Toss Decision</label>
<select id="toss_decision">
<option value="bat">Bat</option>
<option value="field">Field</option>
</select>
</div>
</div>
<button onclick="predict()">Predict</button>
</div>
<div class="card" id="result">
<h2>Probability Breakdown</h2>
<div class="bar-row">
<div class="bar-label"><span id="t1name"></span><span id="t1pct"></span></div>
<div class="bar-track"><div class="bar-fill" id="t1bar"></div></div>
</div>
<div class="bar-row">
<div class="bar-label"><span id="t2name"></span><span id="t2pct"></span></div>
<div class="bar-track"><div class="bar-fill" id="t2bar"></div></div>
</div>
<div class="bar-row">
<div class="bar-label"><span>No Result / Tie</span><span id="drpct"></span></div>
<div class="bar-track"><div class="bar-fill" id="drbar" style="background:#b4ab9c;"></div></div>
</div>
<div class="verdict" id="verdict"></div>
</div>
</main>
<script>
const TEAMS = {{ teams|tojson }};
const CITIES = {{ cities|tojson }};
const COLORS = {{ colors|tojson }};
let syncing = false;
function fill(id, opts){
  const el = document.getElementById(id);
  el.innerHTML = opts.map(o => `<option value="${o}">${o}</option>`).join('');
}
function fillTeamSelect(id, exclude, preserve){
  const el = document.getElementById(id);
  const opts = TEAMS.filter(t => t !== exclude);
  el.innerHTML = opts.map(o => `<option value="${o}">${o}</option>`).join('');
  if (preserve && opts.includes(preserve)) el.value = preserve;
}
function syncTeams(){
  if (syncing) return;
  syncing = true;
  const t1 = document.getElementById('team1').value;
  const t2 = document.getElementById('team2').value;
  fillTeamSelect('team1', t2, t1);
  fillTeamSelect('team2', t1, t2);
  const newT1 = document.getElementById('team1').value;
  const newT2 = document.getElementById('team2').value;
  if (newT1 === newT2) {
    const alt = TEAMS.find(t => t !== newT1);
    document.getElementById('team2').value = alt;
  }
  syncing = false;
  syncToss();
}
fill('city', CITIES);
fillTeamSelect('team2', TEAMS[0], TEAMS[1]);
fillTeamSelect('team1', document.getElementById('team2').value, TEAMS[0]);
function syncToss(){
  const t1 = document.getElementById('team1').value;
  const t2 = document.getElementById('team2').value;
  fill('toss_winner', [t1, t2]);
}
document.getElementById('team1').addEventListener('change', syncTeams);
document.getElementById('team2').addEventListener('change', syncTeams);
syncToss();
async function predict(){
  const payload = {
    team1: document.getElementById('team1').value,
    team2: document.getElementById('team2').value,
    city: document.getElementById('city').value,
    toss_winner: document.getElementById('toss_winner').value,
    toss_decision: document.getElementById('toss_decision').value,
  };
  const res = await fetch('/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  const data = await res.json();
  const c1 = COLORS[payload.team1] || '#0057A6';
  const c2 = COLORS[payload.team2] || '#D9232E';
  document.getElementById('t1name').innerText = payload.team1;
  document.getElementById('t2name').innerText = payload.team2;
  document.getElementById('t1pct').innerText = data.team1_win + '%';
  document.getElementById('t2pct').innerText = data.team2_win + '%';
  document.getElementById('drpct').innerText = data.draw + '%';
  document.getElementById('t1bar').style.width = data.team1_win + '%';
  document.getElementById('t1bar').style.background = c1;
  document.getElementById('t2bar').style.width = data.team2_win + '%';
  document.getElementById('t2bar').style.background = c2;
  document.getElementById('drbar').style.width = data.draw + '%';
  document.getElementById('verdict').innerText = data.verdict;
  document.getElementById('result').style.display = 'block';
}
</script>
</body>
</html>
"""

ABOUT_PAGE = """
<!doctype html>
<html>
<head>
{{ head|safe }}
<title>About Us - MatchState</title>
</head>
<body>
{{ nav|safe }}
<main>
<div class="card">
<h2>About Us</h2>
<p style="font-size:1.05rem;line-height:1.7;color:#444;margin:10px 0 16px;">
<strong>Project Title:</strong> Multi-Class Match Result Prediction Using Multinomial Logistic Regression<br>
<strong>Course:</strong> MACHINE LEARNING (25SC2107E) &mdash; Y25 2026-2027, Trimester 4<br>
<strong>Team No:</strong> 21 &nbsp;|&nbsp; <strong>Section:</strong> 09
</p>
<div style="display:flex;gap:32px;flex-wrap:wrap;margin-bottom:18px;">
<div style="flex:1;min-width:200px;">
<div style="background:var(--ink);color:#fff;padding:14px 18px;border-radius:2px;margin-bottom:8px;">
<div style="font-family:'Fraunces',serif;font-size:1.1rem;">VINAY BANDLA</div>
<div style="font-size:.9rem;color:#b4ab9c;">2520030437</div>
</div>
</div>
<div style="flex:1;min-width:200px;">
<div style="background:var(--ink);color:#fff;padding:14px 18px;border-radius:2px;margin-bottom:8px;">
<div style="font-family:'Fraunces',serif;font-size:1.1rem;">S. PUJITH PAVAN KUMAR</div>
<div style="font-size:.9rem;color:#b4ab9c;">2520030029</div>
</div>
</div>
</div>
<h3 style="font-family:'Fraunces',serif;font-size:1.15rem;margin:18px 0 8px;">Abstract</h3>
<p style="font-size:1rem;line-height:1.7;color:#444;">
Predicting the outcome of a sports fixture is a problem of considerable interest to analysts, broadcasters, and fantasy sports platforms, yet most existing approaches rely on subjective judgement rather than a systematic, data-driven methodology. This project addresses that gap by developing a Multi-Class Match Result Prediction System using Multinomial Logistic Regression &mdash; a statistical classification technique capable of modelling more than two outcome categories simultaneously.
</p>
<p style="font-size:1rem;line-height:1.7;color:#444;">
Unlike conventional binary classifiers restricted to a win/loss outcome, the model predicts three possible match results &mdash; <strong>win</strong>, <strong>loss</strong>, or <strong>draw</strong> &mdash; by estimating the probability associated with each class. The system is trained on historical IPL match data comprising features such as team performance metrics, home and away advantage, recent form, and head-to-head statistics.
</p>
<p style="font-size:1rem;line-height:1.7;color:#444;">
By employing an interpretable statistical model rather than a complex black-box algorithm, the project aims to deliver transparent, explainable, and reasonably accurate match outcome predictions, contributing a practical application of machine learning within the sports and entertainment domain.
</p>
<h3 style="font-family:'Fraunces',serif;font-size:1.15rem;margin:18px 0 8px;">Tools &amp; Technologies</h3>
<div style="display:flex;flex-wrap:wrap;gap:8px;">
<span style="background:var(--ink);color:#fff;padding:6px 14px;border-radius:2px;font-size:.9rem;">Python</span>
<span style="background:var(--ink);color:#fff;padding:6px 14px;border-radius:2px;font-size:.9rem;">NumPy</span>
<span style="background:var(--ink);color:#fff;padding:6px 14px;border-radius:2px;font-size:.9rem;">Pandas</span>
<span style="background:var(--ink);color:#fff;padding:6px 14px;border-radius:2px;font-size:.9rem;">Scikit-learn</span>
<span style="background:var(--ink);color:#fff;padding:6px 14px;border-radius:2px;font-size:.9rem;">Flask</span>
<span style="background:var(--ink);color:#fff;padding:6px 14px;border-radius:2px;font-size:.9rem;">Matplotlib</span>
<span style="background:var(--ink);color:#fff;padding:6px 14px;border-radius:2px;font-size:.9rem;">Seaborn</span>
<span style="background:var(--ink);color:#fff;padding:6px 14px;border-radius:2px;font-size:.9rem;">Git &amp; GitHub</span>
</div>
</div>
</main>
</body>
</html>
"""

@app.route('/')
def home():
    nav = render_template_string(NAV, active='home')
    return render_template_string(HOME_PAGE, head=SHARED_HEAD, nav=nav, teams=TEAMS, cities=CITIES, colors=TEAM_COLORS)

@app.route('/about')
def about():
    nav = render_template_string(NAV, active='about')
    return render_template_string(ABOUT_PAGE, head=SHARED_HEAD, nav=nav)

SIMULATION_PAGE = """
<!doctype html>
<html>
<head>
{{ head|safe }}
<title>Simulation & Background Math - MatchState</title>
<style>
.pill-badge {
  display: inline-block;
  padding: 4px 10px;
  font-size: 0.82rem;
  font-weight: 700;
  letter-spacing: 1px;
  text-transform: uppercase;
  border: 2px solid var(--ink);
  border-radius: 2px;
  background: #eee5d4;
  margin-right: 6px;
}
.pill-badge.active {
  background: var(--ink);
  color: #fff;
}
.pipeline-nav {
  display: flex;
  gap: 12px;
  overflow-x: auto;
  padding-bottom: 8px;
  margin-bottom: 24px;
}
.pipeline-step {
  display: flex;
  align-items: center;
  gap: 8px;
  background: var(--panel);
  border: 2px solid var(--ink);
  padding: 10px 16px;
  font-size: 0.95rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 1px;
  white-space: nowrap;
  box-shadow: 4px 4px 0 rgba(33,30,27,0.06);
  cursor: pointer;
  transition: all 0.2s;
}
.pipeline-step:hover {
  transform: translate(-1px, -1px);
  box-shadow: 6px 6px 0 rgba(33,30,27,0.1);
}
.pipeline-step .step-num {
  background: var(--ink);
  color: #fff;
  width: 22px;
  height: 22px;
  border-radius: 50%;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 0.75rem;
}
.mode-toggle-bar {
  display: flex;
  gap: 12px;
  margin-bottom: 20px;
  flex-wrap: wrap;
}
.mode-btn {
  padding: 10px 20px;
  font-family: 'Fraunces', serif;
  font-size: 1rem;
  border: 2px solid var(--ink);
  background: #fff;
  color: var(--ink);
  cursor: pointer;
  font-weight: 600;
  box-shadow: 4px 4px 0 rgba(33,30,27,0.08);
  transition: all 0.15s;
}
.mode-btn.active {
  background: var(--ink);
  color: #fff;
}
.math-box {
  background: #faf7f2;
  border: 2px solid var(--ink);
  padding: 18px 20px;
  margin: 16px 0;
  border-radius: 2px;
  font-family: 'Courier New', Consolas, monospace;
  font-size: 0.95rem;
  line-height: 1.6;
  overflow-x: auto;
}
.math-box .eq-title {
  font-family: 'Fraunces', serif;
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--ink);
  margin-bottom: 8px;
}
.math-term {
  display: inline-block;
  padding: 2px 6px;
  border-radius: 2px;
  margin: 2px;
  border: 1px solid #d4cbbd;
  background: #fff;
}
.math-term.pos {
  background: #e8f5e9;
  border-color: #2e7d32;
  color: #1b5e20;
}
.math-term.neg {
  background: #ffebee;
  border-color: #c62828;
  color: #b71c1c;
}
.table-wrap {
  overflow-x: auto;
  margin-top: 14px;
}
table.math-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.95rem;
}
table.math-table th, table.math-table td {
  border: 2px solid var(--ink);
  padding: 10px 12px;
  text-align: left;
}
table.math-table th {
  background: #eee5d4;
  text-transform: uppercase;
  letter-spacing: 1px;
  font-size: 0.85rem;
}
table.math-table td.mono {
  font-family: 'Courier New', Consolas, monospace;
  font-weight: 600;
}
.feature-slider-card {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 12px;
  border: 1.5px solid var(--ink);
  background: #fff;
  border-radius: 2px;
}
.feature-slider-card .header-line {
  display: flex;
  justify-content: space-between;
  font-weight: 700;
  font-size: 0.9rem;
}
.slider-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 16px;
  margin-top: 14px;
}
.waterfall-row {
  display: flex;
  align-items: center;
  margin: 8px 0;
  font-size: 0.92rem;
}
.waterfall-label {
  width: 150px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  font-size: 0.82rem;
}
.waterfall-axis {
  flex: 1;
  height: 28px;
  background: #eee5d4;
  border: 2px solid var(--ink);
  position: relative;
  display: flex;
}
.waterfall-center-line {
  position: absolute;
  left: 50%;
  top: 0;
  bottom: 0;
  width: 2px;
  background: var(--ink);
  z-index: 2;
}
.waterfall-bar-left {
  position: absolute;
  right: 50%;
  height: 100%;
  transition: width 0.3s ease;
}
.waterfall-bar-right {
  position: absolute;
  left: 50%;
  height: 100%;
  transition: width 0.3s ease;
}
.waterfall-val {
  width: 80px;
  text-align: right;
  font-family: 'Courier New', Consolas, monospace;
  font-weight: 700;
  padding-left: 8px;
  font-size: 0.9rem;
}
.prob-card-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
  margin-top: 16px;
}
@media (max-width: 768px) {
  .prob-card-grid { grid-template-columns: 1fr; }
}
.prob-card {
  border: 2px solid var(--ink);
  padding: 18px;
  border-radius: 2px;
  text-align: center;
  background: #fff;
  box-shadow: 4px 4px 0 rgba(33,30,27,0.06);
}
.prob-card .team-title {
  font-family: 'Fraunces', serif;
  font-size: 1.15rem;
  margin-bottom: 6px;
  min-height: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
}
.prob-card .pct-num {
  font-size: 2.2rem;
  font-weight: 700;
  font-family: 'Fraunces', serif;
}
.prob-card .logit-tag {
  font-family: 'Courier New', Consolas, monospace;
  font-size: 0.9rem;
  margin-top: 6px;
  color: #655f56;
}
.highlight-val {
  background: #fff3cd;
  padding: 1px 4px;
  border: 1px dashed #e0a800;
  font-weight: 700;
}
</style>
</head>
<body>
{{ nav|safe }}
<main>

<div class="card" style="border-left: 8px solid var(--ink);">
  <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:16px;">
    <div>
      <span class="pill-badge active">Live Simulation Engine</span>
      <span class="pill-badge">Multinomial Logistic Regression</span>
      <h2 style="font-size:2rem; margin:10px 0 6px;">Live Mathematical Simulation</h2>
      <p style="margin:0; font-size:1.05rem; color:#555; max-width:760px; line-height:1.5;">
        Experience the exact mathematics executing in real-time. As you adjust the match fixture or fine-tune feature sliders, observe live feature extraction (<strong>x</strong>), linear dot products (<em>z<sub>k</sub> = <strong>w</strong><sub>k</sub><sup>T</sup><strong>x</strong> + b<sub>k</sub></em>), and the non-linear Softmax probability transformation (<em>&sigma;(<strong>z</strong>)</em>) with zero delay.
      </p>
    </div>
  </div>

  <div class="pipeline-nav" style="margin-top:24px;">
    <div class="pipeline-step" onclick="scrollToId('sec-fixture')"><span class="step-num">1</span> Controls</div>
    <div class="pipeline-step" onclick="scrollToId('sec-features')"><span class="step-num">2</span> Features x</div>
    <div class="pipeline-step" onclick="scrollToId('sec-matrix')"><span class="step-num">3</span> Weights W</div>
    <div class="pipeline-step" onclick="scrollToId('sec-logits')"><span class="step-num">4</span> Logits z</div>
    <div class="pipeline-step" onclick="scrollToId('sec-softmax')"><span class="step-num">5</span> Softmax &amp; Probabilities</div>
    <div class="pipeline-step" onclick="scrollToId('sec-waterfall')"><span class="step-num">6</span> Feature Impact</div>
  </div>
</div>

<!-- SECTION 1: FIXTURE / SANDBOX CONTROLS -->
<div class="card" id="sec-fixture">
  <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px; margin-bottom:16px;">
    <h2>1. Match Parameters &amp; Simulation Mode</h2>
    <div class="mode-toggle-bar" style="margin:0;">
      <button class="mode-btn active" id="btnModeFixture" onclick="setMode('fixture')">Fixture Mode (IPL Data)</button>
      <button class="mode-btn" id="btnModeSandbox" onclick="setMode('sandbox')">Sandbox Mode (Live Sliders)</button>
    </div>
  </div>

  <!-- Fixture Mode Form -->
  <div id="fixtureControls">
    <div class="row">
      <div class="field">
        <label>Team One</label>
        <select id="sim_team1"></select>
      </div>
      <div class="vs">VS</div>
      <div class="field">
        <label>Team Two</label>
        <select id="sim_team2"></select>
      </div>
    </div>
    <div class="row" style="margin-top:18px;">
      <div class="field">
        <label>Venue City</label>
        <select id="sim_city"></select>
      </div>
      <div class="field">
        <label>Toss Winner</label>
        <select id="sim_toss_winner"></select>
      </div>
      <div class="field">
        <label>Toss Decision</label>
        <select id="sim_toss_decision">
          <option value="bat">Bat</option>
          <option value="field">Field</option>
        </select>
      </div>
    </div>
  </div>

  <!-- Sandbox Mode Sliders -->
  <div id="sandboxControls" style="display:none;">
    <div style="background:#fff3cd; border:2px solid var(--ink); padding:12px 16px; margin-bottom:14px; font-size:0.95rem;">
      <strong>Sandbox Active:</strong> Drag sliders or toggle buttons below to freely manipulate the 9 input features and watch the mathematics react live.
      <button onclick="resetSandbox()" style="margin:0 0 0 16px; padding:4px 12px; font-size:0.85rem; background:var(--ink); color:#fff; cursor:pointer;">Reset to Selected Fixture</button>
    </div>
    <div class="slider-grid" id="sandboxSliders"></div>
  </div>
</div>

<!-- SECTION 2: LIVE FEATURE VECTOR EXTRACTION -->
<div class="card" id="sec-features">
  <h2>2. Live Feature Extraction Vector (x &isin; &reals;<sup>9</sup>)</h2>
  <p style="font-size:1rem; color:#555; margin-top:0;">
    The selected inputs are parsed into a normalized 9-dimensional numerical vector <strong>x</strong> = [<em>x</em><sub>1</sub>, <em>x</em><sub>2</sub>, &hellip;, <em>x</em><sub>9</sub>]<sup>T</sup> fed into the trained Multinomial Logistic Regression model.
  </p>

  <div class="table-wrap">
    <table class="math-table">
      <thead>
        <tr>
          <th>Index</th>
          <th>Feature Name</th>
          <th>Symbol</th>
          <th>Derivation &amp; Formula</th>
          <th>Live Value (x<sub>j</sub>)</th>
        </tr>
      </thead>
      <tbody id="featureVectorTable">
        <!-- Injected via JavaScript -->
      </tbody>
    </table>
  </div>
</div>

<!-- SECTION 3: MODEL COEFFICIENTS & INTERCEPTS -->
<div class="card" id="sec-matrix">
  <h2>3. Learned Model Parameters (W &isin; &reals;<sup>3 &times; 9</sup> and b &isin; &reals;<sup>3</sup>)</h2>
  <p style="font-size:1rem; color:#555; margin-top:0;">
    The Multinomial Logistic Regression classifier was trained on historical IPL match fixtures using Scikit-Learn. It learned optimal weight vectors <strong>w</strong><sub>k</sub> and intercept bias <em>b</em><sub>k</sub> for each of the 3 possible outcomes:
    <strong>Class 0 (Team 1 Win)</strong>, <strong>Class 1 (Team 2 Win)</strong>, and <strong>Class 2 (Draw / Tie)</strong>.
  </p>

  <div class="table-wrap">
    <table class="math-table">
      <thead>
        <tr>
          <th>Class (k)</th>
          <th>Target Outcome</th>
          <th>Bias Intercept (b<sub>k</sub>)</th>
          <th>Weights Vector w<sub>k</sub> = [w<sub>k,1</sub>, &hellip;, w<sub>k,9</sub>]</th>
        </tr>
      </thead>
      <tbody id="weightsMatrixTable">
        <!-- Injected via JavaScript -->
      </tbody>
    </table>
  </div>
</div>

<!-- SECTION 4: LIVE LOGIT CALCULATION -->
<div class="card" id="sec-logits">
  <h2>4. Linear Combination &amp; Logit Calculation (z<sub>k</sub> = w<sub>k</sub><sup>T</sup>x + b<sub>k</sub>)</h2>
  <p style="font-size:1rem; color:#555; margin-top:0;">
    For each class <em>k</em>, the dot product of weights and input features plus intercept produces the raw log-odds score (logit) <em>z</em><sub>k</sub> &isin; (&minus;&infin;, +&infin;):
  </p>

  <div id="logitCalculations"></div>
</div>

<!-- SECTION 5: SOFTMAX NORMALIZATION LAYER -->
<div class="card" id="sec-softmax">
  <h2>5. Softmax Activation &amp; Live Probabilities (&sigma;(z))</h2>
  <p style="font-size:1rem; color:#555; margin-top:0;">
    Because match outcome is mutually exclusive across 3 classes, the multi-class Softmax function exponentiates the logits and normalizes them into a valid probability distribution that sums strictly to 100%:
  </p>

  <div class="math-box">
    <div class="eq-title">Softmax Mathematical Formula</div>
    P(Y = k | x) = exp(z<sub>k</sub>) / &sum;<sub>j=0</sub><sup>2</sup> exp(z<sub>j</sub>) = exp(z<sub>k</sub>) / [exp(z<sub>0</sub>) + exp(z<sub>1</sub>) + exp(z<sub>2</sub>)]
  </div>

  <div class="table-wrap">
    <table class="math-table">
      <thead>
        <tr>
          <th>Class (k)</th>
          <th>Outcome</th>
          <th>Logit (z<sub>k</sub>)</th>
          <th>Exponent (e<sup>z<sub>k</sub></sup>)</th>
          <th>Softmax Fraction e<sup>z<sub>k</sub></sup> / &sum; e<sup>z<sub>j</sub></sup></th>
          <th>Probability (%)</th>
        </tr>
      </thead>
      <tbody id="softmaxTable">
        <!-- Injected via JavaScript -->
      </tbody>
    </table>
  </div>

  <div class="prob-card-grid" id="probCardGrid">
    <!-- Live cards injected -->
  </div>

  <div class="verdict" id="simVerdict" style="margin-top:20px;"></div>
</div>

<!-- SECTION 6: FEATURE IMPACT WATERFALL CHART -->
<div class="card" id="sec-waterfall">
  <h2>6. Live Feature Impact Breakdown (Why is a Team Favoured?)</h2>
  <p style="font-size:1rem; color:#555; margin-top:0;">
    This waterfall comparison shows each feature's contribution toward tilting the balance between Team 1 and Team 2:
    &Delta;z<sub>0&minus;1</sub>(x<sub>j</sub>) = (w<sub>0,j</sub> &minus; w<sub>1,j</sub>) &times; x<sub>j</sub>.
    Positive values push odds towards <strong>Team 1</strong>; negative values push odds towards <strong>Team 2</strong>.
  </p>

  <div style="display:flex; justify-content:space-between; margin-bottom:12px; font-weight:700; font-size:0.95rem;">
    <span id="lblTeam1Fav" style="color:var(--ink);">&larr; Team 1 Advantage</span>
    <span style="color:#777;">Center: Neutral (0.0)</span>
    <span id="lblTeam2Fav" style="color:var(--ink);">Team 2 Advantage &rarr;</span>
  </div>

  <div id="waterfallContainer"></div>
</div>

</main>

<script>
// Parse server context
const TEAMS = {{ teams|tojson }};
const CITIES = {{ cities|tojson }};
const COLORS = {{ colors|tojson }};
const HOME_CITIES = {{ home_cities|tojson }};
const STATS = {{ stats|tojson }};
const H2H = {{ h2h|tojson }};
const MODEL_META = {{ model_meta|tojson }};

const FEATURE_NAMES = MODEL_META.features;
const FEATURE_LABELS = {
  't1_wr': 'Team 1 Win Rate',
  't2_wr': 'Team 2 Win Rate',
  't1_form': 'Team 1 Recent Form (Last 5)',
  't2_form': 'Team 2 Recent Form (Last 5)',
  't1_h2h': 'Head-to-Head Win Rate',
  't1_home': 'Team 1 Home Venue Advantage',
  't2_home': 'Team 2 Home Venue Advantage',
  'toss_t1': 'Toss Won by Team 1',
  'toss_bat': 'Toss Decision: Batting'
};

const FEATURE_SYMBOLS = {
  't1_wr': 'x1',
  't2_wr': 'x2',
  't1_form': 'x3',
  't2_form': 'x4',
  't1_h2h': 'x5',
  't1_home': 'x6',
  't2_home': 'x7',
  'toss_t1': 'x8',
  'toss_bat': 'x9'
};

let currentMode = 'fixture';
let currentVector = {};

function scrollToId(id){
  const el = document.getElementById(id);
  if(el) el.scrollIntoView({ behavior: 'smooth' });
}

function setMode(mode){
  currentMode = mode;
  document.getElementById('btnModeFixture').classList.toggle('active', mode === 'fixture');
  document.getElementById('btnModeSandbox').classList.toggle('active', mode === 'sandbox');
  document.getElementById('fixtureControls').style.display = (mode === 'fixture' ? 'block' : 'none');
  document.getElementById('sandboxControls').style.display = (mode === 'sandbox' ? 'block' : 'none');
  if(mode === 'sandbox'){
    syncSandboxSlidersFromCurrent();
  }
  recomputeAll();
}

function getTeamStat(team){
  const s = STATS[team] || { w: 0, p: 0, form: [] };
  const wr = s.p ? (s.w / s.p) : 0.5;
  const formList = s.form ? s.form.slice(-5) : [];
  const form = formList.length ? (formList.reduce((a,b)=>a+b, 0) / formList.length) : 0.5;
  return { wr, form, w: s.w, p: s.p, formList };
}

function getH2H(t1, t2){
  const sorted = [t1, t2].sort();
  const key = `${sorted[0]} vs ${sorted[1]}`;
  const hh = H2H[key] || { p: 0 };
  const t1Wins = hh[t1] || 0;
  const t2Wins = hh[t2] || 0;
  const rate = hh.p ? (t1Wins / hh.p) : 0.5;
  return { rate, t1Wins, t2Wins, total: hh.p };
}

function checkIsHome(team, city){
  const cities = HOME_CITIES[team];
  if(!cities || !city) return 0;
  return cities.includes(city) ? 1 : 0;
}

function fillSelect(id, opts){
  const el = document.getElementById(id);
  el.innerHTML = opts.map(o => `<option value="${o}">${o}</option>`).join('');
}

function fillTeamSelect(id, exclude, preserve){
  const el = document.getElementById(id);
  const opts = TEAMS.filter(t => t !== exclude);
  el.innerHTML = opts.map(o => `<option value="${o}">${o}</option>`).join('');
  if(preserve && opts.includes(preserve)) el.value = preserve;
}

let syncing = false;
function syncTeams(){
  if(syncing) return;
  syncing = true;
  const t1 = document.getElementById('sim_team1').value;
  const t2 = document.getElementById('sim_team2').value;
  fillTeamSelect('sim_team1', t2, t1);
  fillTeamSelect('sim_team2', t1, t2);
  const newT1 = document.getElementById('sim_team1').value;
  const newT2 = document.getElementById('sim_team2').value;
  if(newT1 === newT2){
    const alt = TEAMS.find(t => t !== newT1);
    document.getElementById('sim_team2').value = alt;
  }
  syncing = false;
  syncToss();
  recomputeAll();
}

function syncToss(){
  const t1 = document.getElementById('sim_team1').value;
  const t2 = document.getElementById('sim_team2').value;
  const curr = document.getElementById('sim_toss_winner').value;
  fillSelect('sim_toss_winner', [t1, t2]);
  if(curr === t1 || curr === t2) document.getElementById('sim_toss_winner').value = curr;
}

function extractFeaturesFromFixture(){
  const t1 = document.getElementById('sim_team1').value;
  const t2 = document.getElementById('sim_team2').value;
  const city = document.getElementById('sim_city').value;
  const toss_w = document.getElementById('sim_toss_winner').value;
  const toss_d = document.getElementById('sim_toss_decision').value;

  const s1 = getTeamStat(t1);
  const s2 = getTeamStat(t2);
  const hh = getH2H(t1, t2);
  const t1_home = checkIsHome(t1, city);
  const t2_home = checkIsHome(t2, city);
  const toss_t1 = (toss_w === t1 ? 1 : 0);
  const toss_bat = (toss_d === 'bat' ? 1 : 0);

  return {
    t1_wr: { val: s1.wr, note: `${s1.w} wins / ${s1.p} matches (${(s1.wr*100).toFixed(1)}%)` },
    t2_wr: { val: s2.wr, note: `${s2.w} wins / ${s2.p} matches (${(s2.wr*100).toFixed(1)}%)` },
    t1_form: { val: s1.form, note: `Last 5 matches: [${s1.formList.join(', ')}] = ${(s1.form*100).toFixed(0)}%` },
    t2_form: { val: s2.form, note: `Last 5 matches: [${s2.formList.join(', ')}] = ${(s2.form*100).toFixed(0)}%` },
    t1_h2h: { val: hh.rate, note: `${hh.t1Wins} wins vs ${hh.t2Wins} in ${hh.total} encounters` },
    t1_home: { val: t1_home, note: t1_home ? `${city} is registered home venue` : `Away venue` },
    t2_home: { val: t2_home, note: t2_home ? `${city} is registered home venue` : `Away venue` },
    toss_t1: { val: toss_t1, note: toss_t1 ? `${t1} won coin toss` : `${t2} won coin toss` },
    toss_bat: { val: toss_bat, note: toss_bat ? `Chose to bat first` : `Chose to field first` },
  };
}

function initSandboxSliders(){
  const container = document.getElementById('sandboxSliders');
  container.innerHTML = FEATURE_NAMES.map((fn, idx) => {
    const isBinary = ['t1_home', 't2_home', 'toss_t1', 'toss_bat'].includes(fn);
    const step = isBinary ? '1' : '0.01';
    const min = '0';
    const max = '1';
    return `
      <div class="feature-slider-card">
        <div class="header-line">
          <span>${FEATURE_SYMBOLS[fn]}: ${FEATURE_LABELS[fn]}</span>
          <span class="highlight-val" id="sb_val_${fn}">0.50</span>
        </div>
        <input type="range" id="sb_range_${fn}" min="${min}" max="${max}" step="${step}" value="0.5"
               oninput="onSliderChange('${fn}')" style="width:100%; accent-color:var(--ink); cursor:pointer;">
        <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#777;">
          <span>${isBinary ? '0 (No)' : '0.00'}</span>
          <span>${isBinary ? '1 (Yes)' : '1.00'}</span>
        </div>
      </div>
    `;
  }).join('');
}

function syncSandboxSlidersFromCurrent(){
  const ext = extractFeaturesFromFixture();
  FEATURE_NAMES.forEach(fn => {
    const val = ext[fn].val;
    const slider = document.getElementById(`sb_range_${fn}`);
    const display = document.getElementById(`sb_val_${fn}`);
    if(slider) slider.value = val;
    if(display) display.innerText = Number(val).toFixed(2);
  });
}

function resetSandbox(){
  syncSandboxSlidersFromCurrent();
  recomputeAll();
}

function onSliderChange(fn){
  const slider = document.getElementById(`sb_range_${fn}`);
  const display = document.getElementById(`sb_val_${fn}`);
  if(slider && display){
    display.innerText = Number(slider.value).toFixed(2);
  }
  recomputeAll();
}

function recomputeAll(){
  const t1 = document.getElementById('sim_team1').value;
  const t2 = document.getElementById('sim_team2').value;
  const c1 = COLORS[t1] || '#0057A6';
  const c2 = COLORS[t2] || '#D9232E';

  document.getElementById('lblTeam1Fav').innerHTML = `&larr; <strong style="color:${c1}">${t1}</strong> Advantage`;
  document.getElementById('lblTeam2Fav').innerHTML = `<strong style="color:${c2}">${t2}</strong> Advantage &rarr;`;

  let xVec = {};
  let xDerivations = {};

  if(currentMode === 'fixture'){
    const ext = extractFeaturesFromFixture();
    FEATURE_NAMES.forEach(fn => {
      xVec[fn] = ext[fn].val;
      xDerivations[fn] = ext[fn].note;
    });
  } else {
    FEATURE_NAMES.forEach(fn => {
      const slider = document.getElementById(`sb_range_${fn}`);
      const val = slider ? parseFloat(slider.value) : 0.5;
      xVec[fn] = val;
      xDerivations[fn] = `User Sandbox Override: ${val.toFixed(2)}`;
    });
  }
  currentVector = xVec;

  // Feature vector table
  const featTableBody = document.getElementById('featureVectorTable');
  featTableBody.innerHTML = FEATURE_NAMES.map((fn, idx) => {
    const val = xVec[fn];
    return `
      <tr>
        <td class="mono">x[${idx}]</td>
        <td><strong>${FEATURE_LABELS[fn]}</strong></td>
        <td class="mono" style="font-weight:700;">${FEATURE_SYMBOLS[fn]}</td>
        <td style="color:#555;">${xDerivations[fn]}</td>
        <td class="mono highlight-val">${Number(val).toFixed(4)}</td>
      </tr>
    `;
  }).join('');

  // Weights Matrix
  const W = MODEL_META.coefficients;
  const b = MODEL_META.intercepts;
  const classNames = [
    `${t1} (Team 1 Win)`,
    `${t2} (Team 2 Win)`,
    `Tie / No Result`
  ];

  const weightsTableBody = document.getElementById('weightsMatrixTable');
  weightsTableBody.innerHTML = [0, 1, 2].map(k => {
    const wStr = W[k].map(w => (w >= 0 ? `+${w.toFixed(3)}` : w.toFixed(3))).join(', ');
    return `
      <tr>
        <td class="mono" style="font-weight:700;">k = ${k}</td>
        <td><strong>${classNames[k]}</strong></td>
        <td class="mono">${b[k].toFixed(4)}</td>
        <td class="mono" style="font-size:0.88rem;">[${wStr}]</td>
      </tr>
    `;
  }).join('');

  // Compute Logits
  let logits = [0, 0, 0];
  let dotBreakdown = [[], [], []];

  for(let k = 0; k < 3; k++){
    let sum = b[k];
    for(let j = 0; j < FEATURE_NAMES.length; j++){
      const fn = FEATURE_NAMES[j];
      const prod = W[k][j] * xVec[fn];
      sum += prod;
      dotBreakdown[k].push({
        fn,
        symbol: FEATURE_SYMBOLS[fn],
        weight: W[k][j],
        val: xVec[fn],
        prod: prod
      });
    }
    logits[k] = sum;
  }

  // Render Logit Equations
  const logitDiv = document.getElementById('logitCalculations');
  logitDiv.innerHTML = [0, 1, 2].map(k => {
    const classColor = (k === 0 ? c1 : (k === 1 ? c2 : '#888'));
    const termsHtml = dotBreakdown[k].map(d => {
      const cls = d.prod >= 0 ? 'pos' : 'neg';
      return `<span class="math-term ${cls}">(${d.weight.toFixed(3)} &times; ${d.val.toFixed(2)} = ${d.prod.toFixed(3)})</span>`;
    }).join(' + ');

    return `
      <div class="math-box" style="border-left:6px solid ${classColor};">
        <div class="eq-title" style="color:${classColor};">
          Class ${k}: ${classNames[k]} &mdash; Logit z<sub>${k}</sub>
        </div>
        <div style="margin-bottom:6px;">
          <strong>z<sub>${k}</sub></strong> = b<sub>${k}</sub> + &sum;(w<sub>${k},j</sub> &times; x<sub>j</sub>)
        </div>
        <div style="font-size:0.88rem; color:#444; margin-bottom:8px;">
          = <span class="math-term">${b[k].toFixed(4)}</span> + ${termsHtml}
        </div>
        <div style="font-size:1.1rem; font-weight:700;">
          Result: <span style="background:${classColor}; color:#fff; padding:2px 8px; border-radius:2px;">z<sub>${k}</sub> = ${logits[k].toFixed(4)}</span>
        </div>
      </div>
    `;
  }).join('');

  // Compute Softmax
  const maxZ = Math.max(...logits);
  const expZ = logits.map(z => Math.exp(z - maxZ));
  const sumExp = expZ.reduce((a, b) => a + b, 0);
  const probs = expZ.map(e => e / sumExp);
  const pct = probs.map(p => (p * 100));

  // Render Softmax Table
  const softmaxTableBody = document.getElementById('softmaxTable');
  softmaxTableBody.innerHTML = [0, 1, 2].map(k => {
    const teamCol = (k === 0 ? c1 : (k === 1 ? c2 : '#888'));
    return `
      <tr>
        <td class="mono">k = ${k}</td>
        <td style="font-weight:700; color:${teamCol};">${classNames[k]}</td>
        <td class="mono">${logits[k].toFixed(4)}</td>
        <td class="mono">${expZ[k].toFixed(4)}</td>
        <td class="mono">${expZ[k].toFixed(4)} / ${sumExp.toFixed(4)}</td>
        <td class="mono" style="font-size:1.15rem; font-weight:700; color:${teamCol};">${pct[k].toFixed(1)}%</td>
      </tr>
    `;
  }).join('');

  // Probability Cards
  const probCards = document.getElementById('probCardGrid');
  probCards.innerHTML = `
    <div class="prob-card" style="border-color:${c1}; border-top:6px solid ${c1};">
      <div class="team-title" style="color:${c1};">${t1}</div>
      <div class="pct-num" style="color:${c1};">${pct[0].toFixed(1)}%</div>
      <div class="logit-tag">Logit z<sub>0</sub>: ${logits[0].toFixed(3)}</div>
      <div class="bar-track" style="margin-top:12px;"><div class="bar-fill" style="width:${pct[0].toFixed(1)}%; background:${c1};"></div></div>
    </div>
    <div class="prob-card" style="border-color:${c2}; border-top:6px solid ${c2};">
      <div class="team-title" style="color:${c2};">${t2}</div>
      <div class="pct-num" style="color:${c2};">${pct[1].toFixed(1)}%</div>
      <div class="logit-tag">Logit z<sub>1</sub>: ${logits[1].toFixed(3)}</div>
      <div class="bar-track" style="margin-top:12px;"><div class="bar-fill" style="width:${pct[1].toFixed(1)}%; background:${c2};"></div></div>
    </div>
    <div class="prob-card" style="border-color:#888; border-top:6px solid #888;">
      <div class="team-title" style="color:#666;">No Result / Tie</div>
      <div class="pct-num" style="color:#666;">${pct[2].toFixed(1)}%</div>
      <div class="logit-tag">Logit z<sub>2</sub>: ${logits[2].toFixed(3)}</div>
      <div class="bar-track" style="margin-top:12px;"><div class="bar-fill" style="width:${pct[2].toFixed(1)}%; background:#b4ab9c;"></div></div>
    </div>
  `;

  // Verdict
  let verdictText = '';
  if(pct[0] > pct[1] && pct[0] > pct[2]){
    verdictText = `${t1} favoured (${pct[0].toFixed(1)}% vs ${pct[1].toFixed(1)}%)`;
  } else if(pct[1] > pct[0] && pct[1] > pct[2]){
    verdictText = `${t2} favoured (${pct[1].toFixed(1)}% vs ${pct[0].toFixed(1)}%)`;
  } else {
    verdictText = `Too close to call / Tie likely`;
  }
  document.getElementById('simVerdict').innerText = verdictText;

  // Feature Impact Waterfall
  const diffs = FEATURE_NAMES.map((fn, idx) => {
    const diffWeight = W[0][idx] - W[1][idx];
    const impact = diffWeight * xVec[fn];
    return {
      fn,
      label: FEATURE_LABELS[fn],
      symbol: FEATURE_SYMBOLS[fn],
      impact: impact
    };
  });

  const maxAbsImpact = Math.max(0.5, ...diffs.map(d => Math.abs(d.impact)));
  const waterfallDiv = document.getElementById('waterfallContainer');
  waterfallDiv.innerHTML = diffs.map(d => {
    const val = d.impact;
    const pctWidth = Math.min(50, (Math.abs(val) / maxAbsImpact) * 50);
    const isTeam1 = val > 0;
    const barCol = isTeam1 ? c1 : c2;
    return `
      <div class="waterfall-row">
        <div class="waterfall-label">${d.symbol} (${d.fn})</div>
        <div class="waterfall-axis">
          <div class="waterfall-center-line"></div>
          ${isTeam1 ? `
            <div class="waterfall-bar-left" style="width:${pctWidth}%; background:${barCol};"></div>
          ` : `
            <div class="waterfall-bar-right" style="width:${pctWidth}%; background:${barCol};"></div>
          `}
        </div>
        <div class="waterfall-val" style="color:${barCol};">
          ${val >= 0 ? '+' + val.toFixed(3) : val.toFixed(3)}
        </div>
      </div>
    `;
  }).join('');
}

fillSelect('sim_city', CITIES);
fillTeamSelect('sim_team2', TEAMS[0], TEAMS[1]);
fillTeamSelect('sim_team1', document.getElementById('sim_team2').value, TEAMS[0]);
syncToss();
initSandboxSliders();

document.getElementById('sim_team1').addEventListener('change', syncTeams);
document.getElementById('sim_team2').addEventListener('change', syncTeams);
document.getElementById('sim_city').addEventListener('change', recomputeAll);
document.getElementById('sim_toss_winner').addEventListener('change', recomputeAll);
document.getElementById('sim_toss_decision').addEventListener('change', recomputeAll);

recomputeAll();
</script>
</body>
</html>
"""

@app.route('/simulation')
def simulation():
    nav = render_template_string(NAV, active='simulation')
    h2h_data = {f"{k[0]} vs {k[1]}": v for k, v in H2H.items()}
    model_meta = {
        'classes': [int(c) for c in MODEL.classes_],
        'features': list(X.columns),
        'coefficients': MODEL.coef_.tolist(),
        'intercepts': MODEL.intercept_.tolist(),
    }
    return render_template_string(
        SIMULATION_PAGE,
        head=SHARED_HEAD,
        nav=nav,
        teams=TEAMS,
        cities=CITIES,
        colors=TEAM_COLORS,
        home_cities=HOME,
        stats=STATS,
        h2h=h2h_data,
        model_meta=model_meta
    )

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    t1, t2, city = data['team1'], data['team2'], data['city']
    toss_winner, toss_decision = data['toss_winner'], data['toss_decision']
    t1_wr, t1_form = team_stat(t1)
    t2_wr, t2_form = team_stat(t2)
    row = pd.DataFrame([{
        't1_wr': t1_wr, 't2_wr': t2_wr, 't1_form': t1_form, 't2_form': t2_form,
        't1_h2h': head_to_head(t1, t2), 't1_home': is_home(t1, city), 't2_home': is_home(t2, city),
        'toss_t1': int(toss_winner == t1), 'toss_bat': int(toss_decision == 'bat'),
    }])[X.columns]
    probs = MODEL.predict_proba(row)[0]
    classes = list(MODEL.classes_)
    p = {c: probs[i] for i, c in enumerate(classes)}
    t1p, t2p, drp = p.get(0, 0), p.get(1, 0), p.get(2, 0)
    verdict = f"{t1} favoured" if t1p > t2p else f"{t2} favoured"
    if drp > max(t1p, t2p):
        verdict = "Too close to call"
    return jsonify({
        'team1_win': round(t1p * 100, 1),
        'team2_win': round(t2p * 100, 1),
        'draw': round(drp * 100, 1),
        'verdict': verdict,
    })

if __name__ == '__main__':
    app.run(debug=True)