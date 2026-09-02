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