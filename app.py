# app.py
# MIT License (c) 2025 Sahel
# Flask web UI for Agentic Reasoning Math Solver — 
# Run: pip install flask sympy (sympy optional)
# Start: python app.py  →  http://127.0.0.1:5000/Home

from __future__ import annotations
import re
import ast
import math
from typing import Any, Dict, List, Optional
from flask import Flask, request, jsonify, Response # type: ignore

# ---- Optional SymPy (for algebra solving) ----------------------------------------
try:
    import sympy as sp
    SYMPY_OK = True
except Exception:
    SYMPY_OK = False

app = Flask(__name__)

# =========================
# Dev 1 — PLANNER
# =========================
RE_NUMBER = re.compile(r"\d+(?:\.\d+)?", re.I)

def _detect_intent(question: str) -> str:
    q = (question or "").lower().strip()
    if re.fullmatch(r"[0-9\s+\-*/^().,%]+", q or ""):
        return "arithmetic"
    if any(k in q for k in ["solve for", "equation", " unknown", "find x", "find y"]) or "=" in q:
        return "algebra"
    if "%" in q or "percent" in q or "percentage" in q:
        return "percentage"
    if any(k in q for k in ["speed", "velocity", "km/h", "kmph", "m/s", "distance", "time"]):
        return "sdt"
    if any(k in q for k in ["ratio", "proportion", "is to", " a:b", ":"]):
        return "proportion"
    return "freeform"

def _extract_numbers_and_units(text: str) -> Dict[str, Any]:
    nums = [float(n) for n in RE_NUMBER.findall(text)]
    units = re.findall(r"\b(km/h|kmph|km|m|hours?|hrs?|minutes?|mins?|s|sec|seconds?|m/s)\b", text.lower())
    return {"numbers": nums, "units": units}

def plan_problem(question: str) -> Dict[str, Any]:
    q = (question or "").strip()
    intent = _detect_intent(q)
    parsed: Dict[str, Any] = {}
    variables: List[str] = []

    if intent == "arithmetic":
        parsed["expression"] = q

    elif intent == "algebra":
        m = re.search(r"\bfind\s+([a-zA-Z])\b", q, re.I)
        var = m.group(1) if m else "x"
        variables.append(var)
        eq_match = re.search(r"(.+?)=(.+)", q)
        if eq_match:
            parsed["lhs"] = eq_match.group(1).strip()
            parsed["rhs"] = eq_match.group(2).strip()
            parsed["variable"] = var

    elif intent == "percentage":
        parsed["pattern"] = "generic"
        m = re.search(r"(\d+\.?\d*)\s*%.*?(\d+\.?\d*)", q)
        if m:
            parsed["pct"] = float(m.group(1))
            parsed["base"] = float(m.group(2))
        inc = re.search(r"(increase|decrease)\s+(\d+\.?\d*)\s+by\s+(\d+\.?\d*)\s*%", q, re.I)
        if inc:
            parsed["adj_op"] = inc.group(1).lower()
            parsed["adj_base"] = float(inc.group(2))
            parsed["adj_pct"] = float(inc.group(3))

    elif intent == "sdt":
        parsed.update(_extract_numbers_and_units(q))
        want = None
        ql = q.lower()
        if any(k in ql for k in ["find speed", "what is the speed", "velocity", "km/h", "kmph", "m/s"]):
            want = "speed"
        if "find distance" in ql or "what is the distance" in ql:
            want = "distance"
        if "find time" in ql or "what is the time" in ql:
            want = "time"
        parsed["sdt_target"] = want

    elif intent == "proportion":
        parsed["pattern"] = "a:b = c:d (simple proportion if matched)"
        m = re.search(r"(\d+\.?\d*)\s*:\s*(\d+\.?\d*)\s*=\s*(\d+\.?\d*)\s*:\s*(\d+\.?\d*)", q)
        if m:
            parsed["a"], parsed["b"], parsed["c"], parsed["d"] = map(float, m.groups())
        else:
            parsed["text"] = q

    decomposition = [
        "Parse the problem and identify knowns/unknowns",
        "Choose method/tool",
        "Compute intermediate results",
        "Cross-check / verify",
        "Summarize final answer clearly",
    ]
    tools = ["safe-math"]
    if intent in ("algebra", "proportion") and SYMPY_OK:
        tools.append("sympy.solve")

    return {
        "original_question": q,
        "intent": intent,
        "variables": variables,
        "parsed": parsed,
        "decomposition": decomposition,
        "tools": tools,
    }

# =========================
# Dev 2 — SOLVER & VERIFIER
# =========================
class SafeEval:
    """A safer evaluator for math-like expressions using ast validation and a small function whitelist."""
    ALLOWED_NODE_TYPES = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
        ast.Call, ast.Name, ast.Add, ast.Sub, ast.Mult, ast.Div,
        ast.Mod, ast.USub, ast.UAdd, ast.FloorDiv, ast.Pow, ast.Load,
        ast.Tuple,
    )
    ALLOWED_FUNCS = {
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "log": math.log, "log10": math.log10, "exp": math.exp, "pow": math.pow,
        "abs": abs, "round": round,
    }
    ALLOWED_NAMES = {"pi": math.pi, "e": math.e}

    def __init__(self):
        self._names = dict(self.ALLOWED_NAMES)

    def _validate_ast(self, node: ast.AST) -> None:
        """Walk the AST and ensure only allowed node types are present."""
        for n in ast.walk(node):
            if not isinstance(n, self.ALLOWED_NODE_TYPES):
                # allow operator/operand nodes indirectly by their classes being types above
                raise ValueError(f"Disallowed expression node: {type(n).__name__}")

    def eval(self, expr: str) -> float:
        expr = expr.replace("^", "**")
        # parse in eval mode -> returns an ast.Expression root node
        tree = ast.parse(expr, mode="eval")
        self._validate_ast(tree)
        return self._eval(tree) # type: ignore

    def _eval(self, node: ast.AST):
        # handle Expression wrapper
        if isinstance(node, ast.Expression):
            return self._eval(node.body)

        # numeric constants
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numeric constants are allowed.")

        # binary ops
        if isinstance(node, ast.BinOp):
            left = self._eval(node.left)
            right = self._eval(node.right)
            op = node.op
            if isinstance(op, ast.Add): return left + right
            if isinstance(op, ast.Sub): return left - right
            if isinstance(op, ast.Mult): return left * right
            if isinstance(op, ast.Div): return left / right
            if isinstance(op, ast.FloorDiv): return left // right
            if isinstance(op, ast.Mod): return left % right
            if isinstance(op, ast.Pow): return left ** right
            raise ValueError("Unsupported binary operator.")

        # unary ops
        if isinstance(node, ast.UnaryOp):
            val = self._eval(node.operand)
            if isinstance(node.op, ast.UAdd): return +val
            if isinstance(node.op, ast.USub): return -val
            raise ValueError("Unsupported unary operator.")

        # function calls
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls allowed.")
            fname = node.func.id
            if fname not in self.ALLOWED_FUNCS:
                raise ValueError(f"Function '{fname}' not allowed.")
            args = [self._eval(a) for a in node.args]
            func = self.ALLOWED_FUNCS[fname]
            try:
                return func(*args)
            except Exception as e:
                raise ValueError(f"Function call error: {e}")

        # names (constants)
        if isinstance(node, ast.Name):
            if node.id in self._names:
                return self._names[node.id]
            raise ValueError(f"Unknown name '{node.id}'.")

        # tuple/list of values (e.g., for functions that accept tuples)
        if isinstance(node, ast.Tuple):
            return tuple(self._eval(e) for e in node.elts)

        raise ValueError(f"Invalid or unsupported expression node: {type(node).__name__}")

def solve_subtasks(plan: Dict[str, Any]) -> Dict[str, Any]:
    intent = plan.get("intent")
    parsed = plan.get("parsed", {})
    steps: List[str] = []
    result: Optional[Any] = None
    safee = SafeEval()

    try:
        if intent == "arithmetic":
            expr = parsed.get("expression", "")
            steps.append(f"Evaluate expression: {expr}")
            result = safee.eval(expr)
            steps.append(f"Result = {result}")

        elif intent == "algebra":
            if SYMPY_OK and ("lhs" in parsed or "rhs" in parsed):
                varname = (plan.get("variables", ["x"]) or ["x"])[0]
                steps.append("Using sympy to solve equation")
                lhs = parsed.get("lhs") or parsed.get("expression") or "0"
                rhs = parsed.get("rhs") or "0"
                try:
                    x = sp.symbols(varname)
                    eq = sp.Eq(sp.sympify(lhs), sp.sympify(rhs))
                    sol = sp.solve(eq, x)
                    result = sol
                    steps.append(f"Solution: {sol}")
                except Exception as e:
                    steps.append(f"Sympy failed: {e}.")
                    result = None
            else:
                steps.append("Not enough structure to solve algebraically without SymPy.")
                result = None

        elif intent == "percentage":
            if "pct" in parsed and "base" in parsed:
                pct = parsed["pct"]; base = parsed["base"]
                steps.append(f"Compute {pct}% of {base}")
                result = base * pct / 100.0
                steps.append(f"Result = {result}")
            elif "adj_op" in parsed:
                op = parsed["adj_op"]; base = parsed["adj_base"]; pct = parsed["adj_pct"]
                steps.append(f"Apply {op} of {pct}% to {base}")
                if op == "increase":
                    result = base * (1 + pct / 100.0)
                else:
                    result = base * (1 - pct / 100.0)
                steps.append(f"Result = {result}")
            else:
                steps.append("Could not parse percentage pattern.")
                result = None

        elif intent == "sdt":
            nums = parsed.get("numbers", [])
            target = parsed.get("sdt_target")
            steps.append(f"Numbers found: {nums}; target: {target}")
            if target == "speed" and len(nums) >= 2:
                distance, time_val = nums[0], nums[1]
                steps.append(f"Assume distance={distance}, time={time_val}")
                if time_val == 0:
                    steps.append("Time cannot be zero.")
                    result = None
                else:
                    result = distance / time_val
                    steps.append(f"Speed = {result}")
            else:
                steps.append("Not enough numeric info for s/d/t computation.")
                result = None

        elif intent == "proportion":
            if all(k in parsed for k in ("a", "b", "c", "d")):
                a, b, c, d = parsed["a"], parsed["b"], parsed["c"], parsed["d"]
                steps.append(f"Solve/check proportion {a}:{b} = {c}:{d}")
                if b == 0 or d == 0:
                    steps.append("Denominator cannot be zero.")
                    result = None
                else:
                    result = (a / b) == (c / d)
                    steps.append(f"Equality check => {result}")
            else:
                steps.append("Could not parse proportion pattern fully.")
                result = None

        else:
            steps.append("Freeform / unsupported intent — returning plan summary.")
            result = None

        ok = True
    except Exception as e:
        steps.append(f"Error during solving: {e}")
        result = None
        ok = False

    return {"ok": ok, "answer": result, "steps": steps}

def verify_results(plan: Dict[str, Any], solution: Dict[str, Any]) -> Dict[str, Any]:
    checks: List[str] = []
    ans = solution.get("answer")

    if solution.get("ok") is False:
        checks.append("Solver reported failure (ok=False).")
        return {"verified": False, "checks": checks}

    intent = plan.get("intent")
    if intent == "arithmetic" and isinstance(ans, (int, float)):
        checks.append("Arithmetic produced numeric result — looks OK.")
    elif intent == "algebra":
        if SYMPY_OK and ans is not None:
            checks.append("Algebraic solutions returned (sympy used).")
        else:
            checks.append("Algebra: no symbolic verification available.")
    elif intent == "percentage":
        if isinstance(ans, (int, float)):
            checks.append("Percentage computation numeric and reasonable.")
    elif intent == "sdt":
        if isinstance(ans, (int, float)):
            checks.append("s/d/t returned a numeric value; units should be reviewed by user.")
        else:
            checks.append("s/d/t: non-numeric or incomplete result — manual check needed.")
    else:
        checks.append("No specific verification rules for this intent; user check recommended.")

    verified = all("failure" not in c.lower() for c in checks)
    return {"verified": verified, "checks": checks}

# =========================
# API
# =========================
@app.post("/api/solve")
def api_solve():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    show_plan = bool(data.get("showPlan", True))
    if not question:
        return jsonify({"ok": False, "error": "Question is required."}), 400

    plan = plan_problem(question)
    solution = solve_subtasks(plan)
    verification = verify_results(plan, solution)

    return jsonify({
        "ok": True,
        "plan": plan if show_plan else None,
        "solution": solution,
        "verification": verification
    })

# =========================
# UI (single-page, inline)
# =========================
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Agentic Math Solver</title>
<style>
  :root{
    --bg:#0b0f19;
    --card: rgba(255,255,255,0.08);
    --card-border: rgba(255,255,255,0.12);
    --text:#e6e6ea;
    --muted:#a0a6b2;
    --accent:#7c5cff; /* primary violet */
    --accent2:#22d3ee; /* cyan */
    --success:#22c55e;
    --warn:#f59e0b;
    --danger:#ef4444;
    --shadow: 0 10px 30px rgba(0,0,0,0.35);
    --radius:16px;
  }
  *{box-sizing:border-box}
  html,body{height:100%}
  body{
    margin:0; color:var(--text); background: radial-gradient(1200px 700px at 80% -10%, rgba(124,92,255,0.25), transparent 60%),
                                  radial-gradient(900px 600px at -10% 20%, rgba(34,211,238,0.20), transparent 60%),
                                  var(--bg);
    font: 16px/1.45 system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, "Helvetica Neue", Arial, "Noto Sans", "Apple Color Emoji","Segoe UI Emoji";
  }
  .container{max-width:1100px;margin:0 auto;padding:24px}
  .hero{
    text-align:center; padding:40px 10px 24px;
  }
  .badge{
    display:inline-flex; align-items:center; gap:8px;
    border:1px solid var(--card-border);
    background: rgba(255,255,255,0.06);
    color:var(--muted); padding:6px 10px; border-radius:999px; font-size:13px;
    backdrop-filter: blur(8px);
  }
  .title{
    margin:16px 0 10px; font-size:34px; font-weight:800; letter-spacing:.2px;
    background: linear-gradient(90deg, #fff, #c7c7ff, #9be7ff);
    -webkit-background-clip:text; background-clip:text; color:transparent;
  }
  .subtitle{color:var(--muted); max-width:800px; margin:0 auto 24px}
  .card{
    background: var(--card); border:1px solid var(--card-border); border-radius: var(--radius);
    box-shadow: var(--shadow); backdrop-filter: blur(12px);
    padding: 20px;
  }
  .grid{display:grid; gap:16px}
  @media (min-width: 900px){
    .grid-2{grid-template-columns: 1.2fr .8fr}
  }
  textarea{
    width:100%; min-height:140px; resize:vertical;
    border-radius:12px; border:1px solid var(--card-border);
    background: rgba(0,0,0,0.25); color:var(--text); padding:12px 14px; outline:none;
  }
  textarea:focus{border-color: var(--accent)}
  .row{display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap}
  .controls{display:flex; gap:10px; align-items:center; flex-wrap:wrap}
  .checkbox{display:flex; gap:8px; align-items:center; color:var(--muted); font-size:14px}
  button.primary{
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color:#0a0c10; border:none; border-radius:12px; padding:10px 16px; font-weight:700; cursor:pointer;
    box-shadow: 0 8px 20px rgba(124,92,255,0.35), 0 2px 8px rgba(34,211,238,0.25);
  }
  button.secondary{
    background: transparent; color:var(--text); border:1px solid var(--card-border);
    border-radius:12px; padding:9px 14px; cursor:pointer;
  }
  button:disabled{opacity:.6; cursor:not-allowed}
  .section{margin-top:16px}
  .section h3{margin:0 0 8px; font-size:16px; color:#cfd2ff; letter-spacing:.3px}
  .pill{
    display:inline-block; font-size:12px; color:#cfd2ff; border:1px dashed var(--card-border);
    padding:3px 8px; border-radius:999px; margin-left:6px
  }
  pre{
    margin:0; padding:12px; border-radius:12px; background:#0a0e18; border:1px solid var(--card-border);
    overflow:auto; max-height:260px; white-space:pre-wrap; word-break:break-word;
  }
  ul{margin:8px 0 0 18px; padding:0}
  .muted{color:var(--muted)}
  .good{color:var(--success)} .warn{color:var(--warn)} .bad{color:var(--danger)}
  .footer{color:var(--muted); text-align:center; font-size:13px; margin:26px 0 10px}
  .spinner{
    width:18px; height:18px; border:3px solid rgba(255,255,255,.25); border-top-color:#fff; border-radius:50%;
    animation:spin 1s linear infinite; display:inline-block; vertical-align:middle; margin-right:8px
  }
  @keyframes spin{to{transform:rotate(360deg)}}
  .answer{
    font-size:18px; font-weight:800; color:#d1eaff; padding:10px 12px; border-radius:10px;
    background: rgba(124,92,255,0.12); border:1px solid rgba(124,92,255,0.25);
  }
  .row-right{display:flex; gap:8px; align-items:center}
</style>
</head>
<body>
  <div class="container">
    <div class="hero">
      <span class="badge">⚡ Agentic Reasoning • Math Solver</span>
      <h1 class="title">Solve math problems with steps</h1>
      <p class="subtitle">Enter an expression or word problem. The app will plan → solve → verify. Press
        <b>Ctrl/⌘ + Enter</b> to run.</p>
    </div>

    <div class="grid grid-2">
      <div class="card">
        <div class="row">
          <div class="muted">Question</div>
          <div class="row-right">
            <label class="checkbox">
              <input type="checkbox" id="showPlan" checked/>
              <span>Show plan</span>
            </label>
            <button class="secondary" id="btnClear">Clear</button>
            <button class="primary" id="btnSolve">Solve</button>
          </div>
        </div>
        <div style="margin-top:8px">
          <textarea id="question" placeholder="e.g. 2+3*4  •  What is 20% of 150?  •  Find x: 2*x + 3 = 11  •  A car travels 100 km in 2 hours, find speed."></textarea>
        </div>

        <div class="section">
          <h3>Answer <span class="pill" id="intentPill">—</span></h3>
          <div class="answer" id="answerBox">—</div>
          <div style="margin-top:8px">
            <button class="secondary" id="btnCopy">Copy answer</button>
            <span class="muted" id="statusText"></span>
          </div>
        </div>

        <div class="section">
          <h3>Steps</h3>
          <div id="stepsBox" class="muted">—</div>
        </div>
      </div>

      <div class="card">
        <div class="section">
          <h3>Plan / Parsed Output</h3>
          <pre id="planBox">—</pre>
        </div>
        <div class="section">
          <h3>Verification</h3>
          <pre id="verifyBox">—</pre>
        </div>
      </div>
    </div>

    <div class="footer">
      Tip: install <code>sympy</code> for algebraic solving → <code>pip install sympy</code>. • Built by Sahel (UI styled for a Base44-like vibe).
    </div>
  </div>

<script>
(function(){
  const qEl = document.getElementById('question');
  const showPlanEl = document.getElementById('showPlan');
  const btnSolve = document.getElementById('btnSolve');
  const btnClear = document.getElementById('btnClear');
  const btnCopy = document.getElementById('btnCopy');
  const answerBox = document.getElementById('answerBox');
  const stepsBox = document.getElementById('stepsBox');
  const planBox = document.getElementById('planBox');
  const verifyBox = document.getElementById('verifyBox');
  const statusText = document.getElementById('statusText');
  const intentPill = document.getElementById('intentPill');

  let lastAnswer = null;

  async function solve(){
    const question = (qEl.value || '').trim();
    if(!question){
      status('Please enter a question.', 'warn');
      return;
    }
    lock(true);
    status('Solving…', 'spin');
    try{
      const res = await fetch('/api/solve', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({question, showPlan: showPlanEl.checked})
      });
      const data = await res.json();
      if(!res.ok || data.ok===false){
        throw new Error(data.error || 'Failed to solve.');
      }
      const sol = data.solution || {};
      const ans = sol.answer;
      lastAnswer = ans;

      answerBox.textContent = (ans===null || ans===undefined)? '—' : String(ans);
      intentPill.textContent = (data.plan && data.plan.intent) ? data.plan.intent : '—';

      // Steps
      const steps = Array.isArray(sol.steps)? sol.steps : [];
      stepsBox.innerHTML = steps.length? ('<ul>'+ steps.map(s=>`<li>${escapeHtml(s)}</li>`).join('') + '</ul>') : '—';

      // Plan
      if(data.plan){
        planBox.textContent = JSON.stringify(data.plan, null, 2);
      }else{
        planBox.textContent = 'Hidden (toggle "Show plan")';
      }

      // Verification
      verifyBox.textContent = JSON.stringify(data.verification || {}, null, 2);

      status('Done.', 'ok');
    }catch(e){
      status(e.message || 'Error occurred.', 'bad');
      console.error(e);
    }finally{
      lock(false);
    }
  }

  function lock(on){
    btnSolve.disabled = on;
    btnClear.disabled = on;
  }

  function status(msg, kind){
    if(kind==='spin'){
      statusText.innerHTML = '<span class="spinner"></span>'+escapeHtml(msg);
    }else{
      statusText.textContent = msg;
      statusText.className = 'muted ' + (kind==='ok'?'good': kind==='warn'?'warn': kind==='bad'?'bad':'');
    }
  }

  function escapeHtml(s){
    return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }

  btnSolve.addEventListener('click', solve);
  btnClear.addEventListener('click', ()=>{
    qEl.value=''; answerBox.textContent='—'; stepsBox.textContent='—';
    planBox.textContent='—'; verifyBox.textContent='—'; intentPill.textContent='—';
    status('', '');
  });
  btnCopy.addEventListener('click', ()=>{
    if(lastAnswer===null || lastAnswer===undefined){
      status('No answer to copy.', 'warn'); return;
    }
    navigator.clipboard.writeText(String(lastAnswer)).then(()=>{
      status('Answer copied to clipboard.', 'ok');
    }).catch(()=>status('Copy failed.', 'bad'));
  });
  document.addEventListener('keydown', (e)=>{
    if((e.ctrlKey||e.metaKey) && e.key==='Enter'){ solve(); }
  });
})();
</script>
</body>
</html>
"""

@app.get("/")
@app.get("/Home")  # keep /Home for parity with your link
def home():
    return Response(HTML, mimetype="text/html")

if __name__ == "__main__":
    # Use host="0.0.0.0" if you want to access from other devices on LAN
    app.run(debug=True, port=5000)
