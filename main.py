# streamlit_app.py
# MIT License (c) 2025 Sahel
# Streamlit UI for Agentic Reasoning Math Solver (adapted from Flask version)
# Run: pip install streamlit sympy
# Start: streamlit run streamlit_app.py

from __future__ import annotations

import ast
import json
import math
import re
import time
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# try to import sympy; it's optional and useful for algebra/derivatives
try:
    import sympy as sp
    SYMPY_OK = True
except Exception:  # pragma: no cover - optional
    SYMPY_OK = False

# --------------------------- CONFIG: SUBJECT KEYWORDS ---------------------------
SUBJECT_KEYWORDS = {
    "math_basic": [
        "place value",
        "addition",
        "subtraction",
        "multiplication",
        "division",
        "fractions",
        "decimals",
        "measurement",
        "geometry basics",
        "shapes",
        "patterns",
        "symmetry",
        "pictograph",
        "bar graph",
    ],
    "math_middle": [
        "integers",
        "rational",
        "factors",
        "multiples",
        "lcm",
        "hcf",
        "algebra",
        "simple equation",
        "triangles",
        "angles",
        "mensuration",
        "probability",
        "statistics",
        "ratio",
        "proportion",
        "percentage",
    ],
    "math_high": [
        "quadratic",
        "polynomial",
        "sequence",
        "coordinate geometry",
        "trigonometry",
        "functions",
        "graphs",
        "3d geometry",
    ],
    "math_advanced": [
        "calculus",
        "limits",
        "derivative",
        "integral",
        "vectors",
        "linear algebra",
        "matrices",
        "determinant",
        "differential equation",
        "complex number",
        "probability distribution",
        "mathematical reasoning",
    ],
    "physics": [
        "kinematics",
        "dynamics",
        "work",
        "energy",
        "power",
        "electricity",
        "magnetism",
        "optics",
        "thermodynamics",
        "waves",
    ],
    "chemistry": [
        "stoichiometry",
        "mole",
        "molar mass",
        "chemical reaction",
        "acid",
        "base",
        "periodic",
        "organic",
        "inorganic",
        "thermochemistry",
    ],
}

RE_NUMBER = re.compile(r"\d+(?:\.\d+)?")

# --------------------------- Dev 1 — PLANNER (extended) ---------------------------

def _detect_intent(question: str) -> str:
    q = (question or "").lower().strip()

    # direct subject detection first
    for intent, keywords in SUBJECT_KEYWORDS.items():
        for k in keywords:
            if k in q:
                return intent

    # fall back to previous heuristics
    if re.fullmatch(r"[0-9\s+\-*/^().,%]+", q):
        return "arithmetic"
    if any(k in q for k in ["solve for", "equation", "=", "unknown", "find x", "find y"]):
        return "algebra"
    if "%" in q or "percent" in q or "percentage" in q:
        return "percentage"
    if any(k in q for k in ["speed", "velocity", "km/h", "kmph", "m/s", "distance", "time"]):
        return "sdt"
    if any(k in q for k in ["ratio", "proportion", "is to", "a:b", ":"]):
        return "proportion"
    return "freeform"


def _extract_numbers_and_units(text: str) -> Dict[str, Any]:
    nums = RE_NUMBER.findall(text)
    units = re.findall(r"\b(km/h|kmph|km|m|hours?|hrs?|minutes?|mins?|s|sec|seconds?|m/s|g|kg|mol|L|ml)\b", text.lower())
    return {"numbers": [float(n) for n in nums], "units": units}


def plan_problem(question: str) -> Dict[str, Any]:
    """Produce a plan that includes detected subject area and parsed structure.
    The planner is intentionally conservative; for complex tasks the solver will
    return a scaffold and ask for clarification or more data.
    """
    q = (question or "").strip()
    intent = _detect_intent(q)
    parsed: Dict[str, Any] = {}
    variables: List[str] = []

    # reuse previous parsing logic for math-related intent
    if intent in ("arithmetic", "algebra", "percentage", "sdt", "proportion"):
        # reuse earlier parsing
        if intent == "arithmetic":
            parsed["expression"] = q
        elif intent == "algebra":
            m = re.search(r"\bfind\s+([a-zA-Z])\b", q.lower())
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
            inc = re.search(r"(increase|decrease)\s+(\d+\.?\d*)\s+by\s+(\d+\.?\d*)\s*%", q)
            if inc:
                parsed["adj_op"] = inc.group(1)
                parsed["adj_base"] = float(inc.group(2))
                parsed["adj_pct"] = float(inc.group(3))
        elif intent == "sdt":
            parsed.update(_extract_numbers_and_units(q))
            want = None
            if any(k in q for k in ["speed", "velocity", "km/h", "kmph", "m/s"]):
                want = "speed"
            if "distance" in q:
                want = "distance" if ("find distance" in q or "what is the distance" in q) else want
            if "time" in q:
                want = "time" if ("find time" in q or "what is the time" in q) else want
            parsed["sdt_target"] = want
        elif intent == "proportion":
            parsed["pattern"] = "a:b = c:x  (simple proportion if matched)"
            m = re.search(r"(\d+\.?\d*)\s*:\s*(\d+\.?\d*)\s*=\s*(\d+\.?\d*)\s*:\s*(\d+\.?\d*)", q)
            if m:
                parsed["a"], parsed["b"], parsed["c"], parsed["d"] = map(float, m.groups())

    else:
        # For subject-level intents just store the raw text and detected subject
        parsed["text"] = q

    decomposition = [
        "Parse the problem and identify knowns/unknowns",
        "Choose method/tool",
        "Compute intermediate results or provide conceptual explanation",
        "Cross-check / verify",
        "Summarize final answer clearly",
    ]

    tools = ["safe-math"]
    if intent in ("algebra", "math_high", "math_advanced") and SYMPY_OK:
        tools.append("sympy.solve")

    return {
        "original_question": q,
        "intent": intent,
        "variables": variables,
        "parsed": parsed,
        "decomposition": decomposition,
        "tools": tools,
    }


# --------------------------- Dev 2 — SOLVER & VERIFIER (extended) ---------------------------

class SafeEval(ast.NodeVisitor):
    """A safer evaluator for math-like expressions.
    Supports a subset of Python expressions (numbers, + - * / ** %, function calls
    to a small whitelist and a couple of names like pi/e).
    """

    ALLOWED_NODES = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Num,
        ast.Constant,
        ast.Call,
        ast.Name,
        ast.Pow,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.USub,
        ast.UAdd,
        ast.FloorDiv,
        ast.Load,
        ast.Tuple,
    )

    ALLOWED_FUNCS = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "pow": math.pow,
        "abs": abs,
        "round": round,
    }

    ALLOWED_NAMES = {"pi": math.pi, "e": math.e}

    def __init__(self):
        self._names = dict(self.ALLOWED_NAMES)

    def visit(self, node):
        if not isinstance(node, self.ALLOWED_NODES):
            raise ValueError(f"Disallowed expression: {type(node).__name__}")
        return super().visit(node)

    def eval(self, expr: str) -> float:
        expr = expr.replace("^", "**")  # allow ^ for power
        tree = ast.parse(expr, mode="eval")
        return self._eval(tree.body) # type: ignore

    def _eval(self, node):
        if isinstance(node, ast.Num):
            return node.n
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numeric constants are allowed.")
        if isinstance(node, ast.BinOp):
            left = self._eval(node.left)
            right = self._eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Mod):
                return left % right
            if isinstance(node.op, ast.Pow):
                return left ** right
            raise ValueError("Unsupported operator.")
        if isinstance(node, ast.UnaryOp):
            operand = self._eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand # type: ignore
            if isinstance(node.op, ast.USub):
                return -operand # type: ignore
            raise ValueError("Unsupported unary operator.")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls allowed.")
            fname = node.func.id
            if fname not in self.ALLOWED_FUNCS:
                raise ValueError(f"Function '{fname}' not allowed.")
            args = [self._eval(a) for a in node.args]
            return self.ALLOWED_FUNCS[fname](*args)
        if isinstance(node, ast.Name):
            if node.id in self._names:
                return self._names[node.id]
            raise ValueError(f"Unknown name '{node.id}'.")
        if isinstance(node, ast.Tuple):
            return tuple(self._eval(elt) for elt in node.elts)
        raise ValueError("Invalid expression.")


# --------------------------- Topic handlers (scaffolding) ---------------------------

# Math basic handlers

def handle_basic_arithmetic(text: str) -> Dict[str, Any]:
    plan = plan_problem(text)
    expr = plan.get("parsed", {}).get("expression")
    safe = SafeEval()
    try:
        val = safe.eval(expr)
        return {"ok": True, "answer": val, "explanation": [f"Evaluated {expr} = {val}"]}
    except Exception as e:
        return {"ok": False, "answer": None, "explanation": [f"Could not evaluate expression: {e}"]}


def handle_fraction_decimal(text: str) -> Dict[str, Any]:
    # simple examples: convert fraction string like '3/4' or decimal
    m = re.search(r"(\d+)\s*/\s*(\d+)", text)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        val = a / b
        return {"ok": True, "answer": val, "explanation": [f"{a}/{b} = {val}"]}
    m = re.search(r"(\d+\.?\d*)", text)
    if m:
        return {"ok": True, "answer": float(m.group(1)), "explanation": ["Detected decimal number."]}
    return {"ok": False, "answer": None, "explanation": ["Could not parse fraction/decimal."]}


# Geometry basics (area/perimeter examples)

def handle_geometry_basic(text: str) -> Dict[str, Any]:
    # detect common shapes: square, rectangle, circle
    if "area" in text and "circle" in text:
        m = re.search(r"radius\s*(?:=|is)?\s*(\d+\.?\d*)", text)
        if m:
            r = float(m.group(1))
            area = math.pi * r * r
            return {"ok": True, "answer": area, "explanation": [f"Area(circle) = pi*r^2 = {area}"]}
    if "area" in text and "rectangle" in text:
        m = re.search(r"(length|l)\s*(?:=|is)?\s*(\d+\.?\d*)", text)
        n = re.search(r"(width|w)\s*(?:=|is)?\s*(\d+\.?\d*)", text)
        if m and n:
            l = float(m.group(2)); w = float(n.group(2))
            return {"ok": True, "answer": l * w, "explanation": [f"Area = l*w = {l*w}"]}
    return {"ok": False, "answer": None, "explanation": ["Geometry handler could not parse inputs."]}


# Middle school handlers (simple algebra, LCM/HCF)

def handle_middle_algebra(text: str) -> Dict[str, Any]:
    # try to detect simple linear equation ax + b = c
    m = re.search(r"([\d\.-]+)\s*\*?\s*([a-z])\s*([+-])\s*([\d\.-]+)\s*=\s*([\d\.-]+)", text)
    if m and SYMPY_OK:
        a = float(m.group(1)); var = m.group(2); op = m.group(3); b = float(m.group(4)); c = float(m.group(5))
        x = sp.symbols(var)
        expr = a * x + (b if op == "+" else -b) - c
        sol = sp.solve(sp.Eq(a * x + (b if op == "+" else -b), c), x)
        return {"ok": True, "answer": sol, "explanation": [f"Solved {a}{var}{op}{b}={c} -> {sol}"]}
    return {"ok": False, "answer": None, "explanation": ["Could not parse simple algebra or sympy not available."]}


def lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b) if a and b else 0


def handle_factors_multiples(text: str) -> Dict[str, Any]:
    nums = RE_NUMBER.findall(text)
    if len(nums) >= 2:
        a, b = int(nums[0]), int(nums[1])
        return {"ok": True, "answer": {"gcd": math.gcd(a, b), "lcm": lcm(a, b)}, "explanation": [f"gcd({a},{b})={math.gcd(a,b)}; lcm={lcm(a,b)}"]}
    return {"ok": False, "answer": None, "explanation": ["Need two integers to compute gcd/lcm."]}


# High school handlers (quadratic, trigonometry)

def handle_quadratic(text: str) -> Dict[str, Any]:
    # parse ax^2 + bx + c = 0 roughly
    m = re.search(r"([\d\.-]+)x\^2\s*([+-]\s*[\d\.-]+)x\s*([+-]\s*[\d\.-]+)\s*=\s*0", text.replace(" ", ""))
    if m and SYMPY_OK:
        a = float(m.group(1)); b = float(m.group(2).replace(" ", "")); c = float(m.group(3).replace(" ", ""))
        x = sp.symbols('x')
        sol = sp.solve(sp.Eq(a * x**2 + b * x + c, 0), x)
        return {"ok": True, "answer": sol, "explanation": [f"Solutions: {sol}"]}
    return {"ok": False, "answer": None, "explanation": ["Quadratic parse failed or sympy not available."]}


# Advanced handlers (calculus: derivative using sympy if available)

def handle_derivative(text: str) -> Dict[str, Any]:
    if not SYMPY_OK:
        return {"ok": False, "answer": None, "explanation": ["Sympy required for symbolic derivatives."]}
    # expect formats like "derivative of x**2" or "d/dx x**2"
    m = re.search(r"derivative of (.+)", text)
    if not m:
        m = re.search(r"d/dx (.+)", text)
    if m:
        expr = m.group(1).strip()
        try:
            x = sp.symbols('x')
            dexpr = sp.diff(sp.sympify(expr), x)
            return {"ok": True, "answer": str(dexpr), "explanation": [f"d/dx {expr} = {dexpr}"]}
        except Exception as e:
            return {"ok": False, "answer": None, "explanation": [f"Sympy error: {e}"]}
    return {"ok": False, "answer": None, "explanation": ["Could not parse derivative request."]}


# Physics handlers (basic kinematics and ideal gas law)

def handle_kinematics(text: str) -> Dict[str, Any]:
    # support: "distance time" -> speed
    nums = RE_NUMBER.findall(text)
    if len(nums) >= 2 and "speed" in text or "velocity" in text:
        d, t = float(nums[0]), float(nums[1])
        if t == 0:
            return {"ok": False, "answer": None, "explanation": ["Time cannot be zero."]}
        return {"ok": True, "answer": d / t, "explanation": [f"Speed = distance/time = {d}/{t} = {d/t}"]}
    # basic kinematics: s = ut + 1/2 at^2
    m = re.search(r"s=ut\+1/2at\^2|s=ut\+0.5at\^2", text.replace(" ", ""))
    return {"ok": False, "answer": None, "explanation": ["Kinematics parser too simple; please provide numeric values with labels."]}


def handle_ideal_gas(text: str) -> Dict[str, Any]:
    # expect inputs like P V n T; rough parse
    vals = RE_NUMBER.findall(text)
    if len(vals) >= 3 and "ideal gas" in text:
        # we won't assume units; just return PV=nRT computation scaffold
        return {"ok": True, "answer": None, "explanation": ["Use PV = nRT (ensure consistent units). Provide P,V,n or T to compute the missing value."]}
    return {"ok": False, "answer": None, "explanation": ["Could not parse ideal gas inputs."]}


# Chemistry handlers (mole and molar mass scaffolding)

def handle_mole_calculation(text: str) -> Dict[str, Any]:
    # parse like "how many moles in 18 g of H2O" but we lack periodic table.
    m = re.search(r"(\d+\.?\d*)\s*g of (\w+)", text)
    if m:
        grams = float(m.group(1)); formula = m.group(2)
        return {"ok": True, "answer": None, "explanation": [f"To compute moles: moles = mass (g) / molar_mass(g/mol). Need molar mass of {formula}."]}
    if "mole" in text and "grams" in text:
        return {"ok": True, "answer": None, "explanation": ["Provide mass and formula, e.g. '18 g of H2O'."]}
    return {"ok": False, "answer": None, "explanation": ["Could not parse mole calculation request."]}


# --------------------------- Master solver dispatcher ---------------------------

def solve_subtasks(plan: Dict[str, Any]) -> Dict[str, Any]:
    intent = plan.get("intent")
    parsed = plan.get("parsed", {})
    text = plan.get("original_question", "")

    # arithmetic/algebra/percentage/proportion/sdt handled by previous implementation
    if intent == "arithmetic":
        return handle_basic_arithmetic(text)
    if intent == "percentage":
        # reuse previous percentage logic quickly
        if "pct" in parsed and "base" in parsed:
            pct = parsed["pct"]; base = parsed["base"]
            return {"ok": True, "answer": base * pct / 100.0, "explanation": [f"{pct}% of {base} = {base*pct/100.0}"]}
        return {"ok": False, "answer": None, "explanation": ["Could not parse percentage."]}
    if intent == "sdt":
        nums = parsed.get("numbers", [])
        target = parsed.get("sdt_target")
        if target == "speed" and len(nums) >= 2:
            d, t = nums[0], nums[1]
            return {"ok": True, "answer": d / t, "explanation": [f"Speed = {d}/{t} = {d/t}"]}
        return {"ok": False, "answer": None, "explanation": ["Not enough s/d/t data."]}
    if intent == "proportion":
        if all(k in parsed for k in ("a", "b", "c", "d")):
            a, b, c, d = parsed["a"], parsed["b"], parsed["c"], parsed["d"]
            equal = (a / b) == (c / d)
            return {"ok": True, "answer": equal, "explanation": [f"Equality {a}/{b} == {c}/{d} -> {equal}"]}
        return {"ok": False, "answer": None, "explanation": ["Proportion parsing failed."]}

    # subject-level intents
    if intent == "math_basic":
        # try a few handlers
        if any(k in text for k in ["+", "-", "*", "/"]):
            return handle_basic_arithmetic(text)
        if "/" in text:
            return handle_fraction_decimal(text)
        if "area" in text or "perimeter" in text:
            return handle_geometry_basic(text)
        return {"ok": True, "answer": None, "explanation": ["Math Basic recognized. I can explain concepts, give examples, or solve numeric problems. Provide a specific question."]}

    if intent == "math_middle":
        if any(k in text for k in ["gcd", "hcf", "lcm", "lcm of"]):
            return handle_factors_multiples(text)
        if any(k in text for k in ["equation", "solve", "x"]):
            return handle_middle_algebra(text)
        return {"ok": True, "answer": None, "explanation": ["Middle school math recognized. Ask about integers, fractions, basic algebra, geometry or stats."]}

    if intent == "math_high":
        if "quadratic" in text or "x^2" in text:
            return handle_quadratic(text)
        return {"ok": True, "answer": None, "explanation": ["High school math recognized. I can help with algebra, trig, coordinate geometry and stats. Give a specific problem."]}

    if intent == "math_advanced":
        if "derivative" in text or "d/dx" in text:
            return handle_derivative(text)
        return {"ok": True, "answer": None, "explanation": ["Advanced math recognized. For symbolic calculus tasks install sympy. I can provide scaffolding and numeric checks."]}

    if intent == "physics":
        if "ideal gas" in text:
            return handle_ideal_gas(text)
        if any(k in text for k in ["speed", "velocity", "distance", "time"]):
            return handle_kinematics(text)
        return {"ok": True, "answer": None, "explanation": ["Physics recognized. Ask a focused problem (e.g., 'compute final velocity with u=.. a=.. t=..')."]}

    if intent == "chemistry":
        if "mole" in text or "g of" in text:
            return handle_mole_calculation(text)
        return {"ok": True, "answer": None, "explanation": ["Chemistry recognized. I can explain stoichiometry, mole concepts, acids/bases, and basic reaction balancing with inputs."]}

    # fallback
    return {"ok": False, "answer": None, "explanation": ["Intent not handled yet. Please ask a specific question or provide numbers/variables."]}


def verify_results(plan: Dict[str, Any], solution: Dict[str, Any]) -> Dict[str, Any]:
    checks: List[str] = []
    if not solution.get("ok"):
        checks.append("Solver reported failure or incomplete result.")
        return {"verified": False, "checks": checks}
    # simple heuristics
    if solution.get("answer") is None:
        checks.append("Solver returned no numeric answer — likely an explanatory response.")
    else:
        checks.append("Solver returned a numeric or symbolic answer.")
    return {"verified": True, "checks": checks}


# --------------------------- Dev 3 — STREAMLIT UI (extended) ---------------------------

st.set_page_config(page_title="Agentic Reasoning Chat — STEM Tutor", layout="centered")
st.title("Agentic Reasoning Chat — STEM Tutor")
st.caption("Ask math, physics or chemistry questions from basic to advanced. This is a scaffold — give specific inputs for numeric solves.")

with st.form(key="qform"):
    question = st.text_area("Enter your question (e.g. 'What is 20% of 150', or 'derivative of x**2')", height=140)
    show_plan = st.checkbox("Show plan (parsing & tools)", value=True)
    run_btn = st.form_submit_button("Run")

if run_btn:
    if not question.strip():
        st.warning("Please enter a question or expression to solve.")
    else:
        with st.spinner("Planning..."):
            plan = plan_problem(question)
            time.sleep(0.12)

        if show_plan:
            st.subheader("Plan / Parsed Output")
            st.json(plan)

        with st.spinner("Solving..."):
            solution = solve_subtasks(plan)
            time.sleep(0.12)

        st.subheader("Solution / Explanation")
        if solution.get("ok"):
            if solution.get("answer") is not None:
                st.write("**Answer:**")
                st.write(solution.get("answer"))
            st.write("**Explanation / Steps:**")
            for s in solution.get("explanation", solution.get("steps", [])):
                st.write("- ", s)
        else:
            st.error("Could not solve. See explanation for why.")
            for s in solution.get("explanation", []):
                st.write("- ", s)

        with st.spinner("Verifying..."):
            verification = verify_results(plan, solution)
            time.sleep(0.08)

        st.subheader("Verification")
        st.json(verification)

st.markdown("---")
st.markdown("**Notes:** This expanded scaffold recognizes many STEM topics and provides simple numeric solvers or explanatory scaffolding. For advanced symbolic math (derivatives, symbolic algebra) install sympy (`pip install sympy`). If you want deeper topic-specific solvers, tell me which exact behaviors (e.g., \"balance chemical equations\", \"solve integrals\", \"compute eigenvalues\") you want and I will implement them.")


# --------------------------- Basic unit tests (do not run in Streamlit) ---------------------------

def run_unit_tests() -> None:
    print("Running unit tests...")

    # Arithmetic
    plan = plan_problem("2+3*4")
    assert plan["intent"] in ("arithmetic", "math_basic")
    sol = solve_subtasks(plan)
    assert sol["ok"] and sol["answer"] == 14

    # Percentage
    plan = plan_problem("What is 20% of 150")
    assert plan["intent"] == "percentage"
    sol = solve_subtasks(plan)
    assert sol["ok"] and abs(sol["answer"] - 30.0) < 1e-9

    # SDT
    plan = plan_problem("A car travels 100 2")
    plan["intent"] = "sdt"
    plan["parsed"]["numbers"] = [100, 2]
    plan["parsed"]["sdt_target"] = "speed"
    sol = solve_subtasks(plan)
    assert sol["ok"] and sol["answer"] == 50

    # Fraction
    sol = handle_fraction_decimal("3/4")
    assert sol["ok"] and abs(sol["answer"] - 0.75) < 1e-9

    # LCM/GCD
    sol = handle_factors_multiples("12 18")
    assert sol["ok"] and sol["answer"]["gcd"] == 6

    print("All tests passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tests", action="store_true", help="Run unit tests instead of Streamlit UI")
    args = parser.parse_args()
    if args.run_tests:
        run_unit_tests()
    else:
        print("This file is primarily a Streamlit app. Run with `streamlit run main_expanded.py`. To run tests: `python main_expanded.py --run-tests`.")
