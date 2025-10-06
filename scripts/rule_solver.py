# scripts/rule_solver.py
import re
from typing import List, Dict, Any
try:
    import sympy as sp
    SYMPY_OK = True
except Exception:
    SYMPY_OK = False

NUM_RE = re.compile(r"[-+]?\d*\.?\d+")

def extract_numbers(text: str) -> List[float]:
    return [float(x) for x in NUM_RE.findall(text)]

def parse_option_number(opt: str):
    # try to extract a single numeric value from an option text
    nums = NUM_RE.findall(opt)
    if not nums:
        # handle percent form "20%" maybe present
        m = re.search(r"(\d+\.?\d*)\s*%", opt)
        if m:
            return float(m.group(1))
        return None
    # return first number (good enough for short numeric options)
    try:
        return float(nums[0])
    except:
        return None

def match_numeric_to_options(value: float, options: List[str], tol=1e-2) -> Dict[str,Any]:
    # Compare numeric value to parsed numbers in options. Returns best match or None
    parsed = []
    for i,opt in enumerate(options):
        num = parse_option_number(opt)
        parsed.append((i+1, num, opt))
    # filter those with numeric parsed
    parsed_numeric = [p for p in parsed if p[1] is not None]
    if not parsed_numeric:
        return {"matched": False}
    # find closest
    idx, num, opt = min(parsed_numeric, key=lambda p: abs(p[1] - value))
    # relative tolerance check
    if abs(num - value) <= max(1e-6, abs(value)*0.03):  # within 3%
        return {"matched": True, "option": idx, "option_value": num, "computed_value": value,
                "diff": abs(num - value)}
    return {"matched": False, "closest_option": idx, "closest_value": num, "computed_value": value, "diff": abs(num - value)}

def solve_sdt(problem_text: str, options: List[str]) -> Dict[str,Any]:
    text = problem_text.lower()
    nums = extract_numbers(problem_text)
    # Basic speed = distance/time detection
    if ('km' in text or 'kilomet' in text or 'm' in text) and any(k in text for k in ['hour','hr','hrs','h']):
        if len(nums) >= 2:
            distance, time = nums[0], nums[1]
            if time == 0: return {"matched": False}
            speed = distance / time
            return match_numeric_to_options(speed, options)
    # percentage detection: "what is 20% of 150"
    m = re.search(r"(\d+\.?\d*)\s*%.*of\s*(\d+\.?\d*)", problem_text, re.I)
    if m:
        pct = float(m.group(1)); base = float(m.group(2))
        val = base * pct / 100.0
        return match_numeric_to_options(val, options)
    # simple distance = speed * time detection if phrasing includes 'travels' etc and options numeric
    if any(k in text for k in ['travels','goes','covers']) and len(nums)>=2:
        # assume distance, time or speed/time order; try couple guesses
        distance, second = nums[0], nums[1]
        # guess speed=distance/time
        if second!=0:
            val = distance / second
            r = match_numeric_to_options(val, options)
            if r.get("matched"): return r
    return {"matched": False}

def solve_algebra(problem_text: str, options: List[str]) -> Dict[str,Any]:
    if not SYMPY_OK:
        return {"matched": False}
    # attempt to find equation like "2*x + 3 = 11" or "find x: 2x+3=11"
    eq_match = re.search(r"(.+?)=(.+)", problem_text)
    if not eq_match: return {"matched": False}
    lhs, rhs = eq_match.group(1).strip(), eq_match.group(2).strip()
    # find variable
    var_m = re.search(r"find\s+([a-zA-Z])", problem_text, re.I)
    var = var_m.group(1) if var_m else 'x'
    try:
        x = sp.symbols(var)
        sol = sp.solve(sp.Eq(sp.sympify(lhs), sp.sympify(rhs)), x)
        if not sol:
            return {"matched": False}
        # take first solution
        sol_val = float(sol[0])
        return match_numeric_to_options(sol_val, options)
    except Exception as e:
        return {"matched": False}
