# Agentic Reasoning System

Streamlit STEM problem-solving application with topic detection, step planning, numeric rules, and optional SymPy-based symbolic calculations.

## Run the application

```powershell
python -m pip install -r requirements.txt
streamlit run main.py
```

Open the local URL shown by Streamlit, usually `http://localhost:8501`.

## Complete fresh-start commands

Run these commands from PowerShell after opening the repository:

```powershell
cd "c:\Users\ASUS ROG\OneDrive\ドキュメント\Agentic-Reasoning-System"
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python main.py --run-tests
streamlit run main.py
```

If PowerShell blocks activation, run this once in the same PowerShell window and repeat the activation command:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Python 3.13 is recommended for the pinned legacy versions in `requirements.txt`. With Python 3.14, install compatible current wheels instead:

```powershell
python -m pip install "streamlit>=1.33" "sympy>=1.12" "Flask==3.1.2" "asgiref==3.9.2" "uvicorn==0.37.0" "pandas>=2.2.3" "numpy>=2.1.0" "scikit-learn>=1.4.0" joblib tqdm "peft>=0.4"
```

### Optional data and model pipeline

After dependencies are installed, the model workflow can be run in this order:

```powershell
python scripts\eda.py
python src\prepare_dataset.py
python scripts\train_baseline.py
python src\train_model.py
python src\predict.py
python src\evaluate.py
```

The exact model scripts may require their expected input files and command-line arguments. The Streamlit solver is independent and can be launched with `streamlit run main.py`.

## Problem coverage

The application is a rule-based solver. It does not automatically solve every question in every listed subject. The tables below distinguish working solvers from topics that are recognized and currently return guidance or scaffolding.

### Implemented solvers

| Topic | Problem types currently solved | Example input |
| --- | --- | --- |
| Arithmetic | `+`, `-`, `*`, `/`, powers, modulo, floor division, parentheses, and safe math functions such as `sqrt`, `sin`, `cos`, `log`, `exp`, `abs`, and `round` | `2 + 3 * 4` |
| Fractions and decimals | Basic fraction-to-decimal conversion and decimal detection | `3/4` |
| Percentages | Percentage of a number | `What is 20% of 150` |
| Linear algebra | Simple linear equations in the form `ax + b = c` | `2x + 3 = 7` |
| Factors and multiples | GCD/HCF and LCM of two integers | `Find the gcd and lcm of 12 and 18` |
| Geometry | Area of a circle with a radius; area of a rectangle with length and width | `Find the area of a circle with radius 5` |
| Quadratics | Roots of a polynomial in the approximate form `ax^2 + bx + c = 0` | `2x^2 + 3x - 2 = 0` |
| Derivatives | Symbolic derivative with respect to `x` | `derivative of x**2` |
| Integrals | Indefinite symbolic integrals, including compact notation such as `ex` | `integrate x**2` or `∫x2exdx` |
| Matrix inverse | Inverse of a 2x2 matrix written using LaTeX `pmatrix` | `A=\\begin{pmatrix}2&1\\\\3&2\\end{pmatrix}, find A^{-1}` |
| Maximum and minimum | Absolute extrema of a one-variable function on a closed interval; checks endpoints and critical points | `Find the maximum and minimum values of f(x)=x^3-3x^2+2 on 0<=x<=3` |
| Speed, distance, and time | Speed from distance and time | `A car travels 100 km in 2 hours. Find its speed` |
| Convex lenses | Image distance, magnification, and image nature using the thin-lens formula | `An object is 20 cm in front of a convex lens of focal length 10 cm` |

### Recognized topics with guidance or scaffolding

These keywords are detected by the planner, but they do not all have complete numerical solvers yet.

| Subject | Recognized topics |
| --- | --- |
| Basic mathematics | Place value, addition, subtraction, multiplication, division, fractions, decimals, measurement, basic geometry, shapes, patterns, symmetry, pictographs, and bar graphs |
| Middle-school mathematics | Integers, rational numbers, factors, multiples, LCM, HCF, algebra, simple equations, triangles, angles, mensuration, probability, statistics, ratios, proportions, and percentages |
| High-school mathematics | Quadratics, polynomials, sequences, coordinate geometry, trigonometry, functions, graphs, and 3D geometry |
| Advanced mathematics | Calculus, limits, derivatives, integrals, vectors, linear algebra, matrices, determinants, differential equations, complex numbers, probability distributions, and mathematical reasoning |
| Physics | Kinematics, dynamics, work, energy, power, electricity, magnetism, optics, lenses, thermodynamics, and waves |
| Chemistry | Stoichiometry, moles, molar mass, chemical reactions, acids, bases, the periodic table, organic chemistry, inorganic chemistry, and thermochemistry |

For recognized-but-unimplemented topics, the app returns an explanation or asks for a more specific supported numeric form instead of pretending to have solved the problem.

## Input notes

- Use explicit operators when possible: write `3*x` instead of `3x`.
- For symbolic expressions, use Python/SymPy notation such as `x**2` for $x^2$.
- Matrix inversion currently supports 2x2 LaTeX `pmatrix` input.
- Lens calculations use the Cartesian sign convention: an object in front of a convex lens has negative object distance.
- Advanced symbolic features require SymPy.

## Architecture

1. The planner detects a subject and extracts numbers, variables, expressions, and intervals.
2. The dispatcher selects a topic-specific handler.
3. Safe arithmetic uses a restricted AST evaluator.
4. Symbolic calculations use SymPy when available.
5. The verifier reports whether a handler returned a result.

## Project structure

```text
Agentic-Reasoning-System/
├─ Commands.ps1            # PowerShell setup and execution checklist
├─ main.py                 # Streamlit UI, planner, solvers, and verifier
├─ requirements.txt        # Python dependencies
├─ data/                   # Training, test, and generated CSV/JSONL data
├─ models/                 # Saved baseline model artifacts
├─ scripts/                # Training, inference, EDA, and evaluation scripts
└─ src/                    # Dataset preparation, training, prediction, and evaluation
```

## Built-in tests

```powershell
python main.py --run-tests
```

The tests cover arithmetic, percentages, speed calculation, fractions, and GCD/LCM. The Streamlit UI should be run with `streamlit run main.py`.