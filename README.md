# Soccer match outcome classification (H / D / A)

Ky repo përmban një projekt të plotë **klasifikimi multiclass** për rezultatin e ndeshjes së futbollit (**H** fiton vendasi, **D** barazim, **A** fiton vizitori), me **EDA**, **preprocessing + feature engineering pa leakage**, dhe tre modele: **Logistic Regression (baseline)**, **SVM (RBF)**, dhe **XGBoost**.

This repository is intentionally bilingual (SQ/EN) so you can paste blocks directly into a course report written in Albanian or English.

## What is inside

- `data/raw/`: CSV files downloaded from the public **football-data.co.uk** mirror (not committed by default).
- `src/soccer_ml.py`: download helpers + **pre-match** feature construction (rolling history with `shift(1)` + implied probabilities from Bet365 closing odds).
- `scripts/download_data.py`: downloads **5 big European leagues** × **4 seasons** (`2122`–`2425`).
- `notebooks/soccer_match_hda_classification.ipynb`: end-to-end notebook (EDA + `StratifiedKFold` + metrics + confusion matrices).
- `report/paper.tex`: a **Springer-style** LaTeX paper skeleton (swap in official Springer/LNCS class files if your course requires the exact template).

## Quickstart (Windows / macOS / Linux)

Create a virtual environment (recommended), then install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Download the datasets:

```bash
python scripts/download_data.py
```

Open and run:

- `notebooks/soccer_match_hda_classification.ipynb`

If `notebook` installation fails on Windows due to **long paths**, enable long paths (see pip hint) or run the notebook from **VS Code / Cursor** using the built-in Jupyter UI after installing only:

```bash
python -m pip install pandas numpy scikit-learn xgboost matplotlib seaborn ipykernel
```

## Scientific integrity notes (important for grading)

- **Leakage control**: same-match post outcomes (e.g., `FTHG/FTAG` and other same-match statistics) are **not** used as model inputs. The rolling features are computed from **previous** matches only.
- **Odds features**: closing odds are legitimate **pre-kickoff** information, but you should discuss **market efficiency**, **information leakage from bookmakers**, and **responsible use** (betting harm) in the report.

## LaTeX report

Compile `report/paper.tex` with `pdflatex` (run twice for references). If your instructor requires the official Springer template, download it from Springer’s author instructions and replace the `article` setup while keeping the same section structure.

## AI usage disclosure (template)

Your course asks for explicit AI disclosure. Fill in the bracketed parts in `report/paper.tex` under **Appendix: Generative AI and authorship**.

Suggested honest statement pattern:

- AI assisted with boilerplate code structure and LaTeX formatting.
- Humans verified methodology, checked leakage, ran results, and wrote interpretation/critical analysis.

## Citation (dataset source)

The CSV schema and match records are widely mirrored from **football-data.co.uk**. Cite the dataset source you actually used (direct downloads + column notes):

- Dataset downloads: `https://www.football-data.co.uk/`
- Column glossary: `https://www.football-data.co.uk/notes.txt`

## License

The code in this repository is provided for coursework. The underlying football CSVs are retrieved from a public website; respect their terms and cite appropriately.
