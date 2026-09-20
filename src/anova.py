"""입력 변수 분산분석"""

from pathlib import Path

import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from src.download_data import DATA_PATH

TARGET = "Life expectancy"
CANDIDATES = [
    "Adult Mortality", "infant deaths", "Alcohol", "percentage expenditure",
    "Hepatitis B", "Measles", "BMI", "under-five deaths", "Polio",
    "Total expenditure", "Diphtheria", "HIV/AIDS", "GDP", "Population",
    "thinness  1-19 years", "thinness 5-9 years",
    "Income composition of resources", "Schooling",
]


def run_anova(raw):
    """Type I ANOVA"""
    data = raw.rename(columns=str.strip)[[TARGET, *CANDIDATES]].dropna()
    # - 원본 변수 순서에 따른 순차 검정
    # - 전체 자료의 분석 확인용, ANN 입력 목록 고정
    formula = f"Q({TARGET!r}) ~ " + " + ".join(f"Q({name!r})" for name in CANDIDATES)
    model = ols(formula, data=data).fit()
    table = anova_lm(model, typ=1)
    table.index = [*CANDIDATES, "Residuals"]
    table.index.name = "variable"
    table["mean_sq"] = table["sum_sq"] / table["df"]
    table = table.rename(columns={"PR(>F)": "p_value"})
    table = table[["df", "sum_sq", "mean_sq", "F", "p_value"]]
    return table, len(data)


if __name__ == "__main__":
    table, n_rows = run_anova(pd.read_csv(DATA_PATH))
    output = Path(__file__).resolve().parents[1] / "results" / "anova.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output)
    print(f"ANOVA observations: {n_rows}")
    print(table.to_string())
