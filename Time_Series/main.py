from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, SeasonalNaive

app = FastAPI(title="Time Series Forecasting API")

# Load data and train model at startup
DATA_PATH = "../data/daily_sales_french_bakery.csv"
df = pd.read_csv(DATA_PATH, parse_dates=["ds"])
df = df.groupby("unique_id").filter(lambda x: len(x) >= 28)
df = df.drop(["unit_price"], axis=1)

models = [
    AutoARIMA(season_length=7, alias="SARIMA"),
    SeasonalNaive(season_length=7),
]
sf = StatsForecast(models=models, freq="D")
sf.fit(df=df)


class PredictRequest(BaseModel):
    unique_id: str = "BAGUETTE"
    horizon: int = 7


class PredictResponse(BaseModel):
    unique_id: str
    forecasts: list[dict]


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/products")
def get_products():
    return {"products": sorted(df["unique_id"].unique().tolist())}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if req.unique_id not in df["unique_id"].unique():
        raise HTTPException(status_code=404, detail=f"Product '{req.unique_id}' not found")
    if not 1 <= req.horizon <= 30:
        raise HTTPException(status_code=400, detail="Horizon must be between 1 and 30")

    preds = sf.predict(h=req.horizon).reset_index()
    product_preds = preds[preds["unique_id"] == req.unique_id]
    product_preds = product_preds.copy()
    product_preds["ds"] = product_preds["ds"].dt.strftime("%Y-%m-%d")
    product_preds = product_preds.drop(columns=["index"], errors="ignore")

    return PredictResponse(
        unique_id=req.unique_id,
        forecasts=product_preds.to_dict(orient="records"),
    )
