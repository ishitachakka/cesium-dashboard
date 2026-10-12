"""Tiny FastAPI backend for the soil-health digital twin dashboard.

Serves the REAL project data (SSURGO polygons, FDEP land-use polygons, and
the 72 sampling points with every covariate/prediction column already built
in WiSER_SoilLab) as GeoJSON, for cesium-dashboard's index.html to fetch and
render on the globe. Nothing here is synthetic -- every endpoint reads the
actual files fetch_ssurgo.py / fetch_fdep_lulc.py / build_point_feature_table.py
already produced.

Run:
    pip install fastapi uvicorn geopandas
    uvicorn main:app --reload --port 8000

Then open index.html (served some other way, e.g. `python -m http.server 5500`
from this folder -- not file://, since fetch() from a file:// page is blocked
by the browser) and it'll pull from http://localhost:8000.

DATA_DIR defaults to the WiSER_SoilLab repo as a sibling of
EagleCyberNestLAB/cesium-dashboard's actual location -- override with the
WISER_SOIL_LAB_DIR env var if your layout differs.
"""
from __future__ import annotations

import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from shapely.geometry import Point

DATA_DIR = Path(os.environ.get(
    "WISER_SOIL_LAB_DIR",
    Path(__file__).resolve().parent.parent.parent / "EagleCyberNestLAB" / "WiSER_SoilLab",
))

POINTS_CSV = DATA_DIR / "point_features_master.csv"
SSURGO_GPKG = DATA_DIR / "soil_polygons.gpkg"
FDEP_GPKG = DATA_DIR / "fdep_lulc.gpkg"

app = FastAPI(title="Soil Health Digital Twin API")

# Permissive CORS for local dev -- index.html is served from a different
# port (e.g. a plain `python -m http.server`), so the browser needs this to
# allow the fetch() calls through.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _require(path: Path) -> Path:
    if not path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Missing data file: {path}. Set WISER_SOIL_LAB_DIR if WiSER_SoilLab isn't at this default location.",
        )
    return path


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "data_dir": str(DATA_DIR),
        "points_csv_exists": POINTS_CSV.exists(),
        "ssurgo_exists": SSURGO_GPKG.exists(),
        "fdep_exists": FDEP_GPKG.exists(),
    }


@app.get("/api/points")
def points() -> JSONResponse:
    """The 72 real sampling points, every covariate + prediction column
    build_point_feature_table.py produced, as a GeoJSON FeatureCollection.
    """
    df = pd.read_csv(_require(POINTS_CSV))
    gdf = gpd.GeoDataFrame(
        df, geometry=[Point(lon, lat) for lon, lat in zip(df["lon"], df["lat"])], crs="EPSG:4326"
    )
    # NaN isn't valid JSON -- GeoDataFrame.to_json() already maps it to null,
    # but be explicit so this doesn't silently regress if that behavior changes.
    gdf = gdf.where(pd.notnull(gdf), None)
    return JSONResponse(content=gdf.__geo_interface__)


@app.get("/api/ssurgo")
def ssurgo() -> JSONResponse:
    """Real SSURGO soil-class polygons (fetch_ssurgo.py's output)."""
    gdf = gpd.read_file(_require(SSURGO_GPKG))
    gdf = gdf.set_crs("EPSG:4326") if gdf.crs is None else gdf.to_crs("EPSG:4326")
    return JSONResponse(content=gdf.__geo_interface__)


@app.get("/api/fdep")
def fdep() -> JSONResponse:
    """Real FDEP Statewide Land Use/Land Cover polygons (fetch_fdep_lulc.py's output)."""
    gdf = gpd.read_file(_require(FDEP_GPKG)).to_crs("EPSG:4326")
    return JSONResponse(content=gdf.__geo_interface__)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
