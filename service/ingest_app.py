from __future__ import annotations
import os
from typing import List
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse

from ..src.ingest import read_one_csv, ingest_crawl_dir

app = FastAPI(title="TFT ETL Service (crawl → training CSV)")

@app.post("/ingest/dir")
def ingest_dir(crawl_dir: str = Form(...), out: str = Form("data/sample.csv"), tz: str = Form("Asia/Bangkok")):
    try:
        p = ingest_crawl_dir(crawl_dir, out, tz)
        return {"ok": True, "output": p}
    except Exception as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(e)})

@app.post("/ingest/upload")
async def ingest_upload(files: List[UploadFile] = File(...), tz: str = Form("Asia/Bangkok")):
    frames = []
    for f in files:
        content = await f.read()
        tmp_path = Path("/tmp") / f.filename
        with open(tmp_path, "wb") as w:
            w.write(content)
        frames.append(read_one_csv(str(tmp_path), tz))
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    if not frames:
        return JSONResponse(status_code=400, content={"ok": False, "error": "no files"})
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["site_id","timestamp"], keep="last").sort_values(["site_id","timestamp"]).reset_index(drop=True)
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    tmp = "/tmp/processed.csv"
    out.to_csv(tmp, index=False)
    return FileResponse(tmp, filename="processed.csv", media_type="text/csv")