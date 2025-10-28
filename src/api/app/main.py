from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import briefs

app = FastAPI(title="Civora AI Backend")

app.include_router(briefs.router, prefix="/api")

# Allow frontend requests (Expo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}

# You can import and include other routers here later
# from src.api.routes import metrics
# app.include_router(metrics.router, prefix="/metrics")