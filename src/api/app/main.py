from fastapi import FastAPI
from src.api.routes.health import router as health_router

app = FastAPI(title='Civora Backend')
app.include_router(health_router)
