from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes.health import router as health_router
from src.utils.settings import settings

app = FastAPI(title='Civora Backend', version='0.1.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allow_origins_list(),
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(health_router)
