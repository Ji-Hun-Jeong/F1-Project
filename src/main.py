from fastapi import FastAPI, WebSocket
from server.response_data import router  # routes.py에서 router 가져오기

app = FastAPI()

# 라우터 등록
app.include_router(router)

# uvicorn으로 실행: uvicorn main:app --reload