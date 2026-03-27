from fastapi import WebSocket, WebSocketDisconnect
from typing import Optional


class WebSocketManager:
    def __init__(self):
        self.active: Optional[WebSocket] = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active = websocket

    def disconnect(self):
        self.active = None

    async def broadcast(self, event: dict):
        if self.active:
            try:
                await self.active.send_json(event)
            except (WebSocketDisconnect, RuntimeError):
                self.active = None
