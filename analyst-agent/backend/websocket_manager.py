"""
WebSocket Manager with logging, event buffering, and robust connection handling.
"""
import asyncio
import logging
from collections import deque
from datetime import datetime
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("websocket_manager")


class WebSocketManager:
    """
    Manages a single WebSocket connection with:
    - Event buffering when client is disconnected
    - Automatic event replay on reconnection
    - Comprehensive logging for debugging
    """

    # Maximum events to buffer while client is disconnected
    MAX_BUFFER_SIZE = 100

    def __init__(self):
        self.active: Optional[WebSocket] = None
        self._buffer: deque = deque(maxlen=self.MAX_BUFFER_SIZE)
        self._connected = False
        self._event_count = 0
        self._last_event_time: Optional[datetime] = None

    @property
    def is_connected(self) -> bool:
        return self._connected and self.active is not None

    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection and replay buffered events."""
        await websocket.accept()
        self.active = websocket
        self._connected = True
        logger.info(f"[WS] Client connected. Buffered events: {len(self._buffer)}")

        # Replay buffered events to the newly connected client
        if self._buffer:
            logger.info(f"[WS] Replaying {len(self._buffer)} buffered events...")
            events_to_replay = list(self._buffer)
            self._buffer.clear()
            for event in events_to_replay:
                await self._send_event(event)
            logger.info("[WS] Buffer replay complete")

    def disconnect(self):
        """Mark the WebSocket as disconnected."""
        was_connected = self._connected
        self.active = None
        self._connected = False
        if was_connected:
            logger.info("[WS] Client disconnected")

    async def broadcast(self, event: dict):
        """
        Send an event to the connected client, or buffer it if disconnected.

        Args:
            event: Dict with 'type', 'content', and 'timestamp' keys
        """
        self._event_count += 1
        self._last_event_time = datetime.now()
        event_type = event.get("type", "unknown")

        if self.is_connected:
            success = await self._send_event(event)
            if success:
                logger.debug(f"[WS] Sent event #{self._event_count}: {event_type}")
            else:
                # Send failed, buffer the event
                self._buffer.append(event)
                logger.warning(f"[WS] Send failed, buffered event #{self._event_count}: {event_type}")
        else:
            # No active connection, buffer the event
            self._buffer.append(event)
            logger.warning(f"[WS] No client connected, buffered event #{self._event_count}: {event_type} (buffer size: {len(self._buffer)})")

    async def _send_event(self, event: dict) -> bool:
        """
        Actually send the event via WebSocket.
        Returns True if successful, False otherwise.
        """
        if not self.active:
            return False

        try:
            await self.active.send_json(event)
            return True
        except WebSocketDisconnect:
            logger.warning("[WS] WebSocketDisconnect during send")
            self.disconnect()
            return False
        except RuntimeError as e:
            logger.warning(f"[WS] RuntimeError during send: {e}")
            self.disconnect()
            return False
        except Exception as e:
            logger.error(f"[WS] Unexpected error during send: {e}")
            self.disconnect()
            return False

    def get_stats(self) -> dict:
        """Return connection statistics for debugging."""
        return {
            "connected": self.is_connected,
            "total_events_sent": self._event_count,
            "buffered_events": len(self._buffer),
            "last_event_time": self._last_event_time.isoformat() if self._last_event_time else None,
        }
