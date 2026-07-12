"""WebSocket API router.

This module defines the WebSocket endpoint for real-time chat communication.
"""
import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..dependencies import get_session_manager

router = APIRouter()


@router.websocket("/ws/chat")
async def chat(websocket: WebSocket, session_id: str, token: str):
    """WebSocket endpoint for real-time chat.

    Accepts WebSocket connections and processes user messages through
    the SessionManager, streaming events back to the client.

    Args:
        websocket: WebSocket connection instance.
        session_id: Session ID to associate with this connection.
        token: Authentication token (for future use).
    """
    await websocket.accept()
    sm = get_session_manager()

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") != "user_message":
                await websocket.send_json({
                    "type": "error",
                    "content": "Invalid message type. Expected 'user_message'."
                })
                continue

            if sm is None:
                await websocket.send_json({
                    "type": "error",
                    "content": "SessionManager not initialized."
                })
                continue

            try:
                future = await sm.process_message(session_id, {
                    "role": "user",
                    "content": data.get("content", ""),
                    "sender_id": "websocket_user"
                })
                events = await future
                for event in events:
                    await websocket.send_json({
                        "type": event.type,
                        "content": event.content,
                        "metadata": event.metadata,
                        "timestamp": event.timestamp.isoformat()
                    })
            except ValueError as e:
                await websocket.send_json({
                    "type": "error",
                    "content": str(e)
                })
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "content": f"Processing error: {str(e)}"
                })

    except WebSocketDisconnect:
        pass
