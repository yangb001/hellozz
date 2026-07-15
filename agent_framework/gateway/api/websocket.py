"""WebSocket API router.

This module defines the WebSocket endpoint for real-time chat communication.
"""
import asyncio
import logging
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..dependencies import get_session_manager

logger = logging.getLogger("agent_framework.gateway.websocket")
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
    connection_id = str(uuid.uuid4())[:8]
    logger.debug(f"[WS:{connection_id}] WebSocket connected | session_id={session_id}")

    try:
        while True:
            data = await websocket.receive_json()
            logger.debug(f"[WS:{connection_id}] RECEIVED from client: {data}")

            if data.get("type") != "user_message":
                error_msg = "Invalid message type. Expected 'user_message'."
                logger.debug(f"[WS:{connection_id}] Sending error: {error_msg}")
                await websocket.send_json({
                    "type": "error",
                    "content": error_msg
                })
                continue

            if sm is None:
                error_msg = "SessionManager not initialized."
                logger.debug(f"[WS:{connection_id}] Sending error: {error_msg}")
                await websocket.send_json({
                    "type": "error",
                    "content": error_msg
                })
                continue

            try:
                logger.debug(f"[WS:{connection_id}] Processing message with streaming...")
                async for event in sm.process_message_stream(session_id, {
                    "role": "user",
                    "content": data.get("content", ""),
                    "sender_id": "websocket_user"
                }):
                    event_json = {
                        "type": event.type,
                        "content": event.content,
                        "metadata": event.metadata,
                        "timestamp": event.timestamp.isoformat()
                    }
                    logger.debug(f"[WS:{connection_id}] Streaming event: type={event.type}, content_len={len(event.content) if event.content else 0}")
                    try:
                        await websocket.send_json(event_json)
                        logger.debug(f"[WS:{connection_id}] Event SENT successfully")
                    except Exception as e:
                        logger.error(f"[WS:{connection_id}] WebSocket send failed: {e}", exc_info=True)
                        break
                logger.debug(f"[WS:{connection_id}] Streaming completed")
            except ValueError as e:
                error_msg = str(e)
                logger.debug(f"[WS:{connection_id}] Sending ValueError: {error_msg}")
                await websocket.send_json({
                    "type": "error",
                    "content": error_msg
                })
            except Exception as e:
                logger.error(f"[WS:{connection_id}] Error processing WebSocket message in session {session_id}: {e}", exc_info=True)
                await websocket.send_json({
                    "type": "error",
                    "content": f"Processing error: {str(e)}"
                })

    except WebSocketDisconnect:
        pass
