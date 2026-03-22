"""
Open WebUI Tools — Perception Sandbox.

Model-callable tools that proxy to the perception sidecar.
These are lightweight HTTP clients — no inference runs here.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

log = logging.getLogger("perception_tools")

SIDECAR_URL = os.environ.get("PERCEPTION_SIDECAR_URL", "http://localhost:8200")


class Tools:
    """
    Perception Sandbox tools exposed to the reasoning model in Open WebUI.

    The model can call these tools to inspect image attachments
    that were staged by the inlet filter.
    """

    class Valves:
        """User-configurable settings exposed in Open WebUI."""
        sidecar_url: str = SIDECAR_URL

    def __init__(self):
        self.valves = self.Valves()

    async def list_attachments(
        self,
        scope: str = "turn",
        __metadata__: dict[str, Any] | None = None,
    ) -> str:
        """
        List attachments available in the current perception sandbox.

        Returns a concise manifest of staged files for this turn.

        :param scope: Scope of attachments to list. Currently only "turn" is supported.
        :return: Text listing of available attachments.
        """
        session_id, turn_id = _extract_scope(__metadata__)
        if not session_id or not turn_id:
            return "No perception sandbox is active for this turn. No attachments are available."

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{self.valves.sidecar_url}/attachments/list",
                    json={"session_id": session_id, "turn_id": turn_id},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("display_text", "No attachments found.")
                else:
                    return f"Error listing attachments: {resp.status_code}"
        except httpx.RequestError as e:
            log.error("Sidecar request failed: %s", e)
            return "Perception sidecar is not reachable. Cannot list attachments."

    async def inspect_image(
        self,
        name: str,
        intent: str = "general",
        query: str = "",
        __metadata__: dict[str, Any] | None = None,
    ) -> str:
        """
        Inspect an image attachment using the perception backend.

        Analyses the named image and returns a textual description.

        :param name: Logical filename of the attachment (e.g. "image_1.jpg").
        :param intent: Analysis type — "general" for description, "ocr" for text extraction, "regions" for region analysis.
        :param query: Optional query for region-based analysis (used with intent="regions").
        :return: Textual analysis result from the perception backend.
        """
        session_id, turn_id = _extract_scope(__metadata__)
        if not session_id or not turn_id:
            return "No perception sandbox is active. Cannot inspect image."

        if intent not in ("general", "ocr", "regions"):
            return f"Invalid intent '{intent}'. Use 'general', 'ocr', or 'regions'."

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.valves.sidecar_url}/inspect",
                    json={
                        "session_id": session_id,
                        "turn_id": turn_id,
                        "logical_name": name,
                        "intent": intent,
                        "query": query,
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("result_text", "No result returned.")
                elif resp.status_code == 404:
                    return f"Attachment '{name}' not found in the sandbox. Use list_attachments() to see available files."
                elif resp.status_code == 503:
                    return "The perception backend is not available. The Florence model may not be enabled."
                else:
                    detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type", "").startswith("application/json") else resp.text
                    return f"Inspection failed ({resp.status_code}): {detail}"
        except httpx.RequestError as e:
            log.error("Sidecar request failed: %s", e)
            return "Perception sidecar is not reachable. Cannot inspect image."

    async def tag_image(
        self,
        name: str,
        threshold: float = 0.35,
        __metadata__: dict[str, Any] | None = None,
    ) -> str:
        """
        Tag an image using an optional specialist tagger backend.

        Returns ranked tags with confidence scores. Only works if the tagger
        backend is enabled in the perception sidecar configuration.

        :param name: Logical filename of the attachment (e.g. "image_1.jpg").
        :param threshold: Minimum confidence threshold for tags (0.0-1.0). Default: 0.35.
        :return: Ranked list of tags with confidence scores.
        """
        session_id, turn_id = _extract_scope(__metadata__)
        if not session_id or not turn_id:
            return "No perception sandbox is active. Cannot tag image."

        threshold = max(0.0, min(1.0, threshold))

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.valves.sidecar_url}/tag",
                    json={
                        "session_id": session_id,
                        "turn_id": turn_id,
                        "logical_name": name,
                        "threshold": threshold,
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("display_text", "No tags returned.")
                elif resp.status_code == 404:
                    return f"Attachment '{name}' not found in the sandbox. Use list_attachments() to see available files."
                elif resp.status_code == 503:
                    return "The tagger backend is not enabled. Ask the administrator to enable it in the sidecar config."
                else:
                    detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type", "").startswith("application/json") else resp.text
                    return f"Tagging failed ({resp.status_code}): {detail}"
        except httpx.RequestError as e:
            log.error("Sidecar request failed: %s", e)
            return "Perception sidecar is not reachable. Cannot tag image."


# ── Helpers ───────────────────────────────────────────────────────────

def _extract_scope(metadata: dict[str, Any] | None) -> tuple[str, str]:
    """Extract perception sandbox scope IDs from Open WebUI metadata."""
    if not metadata:
        return "", ""
    session_id = metadata.get("perception_session_id", "")
    turn_id = metadata.get("perception_turn_id", "")
    return session_id, turn_id
