"""Search and play Spotify music on the Windows laptop only."""

from __future__ import annotations

import base64
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

from openjarvis.connectors.oauth import save_tokens
from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec


TOOL_NAME = "spotify_play"
API_BASE = "https://api.spotify.com/v1"
TOKEN_URL = "https://accounts.spotify.com/api/token"
TOKEN_PATH = Path.home() / ".openjarvis" / "connectors" / "spotify.json"

# Never send playback to a TV, speaker, soundbar, etc.
PREFERRED_COMPUTER = "CAMARASURFACE2"

# Time allowed for Windows Spotify to start and register with Spotify Connect.
SPOTIFY_START_TIMEOUT = 20


@ToolRegistry.register(TOOL_NAME)
class SpotifyPlayTool(BaseTool):
    tool_id = TOOL_NAME
    is_local = False

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=TOOL_NAME,
            description=(
                "Search Spotify and play the requested music on the Windows laptop "
                "CAMARASURFACE2. If Spotify is closed, this tool automatically opens "
                "the Windows Spotify application, waits for the laptop to become "
                "available in Spotify Connect, and then starts playback. "
                "Use this tool directly for Spotify playback requests. "
                "Do NOT use shell_exec to open Spotify before or after this tool. "
                "This tool never sends playback to speakers, soundbars, TVs, or "
                "other Spotify Connect devices."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Song and artist to play, for example "
                            "'Back in Black by AC/DC'."
                        ),
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            category="music",
            requires_confirmation=False,
            timeout_seconds=45.0,
            required_capabilities=["network:fetch", "code:execute"],
        )

    def execute(self, **params: Any) -> ToolResult:
        query = str(params.get("query", "")).strip()

        if not query:
            return self._fail("A Spotify search query is required.")

        try:
            tokens = self._load_tokens()

            # ---------------------------------------------------------
            # 1. Look specifically for CAMARASURFACE2.
            # ---------------------------------------------------------
            device = self._find_laptop(tokens)
            spotify_was_opened = False

            # ---------------------------------------------------------
            # 2. If laptop is not available, Spotify is probably closed.
            #    Open Spotify on Windows and wait for Spotify Connect.
            # ---------------------------------------------------------
            if device is None:
                self._open_windows_spotify()
                spotify_was_opened = True

                deadline = time.monotonic() + SPOTIFY_START_TIMEOUT

                while time.monotonic() < deadline:
                    time.sleep(1)
                    device = self._find_laptop(tokens)

                    if device is not None:
                        break

            if device is None:
                return self._fail(
                    "Spotify was opened, but CAMARASURFACE2 did not become "
                    "available in Spotify Connect within 20 seconds."
                )

            # ---------------------------------------------------------
            # 3. Search for requested song.
            # ---------------------------------------------------------
            response = self._request(
                "GET",
                "/search",
                tokens,
                params={
                    "q": query,
                    "type": "track",
                    "limit": 5,
                },
            )

            if response.status_code != 200:
                return self._fail(
                    f"Spotify search failed (HTTP {response.status_code})."
                )

            tracks = response.json().get("tracks", {}).get("items", [])

            if not tracks:
                return self._fail(
                    f"No Spotify track was found for: {query}"
                )

            track = tracks[0]
            track_uri = track.get("uri")

            if not track_uri:
                return self._fail(
                    "Spotify returned a track without a URI."
                )

            # ---------------------------------------------------------
            # 4. Play ONLY on CAMARASURFACE2.
            # ---------------------------------------------------------
            response = self._request(
                "PUT",
                "/me/player/play",
                tokens,
                params={"device_id": device["id"]},
                json={"uris": [track_uri]},
            )

            if response.status_code != 204:
                return self._fail(
                    f"Spotify playback failed (HTTP {response.status_code}): "
                    f"{response.text[:300]}"
                )

            artists = ", ".join(
                artist.get("name", "")
                for artist in track.get("artists", [])
                if artist.get("name")
            )

            result = {
                "status": "playing",
                "track": track.get("name", ""),
                "artist": artists,
                "device": device.get("name", ""),
                "device_type": device.get("type", ""),
                "uri": track_uri,
                "spotify_was_opened": spotify_was_opened,
            }

            return ToolResult(
                tool_name=TOOL_NAME,
                content=json.dumps(result, ensure_ascii=False),
                success=True,
            )

        except Exception as exc:
            return self._fail(f"Spotify playback error: {exc}")

    def _find_laptop(
        self,
        tokens: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Return CAMARASURFACE2 only. Never fall back to another device."""

        response = self._request(
            "GET",
            "/me/player/devices",
            tokens,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Could not list Spotify devices "
                f"(HTTP {response.status_code})."
            )

        devices = response.json().get("devices", [])

        for device in devices:
            name = str(device.get("name", ""))
            device_type = str(device.get("type", ""))

            if (
                name.casefold() == PREFERRED_COMPUTER.casefold()
                and device_type.casefold() == "computer"
                and not device.get("is_restricted", False)
            ):
                return device

        return None

    @staticmethod
    def _open_windows_spotify() -> None:
        """Launch Windows Spotify from WSL2."""

        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                "Start-Process 'spotify:'",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "Could not launch Windows Spotify: "
                + (result.stderr.strip() or "unknown error")
            )

    @staticmethod
    def _load_tokens() -> dict[str, Any]:
        if not TOKEN_PATH.exists():
            raise RuntimeError(
                "Spotify is not connected. Run: jarvis connect spotify"
            )

        tokens = json.loads(
            TOKEN_PATH.read_text(encoding="utf-8")
        )

        if not tokens.get("access_token"):
            raise RuntimeError(
                "Spotify access token is missing."
            )

        return tokens

    @staticmethod
    def _refresh_access_token(
        tokens: dict[str, Any],
    ) -> str:
        refresh_token = tokens.get("refresh_token")
        client_id = tokens.get("client_id")
        client_secret = tokens.get("client_secret")

        if not all(
            (refresh_token, client_id, client_secret)
        ):
            raise RuntimeError(
                "Spotify token expired and refresh credentials "
                "are unavailable."
            )

        basic = base64.b64encode(
            f"{client_id}:{client_secret}".encode()
        ).decode()

        response = httpx.post(
            TOKEN_URL,
            headers={
                "Authorization": f"Basic {basic}"
            },
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            },
            timeout=20,
        )

        response.raise_for_status()

        refreshed = response.json()
        access_token = refreshed["access_token"]

        tokens["access_token"] = access_token

        if refreshed.get("refresh_token"):
            tokens["refresh_token"] = (
                refreshed["refresh_token"]
            )

        save_tokens(str(TOKEN_PATH), tokens)

        return access_token

    def _request(
        self,
        method: str,
        endpoint: str,
        tokens: dict[str, Any],
        **kwargs: Any,
    ) -> httpx.Response:

        token = tokens["access_token"]

        headers = dict(
            kwargs.pop("headers", {})
        )

        headers["Authorization"] = (
            f"Bearer {token}"
        )

        response = httpx.request(
            method,
            f"{API_BASE}{endpoint}",
            headers=headers,
            timeout=20,
            **kwargs,
        )

        # Spotify access tokens expire.
        # Refresh once and retry automatically.
        if response.status_code == 401:
            token = self._refresh_access_token(
                tokens
            )

            headers["Authorization"] = (
                f"Bearer {token}"
            )

            response = httpx.request(
                method,
                f"{API_BASE}{endpoint}",
                headers=headers,
                timeout=20,
                **kwargs,
            )

        return response

    @staticmethod
    def _fail(message: str) -> ToolResult:
        return ToolResult(
            tool_name=TOOL_NAME,
            content=message,
            success=False,
        )


__all__ = ["SpotifyPlayTool"]
