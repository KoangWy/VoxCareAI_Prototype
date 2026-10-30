"""The voice WebSocket: connection lifecycle and client-message dispatch.

One :class:`VoiceConnection` per socket wraps a :class:`VoiceState` and
handles each client ``type`` from ``docs/VOICE_PROTOCOL.md``: ``start``,
binary PCM, ``vad_speech_start`` / ``vad_quiet``, the recovery triplet,
playback acks, ``stop``, and ``reset``. Wire names come from
:mod:`voxcare.voice.protocol`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from ..config import ConfigError, config_error_code, load_turn_settings, turn_mode_from_env
from ..providers import ProviderUnavailable, asr_connection_from_env, provider_status, tts_from_env
from ..session.render import HANDOFF_TEXT
from ..turns import TurnInput
from . import protocol as vm
from .asr_events import handle_asr_events
from .dispatch import dispatch_turn
from .evaluate import cancel_evaluations, evaluate_current_turn, schedule_after_quiet
from .state import VoiceState
from .turn_control import cancel_all_deadlines, send_turn_control

logger = logging.getLogger("voxcare.voice")

MAX_AUDIO_FRAME_BYTES = 64 * 1024
MAX_SESSION_AUDIO_BYTES = 24 * 1024 * 1024
_FALLBACK_ACTIONS = {
    vm.CLIENT_MANUAL_FINISH: "MANUAL_FINISH",
    vm.CLIENT_CALLER_RETRY: "CALLER_RETRY",
    vm.CLIENT_HANDOFF_REQUEST: "CALLER_REQUESTED_HANDOFF",
}


class VoiceConnection:
    """One live WebSocket voice session."""

    def __init__(self, boundary, session_id: str, websocket: WebSocket) -> None:  # type: ignore[no-untyped-def]
        self.boundary = boundary
        self.session_id = session_id
        self.websocket = websocket
        self.state = VoiceState()
        self.send_lock = asyncio.Lock()

    async def send_json(self, payload: dict) -> None:  # type: ignore[type-arg]
        async with self.send_lock:
            await self.websocket.send_json(payload)

    async def send_binary(self, payload: bytes) -> None:
        async with self.send_lock:
            await self.websocket.send_bytes(payload)

    async def evaluate(self) -> None:
        await evaluate_current_turn(self.boundary, self.send_json, self.send_binary, self.session_id, self.state)

    def _spawn_response(self, coro) -> None:  # type: ignore[no-untyped-def]
        task = asyncio.create_task(coro)
        self.state.response_tasks.add(task)
        task.add_done_callback(self.state.response_tasks.discard)

    async def _send_control(self, decision) -> None:  # type: ignore[no-untyped-def]
        await send_turn_control(
            self.boundary,
            self.send_json,
            self.session_id,
            self.state,
            decision,
            version=self.state.transcript_version,
            generation=self.state.response_generation,
        )

    # -- connection lifecycle --

    async def hello(self) -> None:
        """Session hello plus any turn-decision configuration problem, before any audio."""
        from ..http.routes import JUDGED_TURN_MODES, provider_payload, session_payload

        state = self.state
        state.turn_controller = await self.boundary.get_turn_controller(self.session_id)
        config_error = None
        try:
            settings = load_turn_settings()
            state.turn_mode, state.barge_in = settings.mode, settings.barge_in
        except ConfigError as exc:
            # Session creation rejects bad config, but the env can change under a live process.
            state.turn_mode, config_error = turn_mode_from_env(), config_error_code(exc)
        session = await self.boundary.get_session(self.session_id)
        await self.send_json(
            {
                "type": vm.SERVER_SESSION,
                "session": session_payload(session),
                "providers": provider_payload(),
                "protocol": vm.PROTOCOL_VERSION,
                "barge_in": state.barge_in,
            }
        )
        error_code = config_error
        key_name = JUDGED_TURN_MODES.get(state.turn_mode) if config_error is None else None
        if key_name and not os.getenv(key_name, "").strip():
            error_code = f"{key_name}_NOT_CONFIGURED"
        if error_code:
            await self.send_json(
                {"type": vm.SERVER_PROVIDER_UNAVAILABLE, "provider": "turn_decision", "error": error_code}
            )
            await self.boundary.record_observation(
                self.session_id,
                "provider_unavailable",
                provider="turn_decision",
                error_code=error_code,
                mode=state.turn_mode,
            )

    async def close(self) -> None:
        state = self.state
        state.response_generation += 1
        state.audio_generation += 1
        cancel_evaluations(state)
        cancel_all_deadlines(state)
        for task in tuple(state.response_tasks | state.evaluation_tasks):
            task.cancel()
        if state.asr is not None:
            with contextlib.suppress(Exception):
                await state.asr.close()
            state.asr = None
        if state.asr_task is not None:
            state.asr_task.cancel()
            await asyncio.gather(state.asr_task, return_exceptions=True)
            state.asr_task = None

    # -- control messages --

    async def on_start(self, control: dict) -> bool:  # type: ignore[type-arg]
        """Attach ASR/TTS. Returns False when the socket must close."""
        state = self.state
        if (
            control.get("sample_rate") != 16_000
            or control.get("channels") != 1
            or control.get("encoding") != "pcm_s16le"
        ):
            await self.send_json(
                {"type": vm.SERVER_ERROR, "code": vm.ERROR_UNSUPPORTED_AUDIO_FORMAT, "expected": "16 kHz mono PCM16"}
            )
            return True
        clock = control.get("client_clock")
        if isinstance(clock, dict) and isinstance(clock.get("name"), str):
            state.client_clock = {"name": clock["name"], "resolution_ms": clock.get("resolution_ms")}
            await self.boundary.record_observation(self.session_id, "client_clock", clock=state.client_clock)
        if state.asr is None:
            try:
                asr = asr_connection_from_env()
                await asr.connect()
            except ProviderUnavailable as exc:
                await self.send_json(
                    {
                        "type": vm.SERVER_PROVIDER_UNAVAILABLE,
                        "provider": "asr",
                        "error": exc.code or "ASR_NOT_CONFIGURED",
                    }
                )
                await self.websocket.close(code=1013, reason="speech provider not configured")
                return False
            except Exception as exc:
                logger.warning("ASR connection failed", extra={"error_type": type(exc).__name__})
                await self.send_json(
                    {"type": vm.SERVER_PROVIDER_UNAVAILABLE, "provider": "asr", "error": "ASR_CONNECTION_FAILED"}
                )
                return False
            state.asr = asr
            try:
                state.tts = tts_from_env()
            except ProviderUnavailable:
                state.tts = None
            else:
                self._spawn_response(self._warm_tts(state.tts))
            state.asr_task = asyncio.create_task(
                handle_asr_events(self.boundary, self.send_json, self.session_id, state, asr, self.evaluate)
            )
        await self.send_json(
            {"type": vm.SERVER_VOICE_STARTED, "provider": provider_status()["asr"].name, "locale": "en-US"}
        )
        return True

    async def _warm_tts(self, tts) -> None:  # type: ignore[no-untyped-def]
        warm = getattr(tts, "warm", None)
        if warm is None:
            return
        try:
            await warm()
        except Exception as exc:
            logger.warning("TTS warm-up failed", extra={"error_type": type(exc).__name__})

    async def on_vad_speech_start(self, control: dict) -> None:  # type: ignore[type-arg]
        """Local-VAD onset: the only signal that resets the HOLD budget.

        Barge-in ``off`` (default): the onset retires everything in flight,
        including assistant audio. ``semantic``: an onset the browser reports
        while the reply is audible holds the audio and lets the judge rule on
        ``taking_floor``; only an ``interrupt`` verdict stops playback.
        """
        state = self.state
        held = (
            state.barge_in == "semantic" and control.get("agent_speaking") is True and bool(state.assistant_speech_text)
        )
        cancel_evaluations(state)
        if held:
            state.barge_in_held = True
            await self.boundary.record_observation(
                self.session_id,
                "barge_in_held",
                turn_id=state.current_turn_id,
                transcript_version=state.transcript_version,
                agent_speaking=True,
            )
        else:
            state.response_generation += 1
            for task in tuple(state.response_tasks):
                task.cancel()
            cancel_all_deadlines(state)
            state.retire_audio()
            state.pending_recovery = False
            state.asr_final = False
        if state.turn_completed:
            # The previous turn is done; this onset opens a fresh one.
            state.reset_turn_buffers(new_turn_id=True)
            await self.boundary.set_live_transcript(self.session_id, "")
        state.transcript_version += 1
        state.mark_speaking()
        current = await self.boundary.get_session(self.session_id)
        controller = await self.boundary.get_turn_controller(self.session_id)
        if current is not None and controller is not None:
            decision = await controller.on_genuine_speech(
                TurnInput(
                    session_id=self.session_id,
                    turn_epoch=current.turn_epoch + 1,
                    transcript="",
                    source="local_vad",
                    local_vad_quiet=False,
                )
            )
            await self._send_control(decision)
        await self.boundary.record_observation(
            self.session_id,
            "local_vad_speech_start",
            turn_id=state.current_turn_id,
            transcript_version=state.transcript_version,
            segment_id=control.get("segment_id"),
            client_monotonic_ms=control.get("client_monotonic_ms"),
            client_clock=state.client_clock,
            barge_in="held_for_judgment" if held else state.barge_in,
        )

    async def on_vad_quiet(self, control: dict) -> None:  # type: ignore[type-arg]
        state = self.state
        # performance.now() deltas are fractional: accept any number and round.
        raw_silence = control.get("silence_ms")
        silence = (
            round(raw_silence) if isinstance(raw_silence, (int, float)) and not isinstance(raw_silence, bool) else None
        )
        state.mark_quiet(silence)
        speech_end = control.get("speech_end_monotonic_ms")
        state.t0_by_turn[state.current_turn_id] = {
            "client_monotonic_ms": speech_end,
            "client_clock": state.client_clock,
            "source": "local_vad_speech_end",
            "status": "available" if isinstance(speech_end, (int, float)) else "unavailable",
        }
        await self.boundary.record_observation(
            self.session_id,
            "local_vad_quiet",
            turn_id=state.current_turn_id,
            transcript_version=state.transcript_version,
            segment_id=control.get("segment_id"),
            T0_client_monotonic_ms=speech_end,
            quiet_client_monotonic_ms=control.get("client_monotonic_ms"),
            silence_ms=silence,
            client_clock=state.client_clock,
        )
        schedule_after_quiet(state, self.evaluate)

    async def on_recovery_action(self, kind: str, control: dict) -> None:  # type: ignore[type-arg]
        state = self.state
        if control.get("transcript_version") != state.transcript_version:
            await self.send_json({"type": vm.SERVER_ERROR, "code": vm.ERROR_STALE_RECOVERY_ACTION})
            return
        current = await self.boundary.get_session(self.session_id)
        controller = await self.boundary.get_turn_controller(self.session_id) if current is not None else None
        if current is None or controller is None:
            await self.send_json({"type": vm.SERVER_ERROR, "code": vm.ERROR_SESSION_EXPIRED})
            return
        if not state.vad_quiet:
            await self.send_json({"type": vm.SERVER_ERROR, "code": vm.ERROR_WAIT_FOR_LOCAL_VAD_QUIET})
            return
        epoch = current.turn_epoch + 1
        hold_episode_id = control.get("hold_episode_id")
        decision = None
        # A support request needs no degraded episode; finish and retry do.
        if kind != vm.CLIENT_HANDOFF_REQUEST or hold_episode_id:
            decision = await controller.resolve_fallback(
                self.session_id, epoch, hold_episode_id or "", _FALLBACK_ACTIONS[kind]
            )
            accepted = decision.state == "FALLBACK_RESOLVED" or decision.reason == "caller_selected_retry"
            if kind != vm.CLIENT_HANDOFF_REQUEST and not accepted:
                await self.send_json({"type": vm.SERVER_ERROR, "code": vm.ERROR_FALLBACK_REQUIRES_DEGRADED_STATE})
                return
            await self._send_control(decision)
        if kind == vm.CLIENT_MANUAL_FINISH:
            state.turn_completed = True
            cancel_evaluations(state)
            self._spawn_response(
                dispatch_turn(
                    self.boundary,
                    self.send_json,
                    self.send_binary,
                    self.session_id,
                    state,
                    state.transcript,
                    state.transcript_version,
                    state.response_generation,
                    state.current_turn_id,
                )
            )
        elif kind == vm.CLIENT_CALLER_RETRY:
            # "Try again": discard the degraded attempt's words and listen afresh.
            state.response_generation += 1
            cancel_evaluations(state)
            state.reset_turn_buffers(new_turn_id=False)
            state.transcript_version += 1
            state.mark_speaking()
            state.pending_recovery = False
            await self.boundary.set_live_transcript(self.session_id, "")
            await self.send_json(
                {
                    "type": vm.SERVER_RETRY_READY,
                    "turn_id": state.current_turn_id,
                    "transcript_version": state.transcript_version,
                    "hold_episode_id": decision.hold_episode_id if decision else None,
                }
            )
        else:
            await self.boundary.record_handoff(
                self.session_id,
                reason=str(control.get("reason") or "caller_requested_handoff"),
                context=state.transcript,
                turn_id=state.current_turn_id,
                turn_epoch=epoch,
            )
            await self.send_json({"type": vm.SERVER_HANDOFF_RECORDED, "message": HANDOFF_TEXT})

    async def on_playback_started(self, control: dict) -> None:  # type: ignore[type-arg]
        state = self.state
        if (
            control.get("turn_id") != state.active_audio_turn_id
            or control.get("audio_chunk_id") != state.active_audio_chunk_id
        ):
            await self.boundary.record_observation(
                self.session_id, "stale_playback_ack", turn_id=control.get("turn_id")
            )
            return
        t6 = control.get("client_monotonic_ms")
        t0 = state.t0_by_turn.get(state.active_audio_turn_id or "", {})
        resolution = state.client_clock.get("resolution_ms")
        # TTFA needs an acoustic confirmation on the same client clock as T0; this demo never has one.
        same_clock = (
            control.get("acoustic_confirmed") is True
            and bool(state.client_clock.get("name"))
            and control.get("clock_name") == state.client_clock.get("name")
            and isinstance(resolution, (int, float))
            and resolution > 0
            and t0.get("client_clock") == state.client_clock
            and isinstance(t0.get("client_monotonic_ms"), (int, float))
            and isinstance(t6, (int, float))
            and float(t6) >= float(t0["client_monotonic_ms"])
        )
        ttfa = round(float(t6) - float(t0["client_monotonic_ms"]), 3) if same_clock else None  # type: ignore[arg-type]
        await self.boundary.record_observation(
            self.session_id,
            "playback_started",
            turn_id=state.active_audio_turn_id,
            audio_chunk_id=control.get("audio_chunk_id"),
            T6_client_monotonic_ms=t6,
            client_clock=state.client_clock,
            ttfa_ms=ttfa,
            ttfa_status="measured_client_clock" if ttfa is not None else "unavailable_clock_mismatch_or_T0_missing",
        )
        await self.send_json(
            {
                "type": vm.SERVER_PLAYBACK_ACK_RECORDED,
                "turn_id": state.active_audio_turn_id,
                "ttfa_ms": ttfa,
                "ttfa_status": "measured_client_clock" if ttfa is not None else "unavailable",
            }
        )

    async def on_playback_failed(self, control: dict) -> None:  # type: ignore[type-arg]
        await self.boundary.record_observation(
            self.session_id, "playback_failed", turn_id=control.get("turn_id"), error_code=control.get("error_code")
        )
        await self.send_json({"type": vm.SERVER_PLAYBACK_FAILED_RECORDED, "turn_id": control.get("turn_id")})

    async def on_stop(self) -> None:
        if self.state.asr is not None:
            await self.state.asr.finalize()
        await self.send_json({"type": vm.SERVER_VOICE_STOPPING})

    async def on_reset(self) -> None:
        from ..http.routes import session_payload

        await self.close()
        await self.boundary.record_observation(self.session_id, "session_reset_requested")
        new_session = await self.boundary.reset_session(self.session_id)
        if new_session is not None:
            await self.send_json({"type": vm.SERVER_SESSION_RESET, "session": session_payload(new_session)})

    async def on_binary(self, frame: bytes) -> bool:
        """Forward one mic frame. Returns False when the socket must close."""
        state = self.state
        if state.asr is None:
            await self.send_json({"type": vm.SERVER_ERROR, "code": vm.ERROR_SEND_START_FIRST})
            return True
        if len(frame) > MAX_AUDIO_FRAME_BYTES:
            await self.send_json({"type": vm.SERVER_ERROR, "code": vm.ERROR_AUDIO_FRAME_TOO_LARGE})
            with contextlib.suppress(Exception):
                await self.websocket.close(code=1009, reason="audio frame too large")
            return False
        state.audio_bytes += len(frame)
        if state.audio_bytes > MAX_SESSION_AUDIO_BYTES:
            await self.send_json({"type": vm.SERVER_ERROR, "code": vm.ERROR_SESSION_AUDIO_LIMIT_REACHED})
            with contextlib.suppress(Exception):
                await self.websocket.close(code=1008, reason="session audio budget exceeded")
            return False
        await state.asr.send_audio(frame)
        return True

    async def dispatch(self, control: dict) -> bool:  # type: ignore[type-arg]
        """Handle one control message. Returns False when the socket loop must end."""
        kind = control.get("type")
        if kind == vm.CLIENT_START:
            return await self.on_start(control)
        if kind == vm.CLIENT_VAD_SPEECH_START:
            await self.on_vad_speech_start(control)
        elif kind == vm.CLIENT_VAD_QUIET:
            await self.on_vad_quiet(control)
        elif kind in _FALLBACK_ACTIONS:
            await self.on_recovery_action(kind, control)
        elif kind == vm.CLIENT_PLAYBACK_STARTED:
            await self.on_playback_started(control)
        elif kind == vm.CLIENT_PLAYBACK_FAILED:
            await self.on_playback_failed(control)
        elif kind == vm.CLIENT_STOP:
            await self.on_stop()
        elif kind == vm.CLIENT_RESET:
            await self.on_reset()
            return False
        else:
            await self.send_json({"type": vm.SERVER_ERROR, "code": vm.ERROR_UNKNOWN_CONTROL_MESSAGE})
        return True


async def voice_socket(websocket: WebSocket, session_id: str, boundary) -> None:  # type: ignore[no-untyped-def]
    if await boundary.get_session(session_id) is None:
        await websocket.close(code=4404, reason="session unavailable")
        return
    await websocket.accept()
    conn = VoiceConnection(boundary, session_id, websocket)
    try:
        await conn.hello()
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            frame = message.get("bytes")
            if frame is not None:
                if not await conn.on_binary(frame):
                    break
                continue
            payload = message.get("text")
            if payload is None:
                continue
            try:
                control = json.loads(payload)
            except json.JSONDecodeError:
                control = None
            if not isinstance(control, dict):
                await conn.send_json({"type": vm.SERVER_ERROR, "code": vm.ERROR_INVALID_CONTROL_MESSAGE})
                continue
            if not await conn.dispatch(control):
                break
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning("voice session ended", extra={"error_type": type(exc).__name__})
    finally:
        await conn.close()


def register(app: FastAPI) -> None:
    from ..http.bootstrap import boundary

    @app.websocket("/ws/sessions/{session_id}")
    async def ws_sessions(websocket: WebSocket, session_id: str) -> None:
        await voice_socket(websocket, session_id, boundary)


__all__ = ["MAX_AUDIO_FRAME_BYTES", "MAX_SESSION_AUDIO_BYTES", "VoiceConnection", "register", "voice_socket"]
