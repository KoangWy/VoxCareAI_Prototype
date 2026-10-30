"""Shared judge prompts and state — one source for both judged adapters.

`TypeSafeJevJudge` (`jev.py`) and `OpenRouterDecisionsJudge`
(`openrouter.py`) send the same four Noul questions over the same state
shape; only the transport differs (TypeSafe SDK vs OpenRouter HTTP).
The wording ports the reference implementation's prompts
(`convoai-jev-vad/server/src/jev_eos.py`): TIMING_NOTES, the trajectory /
release rule, silence heuristics, and worked Q&A, adapted from Ada/Agora to
the VoxCare caller/assistant setting.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..privacy import mask_pii
from .controller import TurnInput

CONVERSATION_SETTING = (
    "A natural English voice conversation between a caller and a customer-support assistant. "
    "The caller may answer a question briefly, pause while thinking, correct themselves, or "
    "continue after a pause. The assistant should answer only after the caller has handed over. "
    "The caller may be in a room with other people, on speaker, or near a TV."
)

TRANSCRIPT_NOTES = (
    "Live partial from a streaming speech recognizer. What you see is its best guess so far, "
    "not a final transcript. Capitalisation and punctuation are inserted automatically: a "
    "trailing period or question mark appears whenever the recognizer closes a segment and is "
    "NOT evidence the caller finished. Short partials are low confidence and are often revised "
    "or extended on the next update ('Yeah' → 'Yeah, so'). Fillers (um, uh, so, okay), repeated "
    "words and self-corrections are normal spoken behaviour, not signs the caller is done. "
    "Word errors happen — judge the intent, not the exact wording."
)

TIMING_NOTES = (
    "silence_since_last_word_ms: how long the microphone has been quiet since the last "
    "recognised word ended (0 or absent = the caller is still making sound). asr_final: the "
    "recognizer closed its segment (it detected a pause of its own); stable_word_ratio: share "
    "of words the recognizer is no longer revising. transcript_trajectory: how the text grew "
    "during this turn (oldest first, age in ms) — rapid recent growth means the caller is in "
    "full flow. turn_elapsed_ms: time since the turn opened; a question asked just before a "
    "short answer supports completion."
)

TURN_COMPLETE_INSTRUCTIONS = (
    "You are the end-of-turn detector in a live, spoken conversation between a human caller "
    "and a voice AI assistant. You do NOT hear audio. You see only the text a streaming "
    "speech recognizer has produced so far for the caller's turn plus client-measured timing "
    "(see timing_notes). Prosody, pitch, volume, breath, laughter and hums are invisible to "
    "you unless the recognizer rendered them as words. "
    "Decide whether the caller has finished what they wanted to say and now expects the "
    "assistant to respond. Use conversation_history and assistant_last_message: if the "
    "assistant just asked a question, a short direct answer is a complete turn. Treat the "
    "transcript as speech, not writing: punctuation is guessed and often wrong, fillers and "
    "false starts are normal, and a sentence can be grammatically complete yet clearly a "
    "lead-in (e.g. 'I have a question.', 'So here's the thing.', 'Okay but'). Conversely, "
    "short replies are often complete: when the assistant asked something, answers like 'I "
    "don't know', 'not sure', 'yes', 'no', 'maybe later', 'the second one' fully hand the "
    "floor back — do not wait for elaboration that was never promised. "
    "Within-turn pauses are usually under ~800 ms, so longer mic silence after a plausible "
    "answer is strong evidence they are done, while silence after a dangling connector means "
    "they are still thinking. asr_final=true and a high stable_word_ratio support completion "
    "but are not sufficient alone. "
    "Worked examples — assistant asked 'Which platform are you on?': "
    "'Web.' → complete. 'Web, and' → not complete. 'Web. Actually' → not complete. "
    "'I'm not sure yet' → complete (an answer). 'I'm not sure yet, because' → not complete. "
    "Assistant said 'Anything else?': 'No, that's all' → complete. 'Yeah' → ambiguous "
    "lead-in; complete only with clear silence. 'Yeah, one more thing' → not complete "
    "(announced content). 'Can you explain how tokens' → not complete (question unfinished). "
    "'Can you explain how tokens work?' → complete. "
    "'Okay so I have two questions. First' → not complete. "
    "If you say the turn is complete, the assistant will start speaking immediately and cut "
    "the caller off if they were about to continue."
)
TURN_COMPLETE_CRITERIA = {
    "true": (
        "The caller has said a complete thought, question, request, or answer that hands the "
        "floor to the assistant — including brief answers to the assistant's question ('I don't "
        "know', 'yes', 'not really'). Nothing signals more is coming: no trailing connector "
        "(and, but, so, because), no announced content still unsaid, no hedging mid-formulation. "
        "Sustained silence after such a reply confirms it."
    ),
    "false": (
        "The caller is still going: mid-sentence, trailing off, thinking aloud, ended on a "
        "connector or lead-in, announced a question or point they have not yet stated, or gave "
        "only part of an answer the assistant's question calls for."
    ),
}

HOLDING_FLOOR_INSTRUCTIONS = (
    "Judging from the recognizer's TEXT only (you cannot hear tone or pauses inside the "
    "audio), is the caller explicitly signalling that they need more time and intend to keep "
    "the floor — thinking aloud or stalling rather than handing over? Text cues: 'I'm thinking "
    "about it', 'let me think', 'hmm', 'uh, give me a second', 'hold on', 'how do I put this', "
    "'yeah... so...', 'what was it'. This is about the caller asking for patience in words, "
    "not merely being mid-sentence. Judge the MOST RECENT words, not the whole turn: a request "
    "for time is released the moment the caller says something that hands over or moves on "
    "('okay go ahead', 'please continue', 'never mind', 'so my question is …', an actual "
    "question or answer). 'Hold on a second. Please continue.' is NOT holding — the latest "
    "words release the floor. Also use transcript_trajectory: if the holding phrase is old "
    "and the text has since grown into content, the caller is no longer stalling."
)
HOLDING_FLOOR_CRITERIA = {
    "true": (
        "The LATEST words are a request for time or deliberation: hedges, transcribed fillers "
        "('hmm', 'um', 'uh'), 'let me think', 'one sec', 'hold on', or a bare acknowledgement "
        "('yeah', 'okay', 'right') that reads as a lead-in to more, with nothing after it that "
        "hands over."
    ),
    "false": (
        "No such words at the end of the turn — the caller is either clearly done, has released "
        "an earlier 'hold on' ('okay continue', 'go ahead'), has moved on to actual content, or "
        "is simply mid-way through a sentence with no request for time."
    ),
}

ADDRESSED_TO_AGENT_INSTRUCTIONS = (
    "The microphone picks up everything in the room. Judging from the TEXT only, is this "
    "utterance directed at the voice assistant, as opposed to someone else in the room, a "
    "phone call, the caller talking to themselves, or a TV/radio in the background? Use "
    "conversation_history and assistant_last_message: a reply that fits what the assistant "
    "just said or asked is almost certainly addressed to it, even if short or vague ('yes', "
    "'the first one', 'I don't know'). Questions about orders, policies, or the assistant "
    "itself, or generic assistant requests, are addressed to it. Third-party cues — a "
    "different name being addressed ('hey Mark'), household logistics (\"dinner's ready\", "
    "'can you grab the door'), replying to something the assistant never said, or fragments "
    "of someone else's conversation — point away from the assistant. Absent any cue, assume "
    "the caller is talking to the assistant: a single person on a call usually is. Also "
    "treat an unfinished partial as addressed unless there is a clear third-party cue; do "
    "not penalise incompleteness here. A turn can contain a mix (the caller spoke to someone "
    "else, then turned to the assistant): judge by the MOST RECENT part — if the latest "
    "words are for the assistant, the turn is addressed to it."
)
ADDRESSED_TO_AGENT_CRITERIA = {
    "true": (
        "The utterance plausibly continues the conversation with the assistant: it answers, "
        "follows up, asks, requests, or reacts to what the assistant said, or has no cue "
        "pointing to another listener."
    ),
    "false": (
        "The utterance is clearly meant for someone or something else: addresses another "
        "person by name or role, concerns the physical surroundings in a way that makes no "
        "sense for a voice assistant, responds to a question the assistant never asked, or "
        "reads like overheard background speech."
    ),
}

TAKING_FLOOR_INSTRUCTIONS = (
    "The assistant is currently speaking (assistant_current_speech is what it has said so "
    "far in this reply) and the caller's microphone produced the words in "
    "current_user_transcript. From TEXT only, decide whether the caller is genuinely trying "
    "to take the floor — so the assistant should stop talking right now — or merely reacting "
    "while continuing to listen. Backchannels and acknowledgements ('yeah', 'uh-huh', 'mm-hm', "
    "'right', 'okay', 'I see', 'got it', 'sure', 'nice', 'wow', laughter rendered as 'haha') "
    "are NOT interruptions. Neither is a fragment that echoes what the assistant just said "
    "(residual echo of its own voice) or a single ambiguous word. Real interruptions: 'wait', "
    "'stop', 'hold on', 'no no', 'actually', 'sorry but', a new question, a correction, a "
    "request to repeat or slow down, or several words of new content that do not track the "
    "assistant's speech. "
    "IMPORTANT exception: if the assistant asked the caller something — in "
    "assistant_current_speech or as its last message — and the caller's words answer it "
    "('pretty good', 'fine thanks', 'the second one', 'yes', 'no', 'not yet'), the caller IS "
    "taking the floor: they are replying, and the assistant should stop and listen rather "
    "than keep talking over the answer. Short evaluative replies only count as backchannels "
    "when there was no question to answer. "
    "Cutting the assistant off on a backchannel is worse than reacting a moment late to a "
    "real interruption."
)
TAKING_FLOOR_CRITERIA = {
    "true": (
        "The words are a deliberate attempt to stop the assistant or change course: a command "
        "to stop/wait, a correction, a new question or request, substantive new content while "
        "the assistant is talking, or an answer to a question the assistant just asked."
    ),
    "false": (
        "The words are a listener's backchannel, acknowledgement, laugh, a short ambiguous "
        "fragment, or text that mirrors the assistant's own speech — and they do not answer "
        "a question the assistant asked."
    ),
}

QUESTION_NAMES = ("turn_complete", "holding_floor", "addressed_to_agent", "taking_floor")


def judge_questions(*, include_taking_floor: bool) -> dict[str, dict[str, Any]]:
    """Noul question payloads shared by both judge adapters."""
    questions: dict[str, dict[str, Any]] = {
        "turn_complete": {
            "type": "noul",
            "instructions": TURN_COMPLETE_INSTRUCTIONS,
            "criteria": dict(TURN_COMPLETE_CRITERIA),
        },
        "holding_floor": {
            "type": "noul",
            "instructions": HOLDING_FLOOR_INSTRUCTIONS,
            "criteria": dict(HOLDING_FLOOR_CRITERIA),
        },
        "addressed_to_agent": {
            "type": "noul",
            "instructions": ADDRESSED_TO_AGENT_INSTRUCTIONS,
            "criteria": dict(ADDRESSED_TO_AGENT_CRITERIA),
        },
    }
    if include_taking_floor:
        questions["taking_floor"] = {
            "type": "noul",
            "instructions": TAKING_FLOOR_INSTRUCTIONS,
            "criteria": dict(TAKING_FLOOR_CRITERIA),
        }
    return questions


def state_from_turn(turn: TurnInput) -> dict[str, Any]:
    """Text-only decision state shared by every judge adapter.

    The four Noul questions and this state shape are the stable contract; only
    the transport differs between providers (TypeSafe SDK, OpenRouter HTTP).
    """
    state: dict[str, Any] = {
        "setting": CONVERSATION_SETTING,
        "transcript_notes": TRANSCRIPT_NOTES,
        "timing_notes": TIMING_NOTES,
        "conversation_history": [
            {"role": str(item.get("role", "unknown")), "text": mask_pii(str(item.get("text", "")))}
            for item in turn.conversation_history[-6:]
            if str(item.get("text", "")).strip()
        ],
        "assistant_last_message": mask_pii(turn.assistant_last_message or "(none yet)"),
        "current_user_transcript": mask_pii(turn.transcript),
        "assistant_current_speech": mask_pii(turn.assistant_current_speech or "(assistant silent)"),
    }
    if turn.silence_ms is not None:
        state["silence_since_last_word_ms"] = max(0, turn.silence_ms)
    if turn.asr_final is not None:
        state["asr_final"] = turn.asr_final
    if turn.stable_word_ratio is not None:
        state["stable_word_ratio"] = min(1.0, max(0.0, turn.stable_word_ratio))
    trajectory = [
        {"text": str(item.get("text", "")), "ms_ago": int(item.get("ms_ago", 0))}
        for item in turn.transcript_trajectory
        if str(item.get("text", "")).strip() and str(item.get("text", "")).strip() != turn.transcript.strip()
    ]
    if trajectory:
        state["transcript_trajectory"] = trajectory[-6:]
    if turn.turn_elapsed_ms is not None and turn.turn_elapsed_ms > 0:
        state["turn_elapsed_ms"] = int(turn.turn_elapsed_ms)
    return state


def trajectory_from_partials(
    partials: Sequence[Mapping[str, Any]], *, now_ms: int, current_text: str
) -> list[dict[str, Any]]:
    """Build the `transcript_trajectory` judge field from server-kept partials.

    Each partial is `{"text": ..., "at": <monotonic ms>}`; entries matching the
    current text are dropped (the reference sends only prior growth).
    """
    out: list[dict[str, Any]] = []
    for partial in partials:
        text = str(partial.get("text", "")).strip()
        if not text or text == (current_text or "").strip():
            continue
        try:
            at = int(partial.get("at", now_ms))
        except (TypeError, ValueError):
            at = now_ms
        out.append({"text": text, "ms_ago": max(0, now_ms - at)})
    return out[-6:]


def scores_from_answers(answers: Mapping[str, Any]) -> dict[str, float]:
    """Parse the four Noul scores; missing `taking_floor` means the assistant
    is silent and there is no floor to take — default 0.0, never an error."""
    scores = {key: float(answers[key].noul) for key in ("turn_complete", "holding_floor", "addressed_to_agent")}
    taking = answers.get("taking_floor")
    taking_noul = getattr(taking, "noul", None)
    scores["taking_floor"] = float(taking_noul) if taking_noul is not None else 0.0
    return scores


def scores_from_payload(payload: Mapping[str, Any]) -> dict[str, float]:
    """Same parse for the OpenRouter JSON envelope."""
    answers = payload["answers"]
    scores = {name: float(answers[name]["noul"]) for name in ("turn_complete", "holding_floor", "addressed_to_agent")}
    taking_entry = answers.get("taking_floor") if isinstance(answers, Mapping) else None
    taking_noul = taking_entry.get("noul") if isinstance(taking_entry, Mapping) else None
    scores["taking_floor"] = float(taking_noul) if taking_noul is not None else 0.0
    return scores
