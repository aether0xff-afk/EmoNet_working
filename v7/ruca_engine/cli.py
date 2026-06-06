from __future__ import annotations

import argparse
import json
from pathlib import Path

from .llm_client import LLMConfig
from .memory import MemoryStore
from .pipeline import RucaPipeline
from .session import SessionStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one Ruca/Rookie autonomous runtime event.")
    parser.add_argument("text", nargs="?", default="", help="User text to process. Omit for an autonomous no-speech event.")
    event_choices = ["user_message", "delayed_speech", "typing", "processing", "answering", "no_reply", "silence_tick", "long_silence"]
    parser.add_argument(
        "--event-type",
        default="user_message",
        choices=event_choices,
        help="Explicit event type. Use typing/processing/answering for always-on environmental ticks.",
    )
    parser.add_argument("--elapsed-minutes", type=float, default=0.0, help="Minutes elapsed without a new user message.")
    parser.add_argument("--silence", action="store_true", help="Compatibility alias for --event-type silence_tick.")
    parser.add_argument("--memory", type=Path, default=None, help="Optional JSON memory file path.")
    parser.add_argument("--session", type=Path, default=None, help="Optional JSON session file path.")
    parser.add_argument("--debug", action="store_true", help="Print full debug record as JSON.")
    parser.add_argument("--prompt", action="store_true", help="Print the LLM-ready response prompt.")
    parser.add_argument("--trace", action="store_true", help="Print compact EmoNet trace summary when --emonet is enabled.")
    parser.add_argument("--interactive", action="store_true", help="Run a persistent CLI session so EmoNet neurons stay alive across ticks.")
    parser.add_argument("--llm", action="store_true", help="Use an LLM to compose the final visible response.")
    parser.add_argument("--provider", default="openai_compatible", choices=["openai_compatible", "anthropic", "gemini"])
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--model-name", default="gpt-5.4")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--timeout-sec", type=int, default=45)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--emonet", action="store_true", help="Use the EmoNet trace runtime as the emotion source.")
    parser.add_argument(
        "--allow-rule-emotion-fallback",
        action="store_true",
        help="Development-only: continue with the rule emotion fallback when EmoNet trace inference fails.",
    )
    parser.add_argument(
        "--no-rule-composer",
        action="store_true",
        help="Disable the development rule response composer. Speaking events then require --llm.",
    )
    args = parser.parse_args()

    event_type = "silence_tick" if args.silence else args.event_type
    llm_config = LLMConfig(
        provider=args.provider,
        base_url=_default_base_url(args.provider, args.base_url),
        model_name=_default_model_name(args.provider, args.model_name),
        api_key_env=_default_api_key_env(args.provider, args.api_key_env),
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_sec=args.timeout_sec,
        reasoning_effort=args.reasoning_effort,
    )
    pipeline = RucaPipeline(
        memory_store=MemoryStore(args.memory) if args.memory else MemoryStore.from_items(),
        session_store=SessionStore(args.session) if args.session else None,
        use_llm=args.llm,
        llm_config=llm_config,
        use_emonet=args.emonet,
        fallback_to_rule_emotion=args.allow_rule_emotion_fallback,
        fallback_to_rule_composer=not args.no_rule_composer,
    )
    if args.interactive:
        return _run_interactive_cli(pipeline, args)

    if event_type == "user_message":
        if not args.text.strip():
            parser.error("text is required for user_message; use --event-type no_reply for autonomous no-speech")
        result = pipeline.run_turn(args.text)
    else:
        result = pipeline.run_event(event_type=event_type, elapsed_minutes=args.elapsed_minutes, text=args.text)

    _print_result(result, prompt=args.prompt, debug=args.debug, trace=args.trace)
    return 0


def _run_interactive_cli(pipeline: RucaPipeline, args: argparse.Namespace) -> int:
    print("Ruca CLI interactive. Type text to send a user_message.")
    print("Commands: /typing [minutes], /processing [minutes], /answering [minutes], /idle [minutes], /no-speech [minutes], /quit")
    while True:
        try:
            line = input("> ")
        except EOFError:
            break
        clean = line.strip()
        if not clean:
            continue
        if clean in {"/quit", "/exit"}:
            break
        if clean.startswith("/"):
            event_type, elapsed = _parse_interactive_command(clean)
            result = pipeline.run_event(event_type=event_type, elapsed_minutes=elapsed)
        else:
            result = pipeline.run_turn(clean)
        _print_result(result, prompt=args.prompt, debug=args.debug, trace=args.trace)
    return 0


def _parse_interactive_command(command: str) -> tuple[str, float]:
    parts = command.split()
    name = parts[0].lower()
    elapsed = float(parts[1]) if len(parts) > 1 else 0.0
    if name == "/typing":
        return "typing", elapsed
    if name == "/processing":
        return "processing", elapsed
    if name == "/answering":
        return "answering", elapsed
    if name == "/idle":
        return "silence_tick", elapsed
    if name in {"/no-speech", "/no-reply"}:
        return "no_reply", elapsed
    raise ValueError(f"unsupported interactive command: {name}")


def _default_base_url(provider: str, base_url: str) -> str:
    if provider == "gemini" and base_url == "https://api.openai.com/v1":
        return "https://generativelanguage.googleapis.com/v1beta"
    return base_url


def _default_model_name(provider: str, model_name: str) -> str:
    if provider == "gemini" and model_name == "gpt-5.4":
        return "gemini-2.5-flash"
    return model_name


def _default_api_key_env(provider: str, api_key_env: str) -> str:
    if provider == "gemini" and api_key_env == "OPENAI_API_KEY":
        return "GEMINI_API_KEY"
    if provider == "anthropic" and api_key_env == "OPENAI_API_KEY":
        return "ANTHROPIC_API_KEY"
    return api_key_env


def _print_result(result: object, *, prompt: bool, debug: bool, trace: bool) -> None:
    assistant_text = getattr(result, "assistant_text", "")
    print(assistant_text if assistant_text else "[internal_only]")
    if trace:
        record = getattr(result, "debug_record", {}).get("emonet_trace")
        if record:
            profile = record.get("trace_profile", {})
            print(
                "trace: "
                f"event_kind={record.get('event_kind')} "
                f"tick={profile.get('tick_index')} "
                f"stim_dim={len(record.get('stim_vec', []))} "
                f"dominant_cluster={profile.get('dominant_cluster_id')}"
            )
    if prompt:
        print(getattr(result, "response_prompt", ""))
    if debug:
        print(json.dumps(getattr(result, "debug_record", {}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
