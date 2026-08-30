# Maya-Voice-Os

A free, standalone, plug-and-play voice AI pipeline. Runs on a laptop, no
GPU required, no dependency on any paid service or any other project.

**Created by [Jugal Thakkar](./CREDITS.md).** Attribution is required to use
this project — see [LICENSE](./LICENSE) before you deploy or fork it.

| Service | Tech | Cost |
|---|---|---|
| **asr-service** | `faster-whisper` "small", CPU/int8, multilingual (English + Hindi + other Indic languages) | Free, fully local |
| **llm-service** | Fast-path playbook router (instant, no network) → `homemath`-based multi-provider free LLM routing (Groq → OpenRouter free models → local Ollama) | Free, never paid |
| **tts-service** | `edge-tts` by default (free, keyless); pluggable interface to wire in any other TTS API | Free (or your choice) |
| **orchestration-service** | Runs everything — loads ASR/LLM/TTS and exposes a single `/process` HTTP API | Free |
| **telephony-service** | Provider-agnostic gateway that talks to orchestration-service. Twilio adapter is ready to use out of the box; an Exotel-style adapter is included as a working template for other providers | Free to run (your telephony provider has its own call pricing) |

This is a standalone extraction — it does not call out to, depend on, or
share infrastructure with any larger/private project. Every external
connection this code makes is one you explicitly configure in `.env`.

## Quick start — local mic/speaker demo (no telephony, no servers)

```bash
pip install maya-voice-os
cp .env.sample .env    # download from the repo, or create your own — see .env.sample below
# Add a free Groq or OpenRouter key to .env for LLM fallback beyond playbooks.
maya-voice-os local
```

Talk into your mic, press Enter to end your turn. First run downloads the
faster-whisper "small" model (~250MB) once. No `ffmpeg` install needed —
TTS audio decoding uses `av` (PyAV), which bundles its own codecs inside
the Python package.

**Installing from source instead of PyPI** (for development, or if you want
to edit the code):
```bash
git clone https://github.com/Jugalt-iam/Maya-Voice-Os.git
cd Maya-Voice-Os
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -e .
cp .env.sample .env
maya-voice-os local
```

## Test in a browser, no telephony needed

```bash
maya-voice-os orchestrator
```
Then open **http://localhost:8004** in a browser. This serves a small local
test UI straight from orchestration-service itself — no separate server, no
telephony provider, no phone number needed. Click the mic, allow microphone
access, speak, and Maya replies with real synthesized audio — this
exercises the exact same ASR → fast-path/LLM routing → TTS pipeline a phone
call would use. You can also type instead of speaking, to test the
`text_input` path directly. A Settings panel lets you point it at a
different host/port or set an API token if you've configured
`ORCHESTRATION_API_TOKEN`.

## Running it as services (what telephony connects to)

```bash
maya-voice-os orchestrator    # starts orchestration-service on :8004 (/process, /health)
maya-voice-os telephony       # starts telephony-service on :8100, talks to orchestration-service
```

`telephony-service` never talks to anything except your configured telephony
provider and `ORCHESTRATION_URL` (your own orchestration-service). There is
no other outbound connection in this repo.

## Connecting Twilio (ready to use)

1. Start both servers above (or deploy them somewhere reachable).
2. Expose `telephony-service` publicly — for testing, [ngrok](https://ngrok.com):
   ```bash
   ngrok http 8100
   ```
3. In the Twilio console, set your phone number's **"A call comes in"**
   webhook to:
   ```
   https://<your-public-host>/twilio/twiml
   ```
4. Call the number. That's it — no code changes needed.
5. Optional but recommended for production: set `TWILIO_AUTH_TOKEN` in
   `.env` to enable webhook signature validation.

## Connecting Exotel or another turn-based provider

`telephony-service/adapters/exotel_adapter.py` is a working template for
providers that call a webhook per turn (record → POST → reply) rather than
streaming continuously like Twilio. Point your provider's incoming-call
webhook at:
```
https://<your-public-host>/exotel/incoming/<EXOTEL_WEBHOOK_SECRET>
```
(set `EXOTEL_WEBHOOK_SECRET` in `.env` — it's the only auth an inbound
webhook can practically carry, since most providers can't send custom
bearer tokens). If your provider's callback shape differs from Exotel's
(`SpeechResult` / `RecordingUrl`), adjust `ExotelTurnPayload` and the audio
handling in that file — the call into `orchestration_client` stays the same
regardless of provider.

## Bringing your own telephony provider

Any provider works, because `telephony-service` is provider-agnostic by
design: every adapter's only real job is converting your provider's wire
protocol into a call to `orchestration_client.process_audio()` or
`.process_text()`, and converting the JSON reply back into whatever your
provider expects. Two integration patterns are already implemented as
references:

- **Streaming** (`adapters/twilio_adapter.py`) — continuous bidirectional
  audio over a WebSocket; buffers audio and detects end-of-turn with simple
  silence detection.
- **Turn-based** (`adapters/exotel_adapter.py`) — one webhook call per
  utterance.

Add a new file in `adapters/`, mount its router in `telephony-service/server.py`,
and you're integrated.

### A note on the internal SIP server

`telephony-service/sip_reference/internal_sip_server.py` is included as a
**non-functional reference sample only** — it's not wired into the app and
won't work as-is. Building a real SIP stack (RTP media, NAT traversal,
security) from scratch is its own substantial project. If you want to
receive calls without a hosted provider like Twilio/Exotel, get a proper
SIP trunk from a telephony/SIP provider and pair it with a mature library
or PBX (PJSIP, aiosip, Asterisk, FreeSWITCH) — see
`sip_reference/README.md`.

## Making it your own bot: identity files

Everything about the bot's persona lives in `identity/*.yaml`. Copy
`identity/default.yaml`, edit `name`, `system_prompt`, `voice`, `greeting`,
and (optionally) restrict which `playbooks/` it uses, then point
`IDENTITY_FILE` in `.env` at your new file. No code changes needed.

## Free LLM routing (how the "brain" decides)

Every turn tries, in order, until one succeeds:

1. **Fast-path playbooks** (`playbooks/*.yaml`) — instant, zero network calls.
2. **`homemath`-routed LLM call** (`llm-service/llm_router.py`) — tries each
   provider in `LLM_PROVIDER_ORDER` (default `groq,cerebras,mistral,openrouter,ollama`),
   skipping any with no key set, falling through automatically on
   error/timeout/empty response. Local Ollama is always last, so the bot
   never hard-blocks even with zero cloud keys configured, as long as
   Ollama is running.

## The `/process` API contract

`orchestration-service`'s `POST /process` accepts either `audio_data`
(base64 PCM16 mono 16kHz) or `text_input`, plus `conversation_id`. It
returns `transcript`, `llm_response`, `audio_base64` (PCM16 mono 16kHz),
`processing_stages`, `total_processing_time`, and placeholder fields
(`expert_used`, `confidence`, `reasoning_chain`) kept for shape-compatibility
with richer orchestration setups, even though this standalone version
doesn't populate them. Reply audio is returned inline as base64 rather than
a hosted URL, to avoid the extra attack surface of serving files.

## Security notes

- Set `ORCHESTRATION_API_TOKEN` if `orchestration-service` is reachable from
  anywhere other than `telephony-service` on the same machine/network.
- The Twilio adapter validates webhook signatures when `TWILIO_AUTH_TOKEN`
  is set (plain HMAC-SHA1 per Twilio's documented scheme — no SDK needed).
- The Exotel-style adapter is protected by an unguessable secret in the
  webhook path (`EXOTEL_WEBHOOK_SECRET`), since inbound provider webhooks
  generally can't carry custom auth headers.
- `orchestration-service` caps decoded audio payloads at 15MB to avoid
  trivial memory-exhaustion abuse.
- No `eval`/`exec`/`pickle` anywhere; all YAML is loaded with `yaml.safe_load`.
- `orchestration_client.py` makes a single bounded-timeout request per call —
  no retry loops, no risk of telephony-service hammering orchestration-service.
- Every external URL (`ORCHESTRATION_URL`, `OLLAMA_HOST`, etc.) is
  `.env`-configured with a localhost default; nothing points at any other
  project's infrastructure.

## Project layout

```
Maya-Voice-Os/
├── pyproject.toml               # pip/PyPI packaging — `pip install maya-voice-os`
├── maya_voice_os/                # everything below is inside this one importable package
│   ├── cli.py                     # `maya-voice-os local/orchestrator/telephony` entry point
│   ├── run_local.py                # mic/speaker demo
│   ├── identity/                    # drop-in persona files
│   ├── playbooks/                    # fast-path YAML playbooks
│   ├── asr_service/
│   │   └── engine.py                  # faster-whisper wrapper
│   ├── llm_service/
│   │   ├── fast_router.py              # instant playbook matching
│   │   ├── llm_router.py                # multi-provider free LLM routing (homemath)
│   │   └── identity_loader.py            # loads identity/*.yaml
│   ├── tts_service/
│   │   └── engine.py                    # Edge TTS + pluggable provider interface
│   ├── orchestration_service/
│   │   ├── pipeline.py                    # wires asr/llm/tts together in-process
│   │   ├── server.py                       # FastAPI app: /process, /identity, /greeting, /health
│   │   └── ui/
│   │       └── index.html                   # local browser test UI (no telephony needed)
│   ├── telephony_service/
│   │   ├── orchestration_client.py          # HTTP client -> orchestration-service
│   │   ├── session_manager.py                # in-memory per-call session state
│   │   ├── adapters/
│   │   │   ├── twilio_adapter.py               # ready to use
│   │   │   └── exotel_adapter.py                # turn-based template
│   │   └── server.py                             # mounts adapters, FastAPI app
│   └── shared/
│       ├── audio_utils.py               # resampling, mu-law<->PCM conversion
│       └── retry.py                      # exponential-backoff retry for HTTP calls
└── .env.sample
```

Note: `telephony_service/sip_reference/` (the non-functional SIP sample) lives
in the GitHub source tree but isn't bundled into the pip package — it was
never meant to run, only to read.
```

## Attribution (required)

This project is licensed under a modified MIT license with a **mandatory
attribution clause** — see [LICENSE](./LICENSE) for the full text. In short:
if you use, deploy, or build on Maya-Voice-Os, you must keep

> "Built on Maya-Voice-Os by Jugal Thakkar"

visible somewhere in your project — your README, an about/credits screen,
or source comments — regardless of how much of the code you change. This
isn't optional; removing it voids the permissions the license grants.

The license also makes clear that this attribution is a factual credit
only, not an endorsement — no one may imply Jugal Thakkar sponsors,
certifies, or is affiliated with a derivative product without separate
written consent — and that anyone deploying this software (or a
derivative) is solely responsible for their own legal/regulatory
compliance and indemnifies the author against claims arising from their
use. See LICENSE Sections 3 and 4 for the full terms. (Not legal advice —
see the note at the end of LICENSE.)

## ⚠️ Playbooks contain sample data — replace before real use

Everything in `playbooks/*.yaml` — pricing, loan/EMI figures, appointment
timing and availability, conversion-rate statistics, office locations, and
similar specifics — is **illustrative sample data, not real information**.
None of it has been verified as accurate for any actual business. Before
deploying this for a real caller to hear, you must replace every such
figure with your own accurate, current, authorized values. Maya's system
prompt (`identity/default.yaml`) already instructs her not to state sample
figures as confirmed facts, but that's a safety net, not a substitute for
actually updating the data — see LICENSE Section 5 for how responsibility
for this is allocated.

## Testing latency

For a **text-only round trip** (isolates LLM + TTS, skips ASR):
```bash
curl -s -X POST http://localhost:8004/process \
  -H "Content-Type: application/json" \
  -d '{"conversation_id":"latency-test","text_input":"hello there"}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['processing_stages'])"
```

For **full pipeline latency including ASR**, send real audio through the
browser test UI (`http://localhost:8004`) — every reply there already
shows a `round-trip Xms` tag. For the per-stage breakdown (which part is
actually slow: ASR, LLM, or TTS), any `/process` call with `audio_data`
now returns it directly:
```json
"processing_stages": {
  "asr": {"duration_ms": 640.2},
  "respond": {"duration_ms": 210.5},
  "tts": {"duration_ms": 480.1},
  "total": {"duration_ms": 1330.8}
}
```

## Known limitations
- Real-time streaming (`/process/stream`) bypasses `homemath`'s task
  classification and `<think>`-stripping, since `homemath` itself doesn't
  expose token-level streaming to callers — see the section above for the
  full trade-off.
- `smart_memory`/`context_manager` are best-effort; Redis is recommended
  for durability but not required for the app to function.

- The ASR model (`faster-whisper` "small") downloads ~250MB from Hugging Face
  Hub on first run only, then caches locally (`~/.cache/huggingface`) and
  runs fully offline afterward. If you're deploying somewhere with
  restricted network egress, make sure `huggingface.co` is reachable for
  that first run (or pre-download the model and copy the cache over).

- Edge TTS and the cloud LLM providers (Groq/OpenRouter) require internet;
  only the ASR model and Ollama are fully offline.
- Twilio adapter's turn-taking uses simple energy-based silence detection —
  tune `SILENCE_RMS_THRESHOLD` / `SILENCE_FRAMES_TO_END_TURN` in
  `adapters/twilio_adapter.py` for your actual call audio quality.
- `homemath`'s token counting is an approximation (`len(text)//4`, not a
  real tokenizer), least accurate for non-English text — worth knowing
  given the Indic-language use case.
- `exotel_adapter.py`'s `RecordingUrl` path is intentionally left as a
  template (audio format varies per provider) — fill in
  `_extract_audio_from_request`-equivalent logic for your specific provider.
