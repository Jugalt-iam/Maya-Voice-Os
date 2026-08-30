# SIP reference (not functional)

`internal_sip_server.py` here is a **non-functional reference sample**, not
something you can run. It shows the shape of raw SIP message handling
(INVITE/ACK/BYE) but has no audio/RTP handling, no SDP negotiation, and
missing pieces like a logger.

Building your own SIP stack from scratch is a significant undertaking
(NAT traversal, RTP media, security). For a real "no third-party voice
API" deployment, get a proper SIP trunk from your telephony/SIP provider
and pair it with a mature library or PBX (PJSIP, aiosip, Asterisk,
FreeSWITCH) rather than extending this file.

For anything else — Twilio, Exotel, or another provider with a webhook/API
— use `adapters/twilio_adapter.py` (ready to use) or
`adapters/exotel_adapter.py` (turn-based template) instead.
