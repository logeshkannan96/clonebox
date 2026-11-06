# Security

## Reporting a vulnerability

Report vulnerabilities privately through [GitHub Security Advisories](https://github.com/logeshkannan96/clonebox/security/advisories/new). Don't open a public issue.

## Threat model

Clonebox is a **single-user, local** app.

- **There is no authentication or authorization.** Anyone who can reach the port can list, play and delete voices and generate speech with them.
- **Voice recordings are biometric data.** They are stored unencrypted under `CLONEBOX_DATA`.

If you expose Clonebox beyond `127.0.0.1`, put it behind an authenticating reverse proxy with TLS, and restrict access to `CLONEBOX_DATA` at the filesystem level.

## Input handling

- Uploads are capped at 25 MB and decoded by `ffmpeg` with a 60 s timeout.
- Voice and clip IDs are validated against `^[0-9a-f]{32}$`, which prevents path traversal.
- Request bodies are validated with Pydantic, which sets length and range limits on text, seed, temperature and gaps.
