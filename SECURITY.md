# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in Nexus, please report it responsibly.

**Email:** Email **zantryis@gmail.com** directly.

If private GitHub Security Advisories are enabled for this repository later, prefer reporting there instead of opening a public issue.

Please include:
- Description of the vulnerability
- Steps to reproduce
- Affected versions (or commit hash)
- Any potential impact assessment

You should receive an acknowledgement within 48 hours. Please do not open a public issue for security vulnerabilities.

## Scope

The following are in scope:
- Authentication and authorization bypasses (admin token, setup/settings access)
- Injection vulnerabilities (XSS, SQL injection, command injection)
- Secrets exposure (API keys, tokens leaking via logs, responses, or git)
- Path traversal or file access issues
- Denial of service via crafted input

The following are out of scope:
- Vulnerabilities in upstream dependencies (report those to the respective projects)
- Issues requiring physical access to the host machine
- Social engineering

## Security Design

- The dashboard binds to `127.0.0.1` by default; `--host 0.0.0.0` requires explicit opt-in
- `/setup` and `/settings` routes are limited to same-machine access unless `NEXUS_ADMIN_TOKEN` is set
- Docker Compose enables a narrow bridge-gateway exception so same-machine `localhost` setup still works from the host browser
- Config and `.env` files are written with `0o600` permissions
- All user-generated content is sanitized via Jinja2 auto-escaping and `bleach`
- CSP, X-Frame-Options, and other security headers are set by middleware

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes      |
