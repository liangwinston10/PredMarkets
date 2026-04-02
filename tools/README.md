# Tools

This directory contains Python scripts that perform deterministic execution tasks within the WAT framework.

## Principles

- Each script does one thing well
- Credentials are read from `.env` via `python-dotenv`
- Scripts are callable from the command line with clear arguments
- Errors are surfaced clearly so the agent can recover

## Naming Convention

Use descriptive snake_case filenames, e.g.:
- `scrape_single_site.py`
- `export_to_sheets.py`
- `send_slack_message.py`

## Dependencies

Install dependencies with:
```bash
pip install -r requirements.txt
```
