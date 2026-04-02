# Workflows

This directory contains Markdown SOPs (Standard Operating Procedures) that define how tasks should be executed within the WAT framework.

## Structure of a Workflow

Each workflow file should include:

- **Objective** — What this workflow accomplishes
- **Required Inputs** — What the agent needs before starting
- **Tools Used** — Which scripts in `tools/` are called
- **Steps** — The sequence of actions
- **Expected Outputs** — What success looks like
- **Edge Cases** — Known failure modes and how to handle them

## Naming Convention

Use descriptive snake_case filenames, e.g.:
- `scrape_website.md`
- `export_to_sheets.md`
- `process_csv_data.md`
