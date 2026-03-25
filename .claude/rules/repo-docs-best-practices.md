# Repository Documentation Best Practices
> **Audience:** Claude Code — Planning & Generation Mode  
> **Purpose:** Apply this document whenever generating, updating, or auditing documentation for any project. This covers the full documentation surface of a repo: end-user guides, developer references, and the `/docs` folder structure itself.  
> **Source of truth:** Codebase and source files only. Never document features that cannot be confirmed in code.

---

## Repo Documentation Structure

When generating documentation for a project, always create the following folder and file layout. Do not deviate from this structure without explicit instruction.

```
/docs
├── README.md                  ← Docs index (links to all documents below)
├── user-guide.md              ← Non-technical end users
├── use-cases.md               ← Example-based use cases (input → output)
├── installation.md            ← Quick install guide (all audiences)
├── api-reference.md           ← Developers consuming the API
└── contributing.md            ← Developers setting up and contributing
```

The root `README.md` of the project should link to `/docs/README.md` under a **Documentation** section. It should not duplicate content from any doc file.

---

## `/docs/README.md` — The Docs Index

This file is the navigation hub. It must be generated first, then updated as other files are added.

**Rules:**
- One sentence describing what the project does
- A table listing every doc file, its audience, and a one-line description
- No other content — no feature lists, no getting started steps, no duplicated prose

**Template:**

```markdown
# [Project Name] — Documentation

[One sentence: what the project does and who it's for.]

## Guides

| Document | Audience | Description |
|---|---|---|
| [User Guide](./user-guide.md) | End users | How to use the application |
| [Use Cases](./use-cases.md) | End users | Real examples: input → output |
| [Installation](./installation.md) | Everyone | How to install and run the project |
| [API Reference](./api-reference.md) | Developers | Endpoints, parameters, and responses |
| [Contributing](./contributing.md) | Developers | Setup, workflow, and contribution guide |
```

---

## `user-guide.md` — End User Guide

**Audience:** Non-technical end users  
**Tone:** Plain English, 8th-grade reading level, imperative verbs, no jargon

### Document Structure

```
# [Project Name]
One sentence: what it does and who it's for.

## How It Works
Mermaid flowchart (user perspective only) + one caption sentence.

## Getting Started
Link to installation.md. Do not duplicate install steps here.

## How to Use It
Core workflows only. Each workflow = ### subsection with numbered steps.
Maximum 5 workflows, 6 steps each.

## Common Issues
Symptom → fix table. Maximum 5 entries.

## Getting Help
One or two lines: where to go.
```

### Length Budget

| Section | Maximum |
|---|---|
| Intro | 2 sentences |
| How It Works diagram | 1 diagram + 1 caption sentence |
| Diagram node labels | 4 words each |
| Each workflow | 6 numbered steps |
| Total workflows | 5 |
| Common Issues | 5 entries |
| Entire document (excluding diagram) | 600 words |

### Architecture Diagram Rules

Use `flowchart LR`. Show input → transform → output from the user's perspective. Never show internal class names, DB tables, service names, or infrastructure. Every node label must be plain English, 4 words max.

**Good node:** `You upload a file`  
**Bad node:** `FileUploadController → S3Bucket`

Always add a one-sentence plain-English caption beneath the diagram.

### Anti-Bloat Rules

- No throat-clearing: cut any sentence beginning with "This guide will...", "Welcome to...", "In this section..."
- No feature lists: every statement describes an action the user takes
- No passive voice: "Click Upload" not "The upload button should be clicked"
- No over-explained UI: don't describe what a button looks like
- Never use: "simply", "easily", "just", "straightforward", "intuitive"
- No changelog, roadmap, or version history

---

## `use-cases.md` — Use Cases

**Audience:** Non-technical end users  
**Format:** Example-based — always show a concrete input and the resulting output  
**Count:** Exactly 2 use cases per guide generation. Choose the two most common real-world scenarios inferable from the codebase.

### Structure per Use Case

```markdown
## Use Case [N]: [Short title — what the user achieves]

**The situation:** One sentence describing who the user is and what they need to do.

**Input:** What the user provides (be specific — file type, data, action taken).

**What happens:** One or two sentences on what the app does. Plain English, no internals.

**Output:** What the user receives (be specific — file name, format, content summary).

---
### Try It

1. [Step 1]
2. [Step 2]
3. [Step 3 — user sees the output]
```

### Rules

- The input and output must be concrete and specific — not abstract ("a file" → "a CSV file named results.csv with 3 columns: Name, Score, Date")
- Maximum 3 steps in the Try It section
- Do not describe error states in use cases — those belong in Common Issues
- The two use cases must be meaningfully different — not two variations of the same workflow
- Base use cases only on user-facing entry points found in the codebase — never fabricate scenarios

### Example — Good

```markdown
## Use Case 1: Checking your monthly report

**The situation:** You want to see how your team performed last month.

**Input:** You select "November 2024" from the date picker and click Generate.

**Output:** A PDF file named `team-report-nov-2024.pdf` downloads automatically,
containing a summary table and three charts.

---
### Try It

1. Open the app and go to **Reports**
2. Select the month and year from the dropdown
3. Click **Generate** — your report downloads within 10 seconds
```

### Example — Bad

```markdown
## Use Case 1: Using the reporting feature

**Input:** Some data  
**Output:** A report is generated by the system using the ReportGeneratorService
which queries the PostgreSQL database and formats the output via PDFKit.
```

---

## `installation.md` — Quick Installation Guide

**Audience:** Everyone (end users and developers)  
**Goal:** Get the project running in the shortest possible path. One page, no narrative.

### Structure

```
# Installation

## Requirements
Bullet list of prerequisites only. Name + minimum version. Nothing else.

## Install

### Option A: [Simplest method — e.g., Docker, installer, pip]
Numbered steps. Maximum 5.

### Option B: [Alternative — e.g., from source]
Numbered steps. Maximum 8.

## Verify It's Working
One command or one action that confirms success.
Expected output shown as a code block.

## Uninstall
Two steps maximum.
```

### Rules

- Lead with the simplest installation method (Docker, binary installer, pip install) — not from-source
- Every command goes in a fenced code block with the correct language tag
- Never explain why a command works — just show it
- No "optional" steps embedded in the main flow — put them in a collapsible block or omit
- Prerequisite versions must be confirmed from `package.json`, `requirements.txt`, `pyproject.toml`, `go.mod`, or equivalent — never assumed
- The verification step must produce visible output the user can compare against

### Length Budget

| Section | Maximum |
|---|---|
| Requirements list | 5 items |
| Each install option | 8 steps |
| Verify step | 1 command + expected output |
| Entire document | 400 words |

---

## `api-reference.md` — API Reference

**Audience:** Developers integrating with or consuming the project's API  
**Source:** Extract exclusively from route definitions, controller files, schema files, and existing OpenAPI/Swagger specs found in the codebase

### Structure

```
# API Reference

## Base URL
## Authentication
## Endpoints
  ### [METHOD] /path
## Error Codes
## Rate Limits (if applicable)
```

### Per-Endpoint Template

```markdown
### GET /example

**Description:** One sentence — what this endpoint does.

**Authentication:** Required / None

**Parameters:**

| Name | Type | Required | Description |
|---|---|---|---|
| `param` | string | Yes | What it is |

**Request Example:**
\```http
GET /example?param=value
Authorization: Bearer <token>
\```

**Response Example:**
\```json
{
  "field": "value"
}
\```

**Response Fields:**

| Field | Type | Description |
|---|---|---|
| `field` | string | What it contains |

**Error Responses:**

| Code | Meaning |
|---|---|
| 400 | Missing required parameter |
| 401 | Invalid or missing token |
```

### Rules

- Document only endpoints confirmed in route files — never infer undocumented endpoints
- Every endpoint must have a request and response example with realistic (not placeholder) values
- Authentication method must be confirmed from middleware or auth configuration in the codebase
- If a field is optional, mark it clearly in the Parameters table
- Do not document internal or admin-only endpoints in a public-facing API reference unless explicitly instructed

---

## `contributing.md` — Developer Setup & Contributing Guide

**Audience:** Developers setting up a local environment and contributing to the project  
**Tone:** Direct, technical, assumes programming competence — no hand-holding on general concepts

### Structure

```
# Contributing to [Project Name]

## Prerequisites
## Local Setup
## Running Tests
## Project Structure
## Making Changes
## Submitting a Pull Request
## Code Standards
```

### Rules

- Prerequisites must be confirmed from lockfiles, config files, and CI configuration — not assumed
- Local setup steps must reproduce a working dev environment exactly — test the steps against the codebase before finalizing
- Project structure section: show the directory tree (2 levels deep max) with one-line annotations — no prose
- Code standards: extract from `.eslintrc`, `.prettierrc`, `pyproject.toml`, `rubocop.yml`, or equivalent — never invent standards
- PR process: extract from existing `CONTRIBUTING.md`, PR templates, or CI workflow files if present; otherwise use a minimal 4-step process (fork → branch → PR → review)
- Do not include a code of conduct, roadmap, or acknowledgements section — these belong in the root `README.md` if needed

### Length Budget

| Section | Maximum |
|---|---|
| Prerequisites | 6 items |
| Local setup | 10 steps |
| Project structure tree | 2 levels deep |
| Code standards | 8 rules |
| Entire document | 800 words |

---

## Codebase Analysis — What to Extract Per Doc

| Document | Look For | Ignore |
|---|---|---|
| user-guide | Entry points, user-facing routes, UI error strings, user-controlled config | Internal helpers, DB schema, test files |
| use-cases | User-facing entry points, input types accepted, output types returned | Internal processing logic, third-party internals |
| installation | `package.json` scripts, `requirements.txt`, `Dockerfile`, `README` install sections, CI setup steps | Dev-only tooling unless building from source |
| api-reference | Route definitions, controllers, request/response schemas, auth middleware, OpenAPI specs | Internal services, DB queries, helper functions |
| contributing | Lockfiles, CI config, linting config, test runner config, PR templates | Production infrastructure, deployment scripts |

---

## Global Anti-Patterns — Apply Across All Docs

These apply to every document in `/docs`:

- **No duplication across files.** If installation steps appear in `user-guide.md`, replace them with a link to `installation.md`. One fact lives in one place.
- **No undocumented assumptions.** Every command, path, and config value must be confirmable in the source files.
- **No future tense for current features.** "The app will support..." implies it doesn't work yet. Use present tense for implemented features.
- **No invented examples.** Request/response examples, file names, and field values must reflect what the code actually produces.
- **No nested bullet points beyond 2 levels.** Restructure as subsections instead.

---

## Self-Audit Checklist — Run Before Finalizing Any Doc

### All Documents
- [ ] No content duplicated from another doc in `/docs` — cross-linked instead
- [ ] All commands are in fenced code blocks with correct language tags
- [ ] No future tense used for existing features
- [ ] No invented values — all examples confirmed from codebase
- [ ] `/docs/README.md` index updated to include this document

### user-guide.md
- [ ] Intro is 2 sentences max
- [ ] Mermaid diagram present with plain-English labels (4 words max each)
- [ ] One-sentence caption beneath diagram
- [ ] No internal identifiers in diagram nodes
- [ ] "simply", "easily", "just" do not appear
- [ ] Total word count under 600

### use-cases.md
- [ ] Exactly 2 use cases
- [ ] Each has a specific input (file type, format, action) and specific output (file name, format, content)
- [ ] Try It section has 3 steps or fewer
- [ ] No internal implementation details mentioned
- [ ] Use cases cover meaningfully different scenarios

### installation.md
- [ ] Simplest method listed first
- [ ] All prerequisite versions confirmed from lockfiles or config
- [ ] Verification step included with expected output
- [ ] Total word count under 400

### api-reference.md
- [ ] Only documents endpoints confirmed in route files
- [ ] Every endpoint has request + response example with realistic values
- [ ] Auth method confirmed from codebase
- [ ] Error codes table present

### contributing.md
- [ ] Local setup steps reproduce a working environment
- [ ] Code standards extracted from config files, not invented
- [ ] Project structure tree is 2 levels deep max
- [ ] Total word count under 800
