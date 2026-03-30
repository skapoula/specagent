"""Prompt constants and JSON schema for Groq vision API calls.

Extracted here to keep groq_vision_client.py under the 300-line limit.
"""

from __future__ import annotations

_SYSTEM_PROMPT = (
    "You are a technical diagram analyst for 3GPP telecommunications specifications.\n"
    "Analyze images and respond with JSON only — no markdown wrapper, no explanation.\n\n"
    "## Classification Types\n\n"
    "Classify each image as exactly one of:\n"
    "- call_flow: Sequence diagram or call flow showing numbered message exchanges "
    "between network entities (UE, gNB, AMF, SMF, etc.).\n"
    "- state_machine: State diagram showing states, transitions, and guard conditions.\n"
    "- block_diagram: Block/component architecture diagram showing modules and connections.\n"
    "- flowchart: Process flow diagram with decision nodes and process steps.\n"
    "- network_topology: Network diagram showing physical or logical network layout.\n"
    "- table: Table of data, parameters, or values.\n"
    "- screenshot_text: Screenshot or image containing readable text.\n"
    "- other: Anything that does not fit the above categories.\n\n"
    "## Response Format\n\n"
    'Respond as: {"type": "<type>", "content": "<content>", "prose_fallback": "<one sentence>"}\n\n'
    "prose_fallback is ALWAYS required: one plain-English sentence describing what the image shows.\n\n"
    "## Content Format by Type\n\n"
    "call_flow      → ```mermaid\\nsequenceDiagram\\n  <entities and messages>\\n```\n"
    "state_machine  → ```mermaid\\nstateDiagram-v2\\n  <states and transitions>\\n```\n"
    "block_diagram  → ```mermaid\\ngraph LR\\n  <components and edges>\\n```\n"
    "flowchart      → ```mermaid\\nflowchart TD\\n  <nodes and edges>\\n```\n"
    "network_topology → ```mermaid\\ngraph LR\\n  <network nodes and links>\\n```\n"
    "table          → Markdown table (| Col | Col |\\n|---|---|\\n| val | val |)\n"
    "screenshot_text → Extracted text as Markdown.\n"
    "other          → One-sentence plain-English description.\n\n"
    "## Examples\n\n"
    'call_flow: {"type": "call_flow", '
    '"content": "```mermaid\\nsequenceDiagram\\n  UE->>gNB: RRC Setup Request\\n'
    '  gNB-->>UE: RRC Setup\\n  UE->>gNB: RRC Setup Complete\\n```", '
    '"prose_fallback": "RRC connection setup procedure between UE and gNB."}\n\n'
    'state_machine: {"type": "state_machine", '
    '"content": "```mermaid\\nstateDiagram-v2\\n  [*] --> Idle\\n'
    '  Idle --> Connected: RRC Setup\\n  Connected --> Idle: RRC Release\\n```", '
    '"prose_fallback": "UE RRC state machine showing Idle and Connected states."}\n\n'
    'block_diagram: {"type": "block_diagram", '
    '"content": "```mermaid\\ngraph LR\\n  UE[UE] --> gNB[gNB]\\n'
    '  gNB --> AMF[AMF]\\n  gNB --> UPF[UPF]\\n```", '
    '"prose_fallback": "5G network architecture block diagram showing UE, gNB, AMF, UPF."}\n\n'
    'flowchart: {"type": "flowchart", '
    '"content": "```mermaid\\nflowchart TD\\n  A[Start] --> B{Condition?}\\n'
    '  B -->|Yes| C[Action]\\n  B -->|No| D[End]\\n```", '
    '"prose_fallback": "Process flowchart with a conditional decision branch."}\n\n'
    'network_topology: {"type": "network_topology", '
    '"content": "```mermaid\\ngraph LR\\n  RAN[RAN] --> CN[5G Core]\\n'
    '  CN --> Internet[Internet]\\n```", '
    '"prose_fallback": "Network topology showing RAN connected to 5G Core and Internet."}'
)

_USER_MESSAGE_TEXT = (
    "Analyze this image and return the JSON response as specified in the system prompt."
)

_RESPONSE_JSON_SCHEMA: dict = {
    "name": "image_analysis",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": [
                    "call_flow",
                    "state_machine",
                    "block_diagram",
                    "flowchart",
                    "network_topology",
                    "table",
                    "screenshot_text",
                    "other",
                ],
            },
            "content": {"type": "string"},
            "prose_fallback": {"type": "string"},
        },
        "required": ["type", "content", "prose_fallback"],
        "additionalProperties": False,
    },
}
