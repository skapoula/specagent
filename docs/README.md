# SpecAgent — Documentation

SpecAgent answers natural-language questions about 3GPP telecommunications specifications and returns answers with traceable citations to authoritative spec sections.

## Guides

| Document | Audience | Description |
|---|---|---|
| [User Guide](./user-guide.md) | End users | How to query and index specifications |
| [Use Cases](./use-cases.md) | End users | Real examples: question → cited answer |
| [Installation](./installation.md) | Everyone | How to install and run SpecAgent |
| [API Reference](./api-reference.md) | Developers | REST endpoints, CLI commands, and schemas |
| [Contributing](./contributing.md) | Developers | Local setup, testing, and contribution workflow |

## Architecture Diagrams

| Diagram | Description |
|---|---|
| [C4 Context](./diagrams/c4-context.svg) | System-level view: users, SpecAgent, and external systems |
| [C4 Container](./diagrams/c4-container.svg) | Runtime containers: API, agent pipeline, vector store, embedder |
| [C4 Component](./diagrams/c4-component.svg) | LangGraph pipeline: router, retriever, grader, rewriter, generator, hallucination checker |
