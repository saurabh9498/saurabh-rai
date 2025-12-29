# Images Directory

This directory contains images and diagrams for documentation.

## Generating Architecture Diagrams

The architecture diagrams can be generated using:

1. **Mermaid**: Use the Mermaid syntax in `docs/architecture.md`
2. **Draw.io**: Import the architecture and export as PNG
3. **Excalidraw**: Create custom diagrams

## Required Images

- `architecture-banner.png` - Main architecture overview diagram (1200x600px recommended)

## Placeholder

Until actual diagrams are created, you can use placeholder services or generate from Mermaid:

```bash
# Using Mermaid CLI
npx @mermaid-js/mermaid-cli@latest -i architecture.mmd -o architecture-banner.png
```
