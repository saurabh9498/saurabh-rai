# Field Guide — Style

One palette and one set of diagram rules for every leaf. Defined once here; never varied per diagram.

## Colour by domain

Each domain has a stroke colour (borders, headings, arrows) and a fill colour (box backgrounds). A reader should be able to learn that blue always means GPU compute, green always means the fabric, and so on.

| Domain | What it covers | Stroke | Fill |
|---|---|---|---|
| Workload | Traffic, requests, prompts, users, SLAs | `#B45309` amber | `#FEF3C7` |
| Control & software | Schedulers, routers, planners, serving software, APIs | `#6D28D9` purple | `#EDE9FE` |
| Compute | GPUs, CPUs, workers, engines | `#1D4ED8` blue | `#DBEAFE` |
| Memory & storage | HBM, KV cache, DRAM, NVMe, file systems, object stores | `#0F766E` teal | `#CCFBF1` |
| Fabric | NVLink, InfiniBand, RoCE, Ethernet, data transfer between devices | `#15803D` green | `#DCFCE7` |
| Observability | Metrics, logs, traces, dashboards | `#475569` slate | `#F1F5F9` |
| Risk & failure | Failure modes, warnings, what breaks | `#B91C1C` red | `#FEE2E2` |

Text: `#111827` primary, `#374151` secondary. Background: `#FFFFFF`, set explicitly in every SVG so diagrams stay readable in GitHub's dark mode.

## Diagram rules

- **One complete mental model per diagram**, not one fact.
- **Top to bottom along the data flow.** The eye follows the path the data takes.
- **Every box carries 3–6 capability bullets**, not just a label.
- **Comparison panel** wherever an alternative exists.
- **Benefits strip and legend** along the bottom.
- **Numbers on the diagram carry their source** in the legend strip.

## Format

- Hand-authored **SVG**, committed as source next to the leaf in a `diagrams/` folder.
- Font: `Helvetica, Arial, sans-serif`. Title 26px, box headings 16–17px, bullets 13–14px. Nothing below 12px.
- Must read at two sizes: as a thumbnail (structure visible) and at full zoom (every bullet legible). Render and check both before publishing.
- Never screenshots of other people's diagrams.

## Prose

- Every leaf follows the ten-section structure in the outline: problem, first principles, mechanism, numbers, worked example, trade-offs, production failures, interview version, diagram, further reading.
- Expand every acronym on first use in each leaf.
- Every number has a source and a date. Vendor numbers are labelled as vendor numbers. Simulation numbers are labelled as simulation numbers.
- Anchor-project examples are anonymised: no client names, revenue or identifying figures.
