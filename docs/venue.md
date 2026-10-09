# Venue watch

| date checked | ICML 2027 | HiLD |
|---|---|---|
| 2026-10-09 | No call for workshops, dates or city posted; the official Future Meetings page lists only the region ("South America"). Prior cycles: workshop list announced early April, suggested paper deadline late April, notification mid-May, workshops the two days after the main conference (ICML 2026: 10–11 July). | No 2027 page. HiLD ran at ICML 2023–2026 (2026 theme "Science of Scaling"). |

Intermediate milestone: ICLR 2027 workshops (San Francisco, 29–30 April 2027), suggested paper deadline 1 Feb 2027.
Policy items to re-check when the CFP appears: page limit, archival status, LLM-use disclosure, dual-submission wording.

## Formatting measurement (2026-10-09, backlog N5)

`paper/main_icml.tex` builds the same body (`paper/body.tex`, shared with the preprint `paper/main.tex`) with the official `icml2026.sty`
(downloaded from media.icml.cc; the 2027 style is not out) in anonymous two-column mode; appendix in one column. Result: **main text
6.2 pages** (references begin on page 7; appendix pp. 8–14). Fits an 8-page main-body limit (ICML main-track rule) with room; a 4-page
workshop limit (common for HiLD-type workshops) would require cutting ≈2.2 pages, candidates: the transformer section to a half page,
Prop. 3 to a table, E5 controls to the appendix, related work compressed. Figures are already sized for a column. The ICML style requires
the author block via `\icmlauthor`; the draft is anonymous. Both PDFs compile without errors (one 33 pt overfull equation in two-column mode).
