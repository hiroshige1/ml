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

## Four-page variant (2026-10-09, backlog N14)

`paper/main_icml_4p.tex` (body `paper/body_4p.tex`, moved figures in `paper/appendix_figs_4p.tex`) follows the cut plan of review 4: main text
ends on page 4 (references start on page 5), 16 pages with appendix, 0 errors. Cuts are deletions and moves only (no quantitative statement
changed); both transformer kill statements, "we make no transfer claim" and "P22a failed" survive. Passages cut that are not yet in the appendix
(for the author to move if this version is used): intro contributions block; MC-drift sentence; τN and d=32 flow numbers; E1 details (d-scan,
learned Γ vs Γ*, B=8 control, OLS note); E6 mechanism sentence and m*=0.30; all of Sec. 4 (composition) and E4; transformer norm-collapse numbers;
Zhang/He/Bietti-2022 sentences; the Limitations clauses on rare skills and failed N–B predictions; the disclosure paragraph. The full version
(`main_icml.tex`, 6.3 pp main text) remains the primary one.


Refreshed 2026-10-10 from v0.23 (`paper/body_4p.tex` regenerated): main text ends on page 4 with no slack; additionally cut relative to the v0.16
variant: the LayerNorm and Malladi sentences, the Nishikawa and Bietti-2022 related-work sentences, the limitations list. The author should restore
the Nishikawa sentence if space allows (it is part of the novelty positioning).
