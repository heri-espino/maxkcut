# Paper-specific instructions for agents

1. Read repository AGENTS.md, RESEARCH_HANDOFF.md, this directory's README.md and HANDOFF.md, the E025 aggregate analysis, E027 release index and the relevant numerical source before editing.
2. The seven-method list is *descriptive*: highest equal-graph mean final best-of-16 gain over refined GW from E025+E027. Preserve selected names and their separate audit statuses.
3. Keep each algorithm in its own chapter and describe actual code. Historical Sweep's gain-labelled tangent selector uses negative one-flip gains; Angular Gain uses positive gain divided by angular margin. Never silently harmonize the conflicting mechanisms.
4. Keep raw rounding distinct from the common Flip/KL/ILS final best-ever endpoint. Explain fixed-arc/fixed-normal exactness without global claims.
5. Sources and all TeX figures are immutable except for reviewable corrections. Run verify_results.py and the hosted PDF compiler after edits. Never rerun EVA, SDP, numerical rounders or refinement merely to compile paper.
6. Preserve all manuscript source, figures, bibliography, verification code and experiment records. Work on the paper branch; respect other agents and PRs, and never merge without authorization.
7. User has expressly rejected PowerPoint. This manuscript builds **PDF only**.
