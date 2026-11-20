# Transcriberrr GUI review

Open [the interactive mockup](index.html). The **Prototype state** selector below the window loads an empty queue, active run, partial failure, or completed recording. Recordings, downloads, durations, and results use local sample data.

## Original interface audit

1. **The settings footer competes with the queue.** Engine, model, output, timestamps, and speaker settings occupy several permanent rows. The redesign keeps the selection, duration, pass count, format, destination, and run action visible in a compact bottom bar. Run settings occupy the inspector when needed.
2. **The first model and additional passes are configured in different places.** One ordered list now owns all passes. Each pass has its own engine and model ID. Two Qwen passes can use different models. The editor supports moving a pass up, removing it, and choosing the filename suffix for a single pass.
3. **The meeting is the unit of work, but its files account for much of the complexity.** One checkbox includes the meeting. Disclosure shows participants and the reason a mixed export was skipped. The recording inspector previews merged filenames per pass, with the meeting ID when saving alongside the source.
4. **Ordinary files, meetings, model outputs, and failures present results differently.** A shared inspector places these details alongside the selected row. A failed file shows the decoder error and the affected pass. A finished file exposes saved results. The row highlight selects details independently of the checkbox that includes an item in a run.
5. **Import and model status take up space even when idle.** The original empty window has both a drop panel and an empty-queue prompt. The proposed empty window has one drop target. A toolbar control opens the models sheet throughout the workflow.

## Blind review and refinements

1. Two equivalent model passes previewed the same destination. The pass editor now flags the collision by pass number, disables Transcribe, and allows distinct models in the same family. Reorder controls and an explicit Done action are visible in the editor.
2. Changing a pass or clicking a checkbox moved keyboard focus to the page body. Interactive rows and editor controls now retain focus through updates. Closing the editor returns focus to its entry point.
3. Filtering could leave a hidden row in the inspector and silently include hidden files. The inspector follows the visible list, and the footer counts included files hidden by filters. The **Retry this recording** action only targets the inspected file.
4. Progress began above zero, intermediate work looked untouched, and the footer lost the outcome immediately after stopping. The prototype starts at zero, shows per-pass results during the run, reports completion or cancellation, and keeps finished outputs available.
5. The main action fell below the fold in common window sizes. The queue and inspector now scroll within the desktop window while the run bar stays visible.

Chrome checks covered keyboard focus, duplicate passes and reorder, export settings, filtered inclusion, scoped retry, partial cancellation, and layout at 1140×820, 1060×700, 800×650, 640×650, and 390×690. The visual checks also covered light and dark appearances and reduced transparency. [Light queue](queue.png) · [Dark queue](queue-dark.png) · [Model passes](passes.png) · [Partial failure](failed.png)

## SwiftUI handoff

- Use the system title bar and toolbar, an outline list or table for the queue, and the system inspector. Use native sheets, menus, segmented pickers, and form controls. Let macOS choose the accent color and selection treatment. [Adopting Liquid Glass](https://developer.apple.com/documentation/TechnologyOverviews/adopting-liquid-glass)
- Use an ordinary content surface for recordings, speaker tracks, and settings. Keep glass in the window chrome and native control groups. The browser mockup shows this division in light and dark appearances. [Apple materials guidance](https://developer.apple.com/design/human-interface-guidelines/materials)
- Bind planned outputs and run eligibility to the Rust planner. The prototype represents built-in model aliases and sample paths. Native code resolves custom paths, handles output collisions, discovers Zoom libraries and speaker groups, and restores the queue.
- Preserve literal and regular expression search, batch speaker-track marking, per-recording removal, model cache and cancellation, automatic and explicit speaker counts, and custom diarization model selection. Show existing-output reuse and limited-coverage warnings in the recording inspector.
- Drive progress and recovery from existing CLI events. Keep a clear stopping state and the affected model’s failure details for a partial run.
- Converting transcripts and merging previously saved JSON remain CLI workflows for this design.

## Review decisions

Review whether the inspector should open on the selected recording or run settings by default, whether expanded speaker tracks merit the extra queue height, and whether the footer provides enough information to start a run confidently.
