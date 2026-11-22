"use strict";

const el = id => document.getElementById(id);
const engines = {
    qwen: { label: "Qwen3 ASR", model: "qwen3-asr-1.7b", tag: "qw" },
    parakeet: { label: "Parakeet", model: "parakeet-v3", tag: "pk" },
    apple: { label: "Apple SpeechTranscriber", model: "System model", tag: "ap" }
};
const presets = new Map([
    ["qwen3-asr-1.7b", "qw"], ["qwen-1.7b", "qw"], ["Qwen/Qwen3-ASR-1.7B", "qw"],
    ["qwen3-asr-0.6b", "qw06"], ["qwen-0.6b", "qw06"], ["Qwen/Qwen3-ASR-0.6B", "qw06"],
    ["parakeet-v3", "pk"], ["parakeet-unified", "pk"], ["parakeet-unified-en", "pk"],
    ["nvidia/parakeet-unified-en-0.6b", "pk"], ["eschmidbauer/parakeet-unified-en-0.6b-onnx", "pk"],
    ["parakeet-tdt-v3", "pkt"], ["istupakov/parakeet-tdt-0.6b-v3-onnx", "pkt"]
]);
const icons = {
    audio: "M3 10v4m4-7v10m5-14v18m5-14v10m4-7v4",
    folder: "M2 7h8l2 2h10v11H2z M4 4h7l2 2h7",
    person: "M15 7a3 3 0 1 1-6 0 3 3 0 0 1 6 0 M5 19v-2a7 7 0 0 1 14 0v2",
    skip: "M5 12h14"
};
const clock = seconds => Math.floor(seconds / 60) + ":" + String(seconds % 60).padStart(2, "0");
const escape = value => String(value).replace(/[&<>"']/g, char => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
})[char]);
const plural = (size, noun, many = noun + "s") => size + " " + (size === 1 ? noun : many);
const attr = (value, name) => value ? " " + name : "";
const icon = name => '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="' + icons[name] + '"/></svg>';
const section = (label, content) => '<section><h3>' + label + '</h3>' + content + '</section>';
const line = (label, content) => '<div class="line"><span>' + label + '</span>' + content + '</div>';

let nextID = 1;
let jobs, passes, settings, inspected, pane, expanded, regex, run, lastRun, library, sheetName, opener;
let timer, download;

function makePass(engine, model = engines[engine].model) {
    return { id: nextID++, engine, model };
}

function modelName(pass) {
    return pass.model.trim() || engines[pass.engine].model;
}

function modelTag(pass) {
    const name = modelName(pass);
    if (pass.engine === "apple") return "ap";
    if (presets.has(name)) return presets.get(name);
    // The built-in aliases match the app. Custom paths are illustrative in the prototype.
    let hash = 0xcbf29ce484222325n;
    for (const byte of new TextEncoder().encode(name)) {
        hash = BigInt.asUintN(64, (hash ^ BigInt(byte)) * 0x100000001b3n);
    }
    return engines[pass.engine].tag + (hash & 0xffffffn).toString(16).padStart(6, "0");
}

function passProblem() {
    const tags = new Map();
    for (const [index, pass] of passes.entries()) {
        const tag = modelTag(pass);
        if (pass.engine !== "apple" && !tag.startsWith(engines[pass.engine].tag)) {
            return "Pass " + (index + 1) + " needs a model for " + engines[pass.engine].label + ".";
        }
        if (tags.has(tag)) {
            return "Passes " + (tags.get(tag) + 1) + " and " + (index + 1)
                + " would both use ." + tag + ". Choose a different model or remove one pass.";
        }
        tags.set(tag, index);
    }
    return "";
}

function makeRecording(name, kind, duration, included = false) {
    return { id: nextID++, name, kind, duration, included, outputs: [], track: false, diarize: false, speakers: "Auto" };
}

function planOutputs(job) {
    const stem = job.tracks ? job.name : job.name.replace(/\.[^.]+$/, "");
    const meetingID = job.tracks && settings.destination === "Next to source" ? " [" + job.meetingID + "]" : "";
    const ext = { JSON: "json", Text: "txt", SRT: "srt", VTT: "vtt" }[settings.format];
    const location = settings.destination === "Next to source" ? "Next to source" : settings.folder;
    return passes.map(pass => {
        const tag = modelTag(pass);
        const suffix = passes.length > 1 || settings.suffix ? "." + tag : "";
        return {
            name: stem + meetingID + (job.tracks ? ".merged" : "") + suffix + "." + ext,
            engine: engines[pass.engine].label, model: modelName(pass), tag, location
        };
    });
}

const samePath = (a, b) => a.name === b.name && a.location === b.location;
const sameOutput = (a, b) => samePath(a, b) && a.tag === b.tag;
const savedOutput = (job, expected) => job.outputs.find(output => output.state === "Complete" && sameOutput(output, expected));
const outstanding = job => !planOutputs(job).every(expected => savedOutput(job, expected));

function reset(scenario) {
    clearInterval(timer);
    clearTimeout(download);
    run = lastRun = null;
    library = "available";
    passes = [makePass("qwen"), makePass("parakeet")];
    settings = {
        format: "JSON", destination: "Next to source", folder: "~/Transcripts", timestamps: "Word level",
        acceleration: "Metal GPU", diarize: false, diarizationModel: "Sortformer", turns: false, suffix: false
    };
    const meeting = makeRecording("Fieldwork debrief", "Zoom meeting", 2274, true);
    meeting.tracks = ["Ada Moran", "Ellis Reed", "June Lake"];
    meeting.meetingID = "746219035";
    const interview = makeRecording("Orchard interview.wav", "WAV audio", 1096, true);
    const room = makeRecording("Room tone.m4a", "MPEG-4 audio", 128);
    const studio = makeRecording("Studio notes.flac", "FLAC audio", 455);
    room.outputs = planOutputs(room).map(output => ({ ...output, state: "Complete" }));
    studio.outputs = planOutputs(studio).map((output, index) => index
        ? { ...output, state: "Complete" }
        : { ...output, state: "Failed", error: "Qwen could not load the model." });
    jobs = scenario === "Empty" ? [] : [meeting, interview, room, studio];
    inspected = { Failed: studio.id, Complete: room.id }[scenario] ?? jobs[0]?.id ?? null;
    expanded = new Set([meeting.id]);
    pane = "recording";
    regex = false;
    el("query").value = el("filter").value = "";
    el("queueActions").open = false;
    repaint();
    if (scenario === "Running") begin([meeting, interview]);
}

function currentOutputs(job) {
    return planOutputs(job).map(expected => job.outputs.find(output => sameOutput(output, expected)) ?? expected);
}

function status(job) {
    if (run?.steps[run.index]?.job === job) return "Running";
    const results = currentOutputs(job);
    for (const value of ["Failed", "Canceled", "Waiting"]) {
        if (results.some(output => output.state === value)) return value;
    }
    return results.every(output => output.state === "Complete") ? "Complete" : "Ready";
}

function statusText(job, state) {
    if (state === "Running") return "Pass " + (run.steps[run.index].passIndex + 1);
    if (state === "Waiting") {
        const complete = currentOutputs(job).filter(output => output.state === "Complete").length;
        if (complete) return complete + "/" + passes.length + " passes";
    }
    return state === "Complete" ? "✓ Complete" : state;
}

function matchingJobs() {
    const query = el("query").value.trim();
    let pattern, issue;
    if (regex && query) {
        try { pattern = new RegExp(query, "i"); }
        catch { issue = "Enter a valid regular expression."; }
    }
    el("searchError").textContent = issue ?? "";
    el("searchError").hidden = !issue;
    el("query").setAttribute("aria-invalid", Boolean(issue));
    const filter = el("filter").value;
    return jobs.filter(job => (!query || issue || (pattern ? pattern.test(job.name)
        : job.name.toLowerCase().includes(query.toLowerCase()))) && (!filter || status(job) === filter));
}

function replace(container, markup, fallback) {
    const active = document.activeElement;
    const hadFocus = container.contains(active);
    const key = active.dataset.focus || active.id;
    container.innerHTML = markup;
    if (!hadFocus) return;
    const matching = [...container.querySelectorAll("[data-focus], [id]")].find(node =>
        (node.dataset.focus || node.id) === key && !node.disabled);
    (matching ?? fallback)?.focus({ preventScroll: true });
}

function queueRow(job) {
    const meeting = Boolean(job.tracks);
    const open = expanded.has(job.id);
    const state = status(job);
    const name = escape(job.name);
    const disclosure = meeting ? '<button class="quiet disclosure" data-expand="' + job.id + '" data-focus="expand-' + job.id
        + '" aria-expanded="' + open + '" aria-controls="tracks-' + job.id + '" aria-label="Speaker tracks for ' + name + '">'
        + (open ? "⌄" : "›") + '</button>' : "";
    const label = meeting ? " · " + plural(job.tracks.length, "speaker track") : job.track ? " · Speaker track" : "";
    return '<div class="row" data-row="' + job.id + '"><input type="checkbox" aria-label="Include ' + name
        + '" data-include="' + job.id + '" data-focus="include-' + job.id + '"' + attr(job.included, "checked") + attr(run, "disabled") + '>'
        + '<div class="record-controls">' + disclosure + '<button class="record" data-record="' + job.id + '" data-focus="record-' + job.id
        + '" aria-current="' + (job.id === inspected) + '" title="' + name + '"><span class="symbol">' + icon(meeting ? "folder" : "audio")
        + '</span><span class="record-copy"><span class="record-title">' + name + '</span><span class="sub">'
        + job.kind + label + '</span></span></button></div><span class="num">' + clock(job.duration)
        + '</span><span class="state ' + state.toLowerCase() + '">' + statusText(job, state) + '</span></div>'
        + (meeting ? '<ul id="tracks-' + job.id + '" class="tracks"' + attr(!open, "hidden") + '>'
            + job.tracks.map(person => '<li class="child">' + icon("person") + person + '<small>Speaker track</small></li>').join("")
            + '<li class="child">' + icon("skip") + 'Mixed recording<small>Skipped · Speaker tracks available</small></li></ul>' : "");
}

function drawQueue(visible) {
    const meetingCount = jobs.filter(job => job.tracks).length;
    const audioCount = jobs.length - meetingCount;
    el("count").textContent = jobs.length
        ? [meetingCount && plural(meetingCount, "meeting"), audioCount && plural(audioCount, "audio file")].filter(Boolean).join(" · ")
        : "No recordings";
    el("filterBar").hidden = !jobs.length;
    el("columns").hidden = !visible.length;
    for (const button of el("queueActions").querySelectorAll("button")) {
        const action = button.dataset.action;
        button.disabled = Boolean(run) || (action === "clear" ? !jobs.some(job => status(job) === "Complete")
            : action === "tracks" ? !visible.some(job => !job.tracks) : !visible.length);
    }
    const markup = visible.length ? visible.map(queueRow).join("") : jobs.length
        ? '<div class="empty"><h2>No matching recordings</h2><button id="clearFilters">Clear filters</button></div>'
        : '<div class="empty"><span class="symbol">' + icon("folder") + '</span><h2>Add recordings</h2>'
            + '<p>Choose audio files, folders, or a Zoom library.</p><button class="primary" id="emptyAdd">Add recordings…</button></div>';
    replace(el("rows"), markup, jobs.length ? el("query") : el("add"));
}

function picker(id, choices, value, disabled = Boolean(run)) {
    const options = choices.map(choice => {
        const [key, label] = Array.isArray(choice) ? choice : [choice, choice];
        return '<option value="' + escape(key) + '"' + attr(key === value, "selected") + '>' + escape(label) + '</option>';
    });
    return '<select id="' + id + '"' + attr(disabled, "disabled") + '>' + options.join("") + '</select>';
}

const field = (id, label, input) => '<div class="line"><label for="' + id + '">' + label + '</label>' + input + '</div>';
const checkField = (id, label, value, disabled = Boolean(run)) => field(id, label,
    '<input type="checkbox" id="' + id + '"' + attr(value, "checked") + attr(disabled, "disabled") + '>');
const textField = (id, label, value) => field(id, label,
    '<input type="text" id="' + id + '" value="' + escape(value) + '"' + attr(run, "disabled") + '>');
const supportsDiarization = () => passes.some(pass => pass.engine !== "apple");

function drawSettings() {
    const timestamps = settings.format === "Text" ? ["Off"] : settings.format === "JSON"
        ? ["Word level", "Sentence level", "Off"] : ["Word level", "Sentence level"];
    const issue = passProblem();
    return section("Model passes", passes.map(pass => '<div class="out"><strong>' + engines[pass.engine].label
        + '</strong><span class="file">' + escape(modelName(pass)) + '</span></div>').join("")
        + (issue ? '<p class="validation">' + escape(issue) + '</p>' : "")
        + '<button id="edit"' + attr(run, "disabled") + '>Edit passes…</button>')
        + section("Output",
            field("format", "Format", picker("format", ["JSON", "Text", "SRT", "VTT"], settings.format))
            + field("destination", "Destination", picker("destination", ["Next to source", "Output folder"], settings.destination))
            + (settings.destination === "Output folder" ? textField("folder", "Folder", settings.folder) : "")
            + field("timestamps", "Timestamps", picker("timestamps", timestamps, settings.timestamps, Boolean(run) || settings.format === "Text"))
            + (settings.format === "Text" ? checkField("turns", "Speaker turns", settings.turns, Boolean(run) || !supportsDiarization()) : ""))
        + (supportsDiarization() ? section("Recognition",
            field("acceleration", "Acceleration", picker("acceleration", ["Metal GPU", "CPU"], settings.acceleration))
            + checkField("diarize", "Diarize by default", settings.diarize)
            + (settings.diarize ? textField("diarizationModel", "Diarization model", settings.diarizationModel) : "")) : "");
}

function outputRow(output, index) {
    const state = output.state;
    return '<div class="out"><div class="out-meta"><strong class="out-name">' + output.engine + '</strong>'
        + (state ? '<span class="state ' + state.toLowerCase() + '">' + state + '</span>' : '')
        + '</div><small>' + escape(output.model) + '</small>'
        + (state === "Complete" ? '<button class="reveal file" data-reveal="' + index + '" data-focus="reveal-' + index
            + '" aria-label="Show ' + escape(output.name) + ' in Finder">' + escape(output.name) + ' ↗</button>'
            : '<span class="file">' + escape(output.name) + '</span>')
        + (output.error ? '<p class="failure-note">' + escape(output.error) + '</p>' : "") + '</div>';
}

function drawRecording(job) {
    if (!job) return '<section><p>Select a recording to see its details.</p></section>';
    const state = status(job);
    const issue = passProblem();
    const outputs = currentOutputs(job);
    const tracks = job.tracks
        ? section("Speaker tracks", line("Participants", job.tracks.length) + line("Bleed reduction", "Automatic") + line("Merge order", "By timestamp"))
        : section("Track settings", checkField("speakerTrack", "Speaker track", job.track)
            + checkField("speakerDiarize", "Diarize track", job.diarize, Boolean(run) || !supportsDiarization() || job.track)
            + (job.diarize && supportsDiarization() ? field("speakers", "Speakers",
                picker("speakers", ["Auto", "1", "2", "3", "4", "5", "6", "7", "8"], job.speakers)) : "")
            + (job.track ? "<p>Speaker tracks from the same folder are merged.</p>" : ""));
    const previous = outputs.some(output => output.state);
    const oldOutputs = job.outputs.filter(output => output.state === "Complete" && !outputs.includes(output));
    return '<section><p class="recording-kind">' + job.kind + '</p><h2>' + escape(job.name) + '</h2><p>'
        + clock(job.duration) + ' · ' + statusText(job, state) + '</p></section>'
        + (state === "Running" ? section("Current pass", '<progress aria-label="Current pass progress" value="' + run.stepElapsed
            + '" max="' + run.stepSeconds + '"></progress>') : "")
        + (previous ? "" : tracks)
        + section(previous ? "Model results" : "Planned output", issue
            ? '<p class="validation">Resolve the model passes in Run settings.</p>'
            : '<small>' + escape(outputs[0].location) + '</small>' + outputs.map(output => outputRow(
                run?.steps[run.index]?.output === output ? { ...output, state: "Running" } : output,
                job.outputs.indexOf(output))).join(""))
        + (oldOutputs.length ? section("Earlier results", oldOutputs.map(output => outputRow(output, job.outputs.indexOf(output))).join("")) : "")
        + (previous ? tracks : "")
        + '<section class="ins-actions">' + (state === "Failed" || state === "Canceled"
            ? '<button id="retry"' + attr(run || issue, "disabled") + '>Retry This Recording</button>' : "")
        + '<button id="removeRecording"' + attr(run, "disabled") + '>Remove from queue</button></section>';
}

function drawInspector(visible) {
    el("rec").setAttribute("aria-pressed", pane === "recording");
    el("run").setAttribute("aria-pressed", pane === "settings");
    const markup = pane === "settings" ? drawSettings() : drawRecording(visible.find(job => job.id === inspected));
    replace(el("ins"), markup, el(pane === "settings" ? "run" : "rec"));
}

function drawRunbar(visible) {
    const start = el("start");
    const setup = el("setupSummary");
    const detail = el("footSub");
    const title = el("footTitle");
    const progress = el("runProgress");
    if (run) {
        const step = run.steps[run.index];
        const duration = run.steps.length * run.stepSeconds;
        title.textContent = run.canceling ? "Stopping run…" : "Pass " + (step.passIndex + 1)
            + " of " + passes.length + " · " + step.output.engine;
        detail.textContent = step.job.name + " · " + clock(run.elapsed) + " elapsed · ~" + clock(duration - run.elapsed) + " remaining";
        progress.max = duration;
        progress.value = run.elapsed;
        start.textContent = run.canceling ? "Stopping…" : "Cancel run";
        start.disabled = false;
        start.setAttribute("aria-disabled", run.canceling);
    } else {
        const included = jobs.filter(job => job.included);
        const hidden = included.filter(job => !visible.includes(job)).length;
        const scope = included.length + " included" + (hidden ? " · " + hidden + " hidden by filters" : "");
        const time = included.reduce((total, job) => total + job.duration, 0);
        const issue = passProblem();
        title.textContent = lastRun ?? (scope + (included.length ? " · " + clock(time) : ""));
        detail.textContent = " · " + scope;
        setup.textContent = issue
            ? "Resolve model passes"
            : plural(passes.length, "model pass", "model passes")
                + " · " + settings.format + " · " + settings.destination;
        start.textContent = "Transcribe";
        start.disabled = !included.some(outstanding) || Boolean(issue);
        start.removeAttribute("aria-disabled");
    }
    setup.hidden = Boolean(run);
    detail.hidden = !run && !lastRun;
    progress.hidden = !run;
    start.classList.toggle("running-action", Boolean(run));
}

function repaint({ followFilter = false, keepInspector = false } = {}) {
    const visible = matchingJobs();
    if (followFilter && !visible.some(job => job.id === inspected)) inspected = visible[0]?.id ?? null;
    el("regex").setAttribute("aria-pressed", regex);
    el("add").disabled = Boolean(run);
    drawQueue(visible);
    if (!keepInspector) drawInspector(visible);
    drawRunbar(visible);
}

function showSheet(name, heading, markup, focusID) {
    const dialog = el("sheet");
    if (!dialog.open) opener = document.activeElement;
    sheetName = name;
    el("sheetTitle").textContent = heading;
    replace(el("sheetBody"), markup, el("closeSheet"));
    if (!dialog.open) dialog.showModal();
    if (focusID) el(focusID)?.focus();
}

function passEditor(focusID = "engine-" + passes[0].id) {
    const cards = passes.map((pass, index) => {
        const number = index + 1;
        const action = (operation, label, text, disabled) => '<button data-pass-action="' + operation + '" data-pass-id="' + pass.id
            + '" aria-label="' + label + ' pass ' + number + '" title="' + label + '"' + attr(disabled, "disabled") + '>' + text + '</button>';
        return '<div class="pass"><div class="pass-head"><strong>Pass ' + number + '</strong><div class="pass-actions">'
            + action("up", "Move up", "↑", index === 0) + action("down", "Move down", "↓", index === passes.length - 1)
            + action("remove", "Remove", "Remove", passes.length === 1) + '</div></div>'
            + '<div class="field"><label for="engine-' + pass.id + '">Engine</label>'
            + picker("engine-" + pass.id, Object.entries(engines).map(([key, value]) => [key, value.label]), pass.engine) + '</div>'
            + '<div class="field"><label for="model-' + pass.id + '">Model</label><input id="model-' + pass.id
            + '" value="' + escape(pass.model) + '" placeholder="Model ID or local path"' + attr(pass.engine === "apple", "disabled") + '></div></div>';
    });
    const issue = passProblem();
    showSheet("passes", "Model passes", cards.join("")
        + checkField("suffix", "Append model ID to filenames", passes.length > 1 || settings.suffix, passes.length > 1)
        + '<p class="validation" id="passError" aria-live="polite"' + attr(!issue, "hidden") + '>' + escape(issue) + '</p>'
        + '<div class="sheet-actions"><button id="newPass">Add pass</button><button class="primary push" id="done">Done</button></div>', focusID);
}

function showModels(focusID) {
    showSheet("models", "Model library",
        section("On this Mac", line("Qwen3 ASR", "1.7B") + line("Qwen3 forced aligner", "0.6B")
            + line("Parakeet", "Unified EN") + (library === "installed" ? line("Sortformer", "4 speakers") : ""))
        + (library === "installed" ? "" : section("Available", line("Sortformer", library === "downloading"
            ? '<button id="cancelDownload">Cancel download</button>' : '<button id="download">Download</button>')
            + (library === "downloading" ? '<progress aria-label="Downloading Sortformer"></progress>' : ""))), focusID);
}

function showImport() {
    showSheet("import", "Add recordings", '<p>Choose a sample recording for the prototype.</p><div class="sheet-actions">'
        + '<button id="sampleAudio">Add sample audio</button><button id="sampleZoom">Add sample Zoom meeting</button></div>', "sampleAudio");
}

function finish(canceled) {
    clearInterval(timer);
    const complete = run.jobs.filter(job => !outstanding(job)).length;
    const saved = run.jobs.flatMap(currentOutputs).filter(output => output.state === "Complete").length;
    lastRun = canceled ? "Run canceled · " + plural(saved, "saved result") + " available"
        : "Run complete · " + plural(complete, "recording");
    run = null;
    repaint({ followFilter: true });
    if (document.activeElement === document.body) {
        (el("start").disabled ? el("setupSummary") : el("start")).focus({ preventScroll: true });
    }
}

function tick() {
    run.elapsed += 1;
    run.stepElapsed += 1;
    if (run.stepElapsed < run.stepSeconds) {
        drawRunbar(matchingJobs());
        const bar = el("ins").querySelector("progress");
        if (bar) bar.value = run.stepElapsed;
        return;
    }
    const { job, output } = run.steps[run.index];
    output.state = "Complete";
    job.outputs = job.outputs.filter(previous => previous === output || !samePath(previous, output));
    run.stepElapsed = 0;
    run.index += 1;
    if (run.index === run.steps.length) finish(false);
    else repaint({ followFilter: true });
}

function begin(selected) {
    if (!selected.length || run || passProblem()) return;
    const steps = [];
    for (const [passIndex] of passes.entries()) {
        for (const job of selected) {
            const output = planOutputs(job)[passIndex];
            if (savedOutput(job, output)) continue;
            output.state = "Waiting";
            job.outputs = job.outputs.filter(previous => !sameOutput(previous, output));
            job.outputs.push(output);
            steps.push({ job, output, passIndex });
        }
    }
    if (!steps.length) {
        lastRun = "All included results already exist";
        repaint();
        return;
    }
    lastRun = null;
    run = { jobs: selected, steps, index: 0, elapsed: 0, stepElapsed: 0, stepSeconds: 5 };
    timer = setInterval(tick, 1000);
    repaint({ followFilter: true });
}

function cancelRun() {
    if (run.canceling) return;
    clearInterval(timer);
    run.canceling = true;
    drawRunbar(matchingJobs());
    timer = setTimeout(() => {
        for (const { output } of run.steps) if (output.state === "Waiting") output.state = "Canceled";
        finish(true);
    }, 450);
}

function setPane(value) {
    pane = value;
    drawInspector(matchingJobs());
}

el("query").oninput = el("filter").onchange = () => repaint({ followFilter: true });
el("regex").onclick = () => {
    regex = !regex;
    repaint({ followFilter: true });
};
el("rec").onclick = () => setPane("recording");
el("run").onclick = () => setPane("settings");
el("setupSummary").onclick = () => {
    setPane("settings");
    el("run").focus();
};
el("models").onclick = () => showModels("closeSheet");
el("add").onclick = showImport;
el("start").onclick = () => run ? cancelRun() : begin(jobs.filter(job => job.included));
el("scenario").onchange = event => reset(event.target.value);
el("closeSheet").onclick = () => el("sheet").close();
el("sheet").onclose = () => {
    if (opener?.isConnected && !opener.disabled) opener.focus({ preventScroll: true });
    else if (sheetName === "passes" && el("edit")) el("edit").focus({ preventScroll: true });
    else el("add").focus({ preventScroll: true });
    sheetName = null;
};

el("rows").onclick = event => {
    const button = event.target.closest("button");
    if (button?.id === "emptyAdd") return showImport();
    if (button?.id === "clearFilters") {
        el("query").value = el("filter").value = "";
        repaint({ followFilter: true });
        return;
    }
    if (button?.dataset.expand) {
        const id = Number(button.dataset.expand);
        if (expanded.has(id)) expanded.delete(id);
        else expanded.add(id);
        drawQueue(matchingJobs());
        return;
    }
    if (event.target.closest("input")) return;
    const row = event.target.closest("[data-row]");
    if (row) {
        inspected = Number(row.dataset.row);
        pane = "recording";
        repaint();
    }
};

el("rows").onchange = event => {
    if (!event.target.dataset.include || run) return;
    const job = jobs.find(job => job.id === Number(event.target.dataset.include));
    job.included = event.target.checked;
    lastRun = null;
    repaint();
};

el("queueActions").onclick = event => {
    const action = event.target.dataset.action;
    if (!action || run) return;
    const visible = matchingJobs();
    el("queueActions").open = false;
    el("queueActions").querySelector("summary").focus();
    if (action === "tracks") {
        const count = visible.filter(job => !job.tracks).length;
        showSheet("tracks", "Set speaker tracks", '<p>' + plural(count, "visible audio file")
            + '</p><div class="sheet-actions"><button id="clearTracks">Treat as recordings</button>'
            + '<button class="primary" id="markTracks">Mark as speaker tracks</button></div>', "markTracks");
        return;
    }
    if (action === "clear") jobs = jobs.filter(job => status(job) !== "Complete");
    else for (const job of visible) job.included = action === "include";
    lastRun = null;
    repaint({ followFilter: true });
};

el("ins").onclick = event => {
    const button = event.target.closest("button");
    if (!button) return;
    const job = jobs.find(job => job.id === inspected);
    if (button.id === "edit") passEditor();
    if (button.id === "retry" && job) begin([job]);
    if (button.id === "removeRecording" && job) {
        jobs = jobs.filter(item => item !== job);
        lastRun = null;
        repaint({ followFilter: true });
    }
    if (button.dataset.reveal && job) {
        const output = job.outputs[Number(button.dataset.reveal)];
        showSheet("reveal", "Finder preview", '<strong>' + output.engine + '</strong><p class="file">'
            + escape(output.name) + '</p><p>' + escape(output.location) + '</p>', "closeSheet");
    }
};

el("ins").onchange = event => {
    const input = event.target;
    if (input.type === "text") return;
    const job = jobs.find(item => item.id === inspected);
    const value = input.type === "checkbox" ? input.checked : input.value;
    if (input.id === "speakerTrack") {
        job.track = value;
        if (value) job.diarize = false;
    } else if (input.id === "speakerDiarize") {
        job.diarize = value;
    } else if (input.id === "speakers") {
        job.speakers = value;
    } else {
        settings[input.id] = value;
        if (input.id === "format") {
            if (value === "Text") settings.timestamps = "Off";
            else if (value !== "JSON" && settings.timestamps === "Off") settings.timestamps = "Sentence level";
        }
    }
    repaint({ followFilter: true });
};

el("ins").oninput = event => {
    const input = event.target;
    if (input.type !== "text") return;
    settings[input.id] = input.value;
    repaint({ followFilter: true, keepInspector: true });
};

el("sheetBody").onchange = event => {
    const input = event.target;
    if (input.id.startsWith("engine-")) {
        const pass = passes.find(pass => pass.id === Number(input.id.slice(7)));
        pass.engine = input.value;
        pass.model = engines[pass.engine].model;
        passEditor(input.id);
    }
    if (input.id === "suffix") settings.suffix = input.checked;
    repaint({ followFilter: true });
};

el("sheetBody").oninput = event => {
    const input = event.target;
    if (!input.id.startsWith("model-")) return;
    passes.find(pass => pass.id === Number(input.id.slice(6))).model = input.value;
    const issue = passProblem();
    el("passError").textContent = issue;
    el("passError").hidden = !issue;
    repaint({ followFilter: true });
};

function editPass(button) {
    const index = passes.findIndex(pass => pass.id === Number(button.dataset.passId));
    const action = button.dataset.passAction;
    const [pass] = passes.splice(index, 1);
    if (action !== "remove") passes.splice(index + (action === "up" ? -1 : 1), 0, pass);
    const focus = action === "remove" ? passes[Math.min(index, passes.length - 1)] : pass;
    passEditor("engine-" + focus.id);
    repaint({ followFilter: true });
}

function importSample(zoom) {
    const name = zoom ? "Lakeside planning" : "Lakeside walk.wav";
    const existing = jobs.find(job => job.name === name);
    const job = existing ?? makeRecording(name, zoom ? "Zoom meeting" : "WAV audio", 494, true);
    if (zoom && !existing) {
        job.tracks = ["Avery Lane", "Riley Park"];
        job.meetingID = "271864205";
    }
    if (!existing) jobs.push(job);
    el("query").value = el("filter").value = "";
    inspected = job.id;
    pane = "recording";
    el("sheet").close();
    lastRun = null;
    repaint();
}

el("sheetBody").onclick = event => {
    const button = event.target.closest("button");
    if (!button) return;
    if (button.dataset.passAction) return editPass(button);
    switch (button.id) {
    case "done":
        el("sheet").close();
        break;
    case "newPass": {
        const used = new Set(passes.map(modelTag));
        const choices = [
            ...Object.entries(engines).map(([engine, { model, tag }]) => ({ engine, model, tag })),
            { engine: "qwen", model: "qwen3-asr-0.6b", tag: "qw06" },
            { engine: "parakeet", model: "parakeet-tdt-v3", tag: "pkt" }
        ];
        const { engine, model } = choices.find(choice => !used.has(choice.tag))
            ?? { engine: "qwen", model: "" };
        const pass = makePass(engine, model);
        passes.push(pass);
        passEditor("engine-" + pass.id);
        repaint();
        break;
    }
    case "download":
        library = "downloading";
        showModels("cancelDownload");
        download = setTimeout(() => {
            library = "installed";
            if (sheetName === "models" && el("sheet").open) showModels("closeSheet");
        }, 1800);
        break;
    case "cancelDownload":
        clearTimeout(download);
        library = "available";
        showModels("download");
        break;
    case "sampleAudio":
    case "sampleZoom":
        importSample(button.id === "sampleZoom");
        break;
    case "clearTracks":
    case "markTracks":
        for (const job of matchingJobs().filter(job => !job.tracks)) {
            job.track = button.id === "markTracks";
            if (job.track) job.diarize = false;
        }
        el("sheet").close();
        repaint();
        break;
    }
};

reset(el("scenario").value);
