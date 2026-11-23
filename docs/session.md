# Recorded audio sessions

Run `transcriberrr capabilities` and require `session_schema_version: 1`.
The `session` capability reports the sample rate, language, bleed policy, and
maximum supported diarization speaker count. Each build reports its engine
availability separately.

```sh
transcriberrr session --manifest session.json --jsonl --offline
transcriberrr session --manifest - --jsonl --reconcile-only < updated-session.json
```

The [manifest schema](schema/transcript-session-v1.schema.json) declares track
identity, exact source and session clocks, and all artifact paths. A job may
contain audio tracks from multiple streams of the same source. Pass each input
as a 16 kHz, mono, signed 16-bit PCM RIFF WAV. Supply silence for a presentation
gap. The helper uses the PCM samples and the clocks in the manifest. Session placement uses the signed `start_frame`
rounded from `binding.session_start` to the nearest sample, with ties away from
zero. `binding.origin` is the exact source presentation time of sample zero.

Use `isolated_mic` for a track dedicated to one speaker, `mixed_audio` for a
mixed recording, and `unknown` when the role is uncertain. Speaker hints belong
to the speaker and can differ from track names. A silent camera without an
audio stream is a coverage-only track with null stream index and no audio or
artifact paths. Use `recognize: false` to skip recognition for an audio track.
Staging its WAV and an activity path still yields level frames.
Supply an `unavailable` reason for an offline or missing source. The result
reports that track as failed and preserves its placement and any validated levels.

The engine defaults to `qwen3`. The batch engines currently recognize English.
Mixed-track diarization defaults to enabled on builds and engines that include it.
Set `diarization: false` to disable it or supply an expected speaker count to request it. Reports for isolated mics derive session speaker identity from the
track and its speaker hint. A per-engine numeric limit in capabilities reflects
the compiled Sortformer speaker output count. The Apple engine requires a
bundled native helper and installed Apple speech assets. In a mixed-track session, Apple supplies the word times and Sortformer assigns speaker labels from the same PCM samples. Isolated headsets keep their own speaker hints. An explicit `"diarization": false` with no speaker-count override runs Apple recognition alone. The `capabilities` response reports support for each engine.

Recognition uses installed assets by default. Pass `--offline` to make that
choice explicit or `--download` to let Transcriberrr fetch missing assets.
Missing offline models are named in the result. `--cached-model` reports the
local model snapshot path. The caller should record the model snapshot,
extraction version, stream evidence, and recognition options in its recognition
key. A track's role can change its diarization settings. Session placement can
change independently of recognition because this mode does not mute bleed.

For each recognized track, the helper publishes an atomic recognition artifact
at its `result` path. It contains version, sample rate, recognition key,
stream evidence, duration in samples, verbatim words, and an optional warning.
Word and utterance IDs are local to that artifact. When configured, the
`activity` path receives the level frames independently with the same clock.
Otherwise levels are embedded in the recognition artifact. Levels are linear
RMS over 20 ms windows. A final short window spans only the available samples.

Use `reuse: true` to read an existing recognition artifact for a track during a
mixed new-and-cached run. Use `--reconcile-only` to read stored artifacts for all
selected tracks with no model or WAV access. A caller can stage valid cached artifacts at the declared paths. An MPCut
standalone manifest can also supply a shared-cache descriptor. The provider
checks the staged WAV and source snapshots, derives the recognition key from
its executable and installed model revisions, and publishes reusable results
under that key. A mismatched or
missing recognition artifact is a failed track. Skipped or failed audio tracks
can retain independently published activity frames.

The [result schema](schema/transcript-result-v1.schema.json) contains the merged
conversation, all track-local words, and the available level frames. Every
conversation word retains its individual, original observations. Reconciliation
associates words using spelling, signed placement, and speaker consistency.
Ambiguous word associations remain separate conversation words and can include
inferred references to another decoded word. Speech and speaker observations
come from recognized words. The result records level frames independently.
For isolated mics, repeated word matches can calibrate the level difference
and modulation shared by one mic and its quieter copy. That evidence lets the
conversation link spill to the source mic while preserving each decoded word
on its original track.

With `--jsonl`, events carry `schema_version`, `job_id`, `fingerprint`,
`event`, and `status`. The helper emits `session_started`, then progress and
outcomes for individual tracks, followed by `session_finished`. Track progress
includes `track_id`, `progress` (0–1), and `phase`. Track completion includes
`track_id`, artifact destinations, and any error or warning. The terminal status
is `completed`, `partial`, `failed`, or `canceled`. Completed and partial runs
publish the result atomically and report its path. A run whose selected tracks all
failed retains their error events and any previous result, and finishes as failed. An invalid manifest that
cannot be decoded produces a command error. Unstructured diagnostics go to
stderr.

Send SIGTERM to cancel and keep already published artifacts. Set
`TRANSCRIBERRR_PARENT_PID` when launching from a host to cancel at checkpoints
if the host exits. Keep staged files until the caller records completion,
cancellation, or discard.

Convert a native result offline with
`transcriberrr convert result.json --format srt -o result.srt`.
Use `--format json` to preserve sample bounds and evidence.
Text and subtitle output use the merged
conversation. A timestamped conversion with speech before session zero reports
the session sample used for cue zero.

To use a development build with MPCut, set `MPCUT_TRANSCRIBERRR_HELPER`
to the absolute path of the bundled CLI when launching MPCut from the shell.

```sh
MPCUT_TRANSCRIBERRR_HELPER="/path/to/Transcriberrr-Session.app/Contents/Helpers/transcriberrr" \
  "/path/to/MPCut.app/Contents/MacOS/MPCut"
```

Check that the helper's `capabilities` output reports `session_schema_version: 1`.
The bundle keeps the Apple SpeechTranscriber helper beside the CLI.

### Reading overlapping dialogue

Each track keeps its own words and sample times. The session merge uses matching word order, timing, and measured microphone activity to distinguish bleed from simultaneous speech. Two people can say the same word at the same time. Evidence from distinct speaker microphones can preserve both occurrences.

A participant without a microphone is heard clearly on a mixed recording and only as spill on every isolated microphone. With two or more isolated microphones, the session compares each word's level on a microphone with the balance that wearer's own speech keeps against the same mixed recording. A word that every microphone holds at spill level belongs to none of the wearers. It appears once, with the mixed recording's wording and timing, under a speaker of its own, and the microphones' faint copies stay attached as recognized observations with no speaker. A mixed recording's diarizer can give such a participant a wearer's label. That label is set aside for those words only. Such a participant can keep talking after a wearer resumes. The session follows the phrase on the mixed recording from a word in the wearers' pause through the words that follow or precede it within half a second. Each of those words stays with the participant when every microphone is either at spill level or recognized its wearer saying something else there, with no copy of the word. A phrase that lies entirely under a wearer's speech has no word in a pause. It is still placed when three or more of its words in a row leave the mixed recording at least twice as loud as any wearer's microphone accounts for, while those microphones recognized their wearers saying something else. A quieter voice under a wearer, or a phrase of one or two words, keeps the diarizer's label. A participant that one recording's diarizer keeps apart and another's merges with a wearer is one speaker. Words placed off microphone under a wearer's borrowed label belong to the one off-microphone voice a diarizer kept apart when the session has exactly one. With none, or with several, they share a speaker of their own. A word of such a phrase that only a microphone caught, far below its wearer's level and beside its faint copies of the rest, belongs to that participant too.

A word's speaker comes first from a microphone that heard it as its wearer's speech or as calibrated spill, then from a participant placed off all microphones, and last from a mixed recording's diarizer label.

Parakeet Unified reports the encoder frame that emitted a word, about 0.28 s after the spoken onset, with one or two frames of duration. Session and batch results move each onset back by that latency and end each word at the next onset, or at a length estimated from its spelling before a pause. Qwen3 with the forced aligner remains the more exact clock for edits on word boundaries.

Occasionally two microphones supply different spellings or readings of one utterance. When the surrounding words, speaker, and timing support that interpretation, the conversation presents one reading. When the recordings split evenly, the speaker's own microphone supplies the reading. The other track's observation carries its original text, local word ID and sample range, with `evidence: "inferred"` and a basis such as `alternative_reading_between_matched_words`. Open the source track to inspect what that recognizer actually returned. When three or more recordings each read a word differently, the readings one letter away from the chosen one attach to it and an unrelated reading stays a word of its own.

A recognizer can write a compound as one word where another writes its two parts. After the same shared word and before the same word or the end of the recording these are one reading, and the parts or the compound attach to the chosen spelling as inferred observations. A number written in digits on one recording and said in words on another follows the same rule, with the basis `equivalent_spoken_number`, or `equivalent_spoken_ordinal` for a word such as `3rd`. So `200` joins `two hundred`, `1,250` joins `twelve fifty`, `$200` joins `two hundred dollars`, and `4:00` and `four` are one word.

The session holds no table of number words. Every way a language says a number comes from the rule sets of ICU, the Unicode library in macOS: each gender and case, `two hundred and fifty`, a year said in two halves, the ending an ordinal is written with (`3rd`, `3.`, `3.ª`, `3-й`), the grouping and decimal marks, and the names of currency and percent signs. A time of day is read as its two figures side by side. The language is the manifest's `language` (a tag such as `de-DE` or an English name such as `German`), or else the language the system reads in the words when there are at least 200 characters of them, or else English, the language of the batch recognizers. A session without a word in digits does none of this.

A microphone can read its neighbour's faint speech as different words. Where the neighbour's own microphone recognized that stretch, its wording stands and the faint reading attaches with the basis `alternative_reading_of_spill`. Two faint words that together spell the neighbour's word attach the same way, and so does a microphone's faint reading of a participant who has no microphone. A microphone can also recognize its neighbour's word where no spill was calibrated, or where its own wearer starts to talk before the word ends. When the neighbour's microphone recognized the same word at speaking level and this microphone was still quiet as the word began, the word belongs to the neighbour. Two wearers who say the same word together are each loud on their own microphone and stay separate.

A far recording such as a camera microphone can garble what the speaker's own microphone hears cleanly, in other words and a different number of them. Its unmatched word joins the microphone's wording as an inferred observation with the basis `far_reading_within_own_microphone_phrase` when it stands beside a word both recordings share, inside the stretch the microphone recognized at speaking level, and the wearer's speech accounts for the far recording's level there. The word stays a conversation word when the far recording is louder than the microphone accounts for, which is how a second voice shows; when another recording, even a faint copy, has the same or nearly the same word; when it stands alone against a single word of the microphone, where the rule for one alternate reading applies; and when the microphone recognized nothing between the shared words.

Recognizers place a word poorly when other speech covers it, and Qwen3's aligner can leave such a word with almost no length. Word order still holds. Directly before or after a word that two recordings share, the same word from the same speaker is one conversation word even when its placement differs by up to a second. Its observation carries the basis `ordered_phrase_beside_matched_word`. Words from different speakers stay separate, and so do words that no recording gives a speaker. A final word that one recognizer stretches into the following pause joins the recordings it overlaps once any of them matches it.

Where no speaker's own microphone heard a turn, the recording that heard the most of it supplies the wording and timing of all its words. A conversation utterance is one speaker's part of one recording's utterance. It keeps its ID when another speaker's words fall between its own, so utterance IDs can alternate during overlapping speech.

Named speakers receive IDs in order of first appearance. Speakers without a name follow them: first a participant without a microphone, then the remaining voices in manifest track order. A voice that appears or disappears when a recording is moved therefore leaves the IDs of the named speakers unchanged.
