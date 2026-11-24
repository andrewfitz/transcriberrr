//! An isolated mic's decaying syllable can be decoded twice, with the second copy
//! assigned to the quiet tail. Keep repetitions with a second attack or uncertain levels.
use super::{ACTIVITY_WINDOW, Activity, LocalWord, reconcile::word_text};

pub(super) fn remove_decay_repeats(words: &mut Vec<LocalWord>, activity: &[Activity]) {
    words.dedup_by(|later, earlier| is_decay_repeat(earlier, later, activity));
}

fn is_decay_repeat(first: &LocalWord, second: &LocalWord, activity: &[Activity]) -> bool {
    let frame = ACTIVITY_WINDOW as i64;
    let gap = second.start_sample - first.end_sample;
    if !(0..=frame).contains(&gap)
        || !(3 * frame..=7 * frame).contains(&(second.end_sample - second.start_sample))
        || !(3 * frame..=30 * frame).contains(&(first.end_sample - first.start_sample))
        || first.speaker_id != second.speaker_id
    {
        return false;
    }
    let text = word_text(&first.text);
    if text.is_empty() || text != word_text(&second.text) {
        return false;
    }
    // Activity is validated on a 20 ms grid. Include the frame containing the
    // second onset and two frames after its end to allow for quantized word times.
    let onset = (second.start_sample / frame) as usize;
    let end = (second.end_sample as usize).div_ceil(ACTIVITY_WINDOW);
    let Some(tail) = activity.get(onset..end + 2) else {
        return false;
    };
    let first_peak = activity[(first.start_sample / frame) as usize..onset]
        .iter()
        .map(|f| f.level)
        .fold(0.0_f32, f32::max);
    // Require a voiced first copy and a tail at least 30 dB below it. A renewed
    // rise, even one still below that threshold, may be a quietly spoken repeat.
    let mut minimum = activity[onset - 1].level;
    if first_peak < 0.01 || minimum > first_peak * 0.1 {
        return false;
    }
    for f in tail {
        if f.level > first_peak / 32.0 || f.level > minimum * 1.5 + 0.0001 {
            return false;
        }
        minimum = minimum.min(f.level);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::Precision;

    fn word(id: &str, text: &str, from: i64, to: i64) -> LocalWord {
        LocalWord {
            id: id.into(),
            utterance_id: id.into(),
            text: text.into(),
            start_sample: from,
            end_sample: to,
            speaker_id: None,
            precision: Precision::Word,
        }
    }

    #[test]
    fn only_removes_a_short_copy_in_the_same_syllables_decay() {
        let frames = [
            0.12, 0.17, 0.10, 0.04, 0.012, 0.006, 0.003, 0.002, 0.001, 0.0003, 0.0002, 0.0001, 0.0,
        ];
        let mut activity: Vec<_> = frames
            .into_iter()
            .enumerate()
            .map(|(i, level)| Activity {
                start_sample: i as i64 * 320,
                end_sample: (i + 1) as i64 * 320,
                level,
            })
            .collect();
        let first = word("w8", "Yep.", 25, 1990);
        let second = word("w9", " yep!", 1990, 3270);
        let mut words = vec![
            first.clone(),
            second.clone(),
            word("w10", "Now", 6000, 7000),
        ];
        remove_decay_repeats(&mut words, &activity);
        assert_eq!(
            words.iter().map(|w| w.id.as_str()).collect::<Vec<_>>(),
            ["w8", "w10"]
        );
        assert!(!is_decay_repeat(&first, &second, &activity[..9]));
        assert!(!is_decay_repeat(
            &first,
            &word("w9", "Yes", 1990, 3270),
            &activity
        ));
        activity[8].level = 0.08;
        assert!(!is_decay_repeat(&first, &second, &activity));
        activity[8].level = 0.004;
        assert!(!is_decay_repeat(&first, &second, &activity));
    }
}
