//! How a number written in digits is said aloud.
//!
//! One recognizer writes "200" where another writes "two hundred", and reconciliation has to
//! see one reading in both. Nothing here knows a language. The words for a number in every
//! gender, case and style ("two hundred and fifty", "twelve fifty", "dritten", "primera"),
//! the way digits are grouped, the ending an ordinal is written with and the name of a
//! currency or percent sign all come from ICU, the Unicode library that macOS ships.
use super::RECOGNIZER_LANGUAGE;
use super::reconcile::word_text;
use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_void};
use std::ptr;
use unicode_general_category::{GeneralCategory, get_general_category};
use unicode_normalization::UnicodeNormalization;

type Status = *mut i32;

#[link(name = "icucore")]
unsafe extern "C" {
    fn unum_open(
        style: i32,
        pattern: *const u16,
        pattern_length: i32,
        locale: *const c_char,
        parse_error: *mut c_void,
        status: Status,
    ) -> *mut c_void;
    fn unum_close(format: *mut c_void);
    fn unum_setAttribute(format: *mut c_void, attribute: i32, value: i32);
    fn unum_getTextAttribute(
        format: *const c_void,
        tag: i32,
        result: *mut u16,
        capacity: i32,
        status: Status,
    ) -> i32;
    fn unum_setTextAttribute(
        format: *mut c_void,
        tag: i32,
        value: *const u16,
        length: i32,
        status: Status,
    );
    fn unum_formatDouble(
        format: *const c_void,
        number: f64,
        result: *mut u16,
        capacity: i32,
        position: *mut c_void,
        status: Status,
    ) -> i32;
    fn unumf_openForSkeletonAndLocale(
        skeleton: *const u16,
        length: i32,
        locale: *const c_char,
        status: Status,
    ) -> *mut c_void;
    fn unumf_close(formatter: *mut c_void);
    fn unumf_openResult(status: Status) -> *mut c_void;
    fn unumf_closeResult(result: *mut c_void);
    fn unumf_formatDouble(
        formatter: *const c_void,
        value: f64,
        result: *mut c_void,
        status: Status,
    );
    fn unumf_resultToString(
        result: *const c_void,
        buffer: *mut u16,
        capacity: i32,
        status: Status,
    ) -> i32;
    fn ucurr_openISOCurrencies(kind: u32, status: Status) -> *mut c_void;
    fn uenum_next(enumeration: *mut c_void, length: *mut i32, status: Status) -> *const c_char;
    fn uenum_close(enumeration: *mut c_void);
    fn uloc_forLanguageTag(
        tag: *const c_char,
        locale: *mut c_char,
        capacity: i32,
        parsed: *mut i32,
        status: Status,
    ) -> i32;
    fn uloc_getLanguage(
        locale: *const c_char,
        language: *mut c_char,
        capacity: i32,
        status: Status,
    ) -> i32;
    fn uloc_getISOLanguages() -> *const *const c_char;
    fn uloc_getDisplayLanguage(
        locale: *const c_char,
        display_locale: *const c_char,
        result: *mut u16,
        capacity: i32,
        status: Status,
    ) -> i32;
    fn u_charDigitValue(character: i32) -> i32;
}

#[repr(C)]
struct Range {
    location: isize,
    length: isize,
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFStringCreateWithBytes(
        allocator: *const c_void,
        bytes: *const u8,
        length: isize,
        encoding: u32,
        external: u8,
    ) -> *const c_void;
    fn CFStringGetLength(string: *const c_void) -> isize;
    fn CFStringGetCString(
        string: *const c_void,
        buffer: *mut c_char,
        size: isize,
        encoding: u32,
    ) -> u8;
    fn CFStringTokenizerCopyBestStringLanguage(
        string: *const c_void,
        range: Range,
    ) -> *const c_void;
    fn CFRelease(object: *const c_void);
}

// ICU's names for these are UNUM_DECIMAL, UNUM_SPELLOUT, UNUM_ORDINAL, UNUM_DEFAULT_RULESET,
// UNUM_PUBLIC_RULESETS, UNUM_GROUPING_USED, UNUM_FRACTION_DIGITS,
// UNUM_MINIMUM_GROUPING_DIGITS, UCURR_COMMON | UCURR_NON_DEPRECATED and kCFStringEncodingUTF8.
const DECIMAL: i32 = 1;
const SPELL_OUT: i32 = 5;
const ORDINAL: i32 = 6;
const DEFAULT_RULE_SET: i32 = 6;
const PUBLIC_RULE_SETS: i32 = 7;
const GROUPING_USED: i32 = 1;
const FRACTION_DIGITS: i32 = 8;
const MINIMUM_GROUPING_DIGITS: i32 = 22;
const CURRENCIES_IN_USE: u32 = 1 | 8;
const UTF8: u32 = 0x0800_0100;

// A whole number above this loses its last digit in the arithmetic the rules do.
const LARGEST: f64 = 1e15;
// Text the system needs before its guess at a language can be relied on.
const DETECTABLE: usize = 200;
const SAMPLE: usize = 4_000;

/// Text that ICU writes into a buffer the caller provides.
fn text(mut fill: impl FnMut(*mut u16, i32, Status) -> i32) -> Option<String> {
    let mut buffer = vec![0_u16; 256];
    for _ in 0..2 {
        let mut status = 0;
        let capacity = i32::try_from(buffer.len()).ok()?;
        let length = usize::try_from(fill(buffer.as_mut_ptr(), capacity, &mut status)).ok()?;
        if length <= buffer.len() {
            return (status <= 0).then(|| String::from_utf16_lossy(&buffer[..length]));
        }
        buffer = vec![0; length + 1];
    }
    None
}

fn units(text: &str) -> Option<(Vec<u16>, i32)> {
    let units: Vec<u16> = text.encode_utf16().collect();
    let length = i32::try_from(units.len()).ok()?;
    Some((units, length))
}

fn digit(character: char) -> Option<char> {
    if character.is_ascii_digit() {
        return Some(character);
    }
    // SAFETY: the call takes any code point and has no other input.
    let value = unsafe { u_charDigitValue(character as i32) };
    char::from_digit(u32::try_from(value).ok()?, 10)
}

/// Lowercase text with the digits of any script as ASCII digits and without the width or
/// superscript variants of a character, so "１ｅｒ", "1ᵉʳ" and "1er" compare equal.
fn plain(text: &str) -> String {
    text.nfkc()
        .flat_map(char::to_lowercase)
        .map(|character| digit(character).unwrap_or(character))
        .collect()
}

fn is_currency(character: char) -> bool {
    get_general_category(character) == GeneralCategory::CurrencySymbol
}

/// One of ICU's number formatters.
struct Format {
    pointer: *mut c_void,
    chosen: Option<String>,
}

impl Format {
    fn open(style: i32, locale: &CStr) -> Option<Self> {
        let mut status = 0;
        // SAFETY: the locale is a terminated string, no pattern is passed, and the result is
        // checked before use.
        let pointer = unsafe {
            unum_open(
                style,
                ptr::null(),
                0,
                locale.as_ptr(),
                ptr::null_mut(),
                &mut status,
            )
        };
        (status <= 0 && !pointer.is_null()).then_some(Self {
            pointer,
            chosen: None,
        })
    }

    /// The rule sets a spell-out or ordinal formatter offers, in ICU's order.
    fn rule_sets(&self) -> Vec<String> {
        // SAFETY: the formatter is open and `text` passes a buffer with its capacity.
        text(|buffer, capacity, status| unsafe {
            unum_getTextAttribute(self.pointer, PUBLIC_RULE_SETS, buffer, capacity, status)
        })
        .map(|sets| {
            sets.split(';')
                .filter(|set| !set.is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
    }

    fn set(&self, attribute: i32, value: i32) {
        // SAFETY: the formatter is open.
        unsafe { unum_setAttribute(self.pointer, attribute, value) }
    }

    fn write(&self, value: f64) -> Option<String> {
        // SAFETY: the formatter is open and `text` passes a buffer with its capacity.
        text(|buffer, capacity, status| unsafe {
            unum_formatDouble(
                self.pointer,
                value,
                buffer,
                capacity,
                ptr::null_mut(),
                status,
            )
        })
    }

    /// A value as one rule set writes it.
    fn say(&mut self, value: f64, set: &str) -> Option<String> {
        if self.chosen.as_deref() != Some(set) {
            let (name, length) = units(set)?;
            let mut status = 0;
            // SAFETY: the formatter is open and the name is passed with its length.
            unsafe {
                unum_setTextAttribute(
                    self.pointer,
                    DEFAULT_RULE_SET,
                    name.as_ptr(),
                    length,
                    &mut status,
                );
            }
            self.chosen = None;
            if status > 0 {
                return None;
            }
            self.chosen = Some(set.to_owned());
        }
        self.write(value)
    }
}

impl Drop for Format {
    fn drop(&mut self) {
        // SAFETY: the formatter was opened by `unum_open` and is closed once.
        unsafe { unum_close(self.pointer) }
    }
}

/// The locale ICU knows a declared language by. A language is declared as a tag such as
/// "en-US", or by its English name, which is how a transcript labels it.
fn locale(language: &str) -> Option<(CString, String)> {
    let language = language.trim();
    let known = |code: &CStr| {
        // SAFETY: ICU returns a static list of terminated strings that ends in a null.
        unsafe {
            let mut entry = uloc_getISOLanguages();
            while !(*entry).is_null() {
                if CStr::from_ptr(*entry) == code {
                    return true;
                }
                entry = entry.add(1);
            }
        }
        false
    };
    let tag = CString::new(language.replace('_', "-")).ok()?;
    let mut locale = [0 as c_char; 96];
    let mut code = [0 as c_char; 16];
    let (mut parsed, mut status) = (0, 0);
    // SAFETY: both buffers are passed with their sizes, and ICU terminates what it writes
    // unless it reports that the buffer was too small.
    let tagged = unsafe {
        uloc_forLanguageTag(
            tag.as_ptr(),
            locale.as_mut_ptr(),
            95,
            &mut parsed,
            &mut status,
        );
        let complete = status == 0 && usize::try_from(parsed) == Ok(tag.as_bytes().len());
        uloc_getLanguage(locale.as_ptr(), code.as_mut_ptr(), 15, &mut status);
        (complete && status == 0).then(|| {
            (
                CStr::from_ptr(locale.as_ptr()).to_owned(),
                CStr::from_ptr(code.as_ptr()).to_owned(),
            )
        })
    };
    if let Some((locale, code)) = tagged
        && !code.is_empty()
        && known(&code)
    {
        return Some((locale, code.to_str().ok()?.to_owned()));
    }
    // SAFETY: as above, and every display name is written through `text`.
    unsafe {
        let mut entry = uloc_getISOLanguages();
        while !(*entry).is_null() {
            let name = text(|buffer, capacity, status| {
                uloc_getDisplayLanguage(*entry, c"en".as_ptr(), buffer, capacity, status)
            });
            if name.is_some_and(|name| name.to_lowercase() == language.to_lowercase()) {
                let code = CStr::from_ptr(*entry);
                return Some((code.to_owned(), code.to_str().ok()?.to_owned()));
            }
            entry = entry.add(1);
        }
    }
    None
}

/// The ISO code of a declared language: "en" for "en-US", "EN" and "English".
pub(super) fn language_code(language: &str) -> Option<String> {
    locale(language).map(|(_, code)| code)
}

/// The language the system reads in a sample of the words, when there is enough to tell.
fn detected(sample: &str) -> Option<CString> {
    if sample.chars().count() < DETECTABLE {
        return None;
    }
    // SAFETY: the bytes are valid UTF-8 of the given length, each object is released once,
    // and the code is copied into a buffer passed with its size.
    unsafe {
        let string = CFStringCreateWithBytes(
            ptr::null(),
            sample.as_ptr(),
            isize::try_from(sample.len()).ok()?,
            UTF8,
            0,
        );
        if string.is_null() {
            return None;
        }
        let range = Range {
            location: 0,
            length: CFStringGetLength(string),
        };
        let language = CFStringTokenizerCopyBestStringLanguage(string, range);
        CFRelease(string);
        if language.is_null() {
            return None;
        }
        let mut code = [0 as c_char; 32];
        let copied = CFStringGetCString(language, code.as_mut_ptr(), 32, UTF8);
        CFRelease(language);
        (copied != 0).then(|| CStr::from_ptr(code.as_ptr()).to_owned())
    }
}

/// A word written in digits and the ways its language says it.
#[derive(Debug, Default)]
pub(super) struct Figure {
    /// Each reading as `word_text` writes words. The plain one is first.
    readings: Vec<String>,
    /// What can follow a reading to name the sign the figure is written with: "dollars"
    /// after "$200", "percent" after "50%".
    names: Vec<String>,
    /// Written only as an ordinal, such as "3rd".
    pub(super) ordinal: bool,
}

impl Figure {
    /// Whether these words, written together, say the whole figure.
    pub(super) fn says(&self, spoken: &str) -> bool {
        self.readings.iter().any(|reading| {
            spoken
                .strip_prefix(reading.as_str())
                .is_some_and(|rest| rest.is_empty() || self.names.iter().any(|name| name == rest))
        })
    }

    /// Whether these words can go on to say the figure.
    pub(super) fn begins(&self, spoken: &str) -> bool {
        self.readings.iter().any(|reading| {
            reading.starts_with(spoken)
                || spoken
                    .strip_prefix(reading.as_str())
                    .is_some_and(|rest| self.names.iter().any(|name| name.starts_with(rest)))
        })
    }

    /// Whether a word is part of how the figure is said.
    pub(super) fn includes(&self, word: &str) -> bool {
        (self.readings.iter().chain(&self.names)).any(|said| said.contains(word))
    }
}

/// The figures of one session and what they are said as.
#[derive(Default)]
pub(super) struct Numbers {
    figures: HashMap<String, Figure>,
    /// The plain reading of the figure that each reading belongs to.
    spoken: HashMap<String, String>,
}

impl Numbers {
    /// Reads every word of a session that is written in digits. The language is the one
    /// declared, or else the one the words are in, or else the recognizer's.
    pub(super) fn of<'a>(
        language: Option<&str>,
        words: impl Iterator<Item = &'a str> + Clone,
    ) -> Self {
        let mut numbers = Self::default();
        let mut written: Vec<_> = words
            .clone()
            .filter(|word| word.chars().any(|character| digit(character).is_some()))
            .collect();
        written.sort_unstable();
        written.dedup();
        if written.is_empty() {
            return numbers;
        }
        let declared = language.and_then(locale).map(|(locale, _)| locale);
        let locale = declared.or_else(|| {
            let mut sample = String::new();
            for word in words {
                if sample.len() >= SAMPLE {
                    break;
                }
                sample += word.trim();
                sample.push(' ');
            }
            detected(&sample)
        });
        let recognizer = CString::new(RECOGNIZER_LANGUAGE).unwrap_or_default();
        let Some(mut rules) = Rules::open(&locale.unwrap_or(recognizer)) else {
            return numbers;
        };
        for word in written {
            let Some(figure) = rules.figure(word) else {
                continue;
            };
            for reading in &figure.readings {
                numbers
                    .spoken
                    .entry(reading.clone())
                    .or_insert_with(|| figure.readings[0].clone());
            }
            numbers.figures.insert(word.to_owned(), figure);
        }
        numbers
    }

    /// What a word of the session written in digits stands for.
    pub(super) fn figure(&self, text: &str) -> Option<&Figure> {
        self.figures.get(text)
    }

    /// The form under which a word meets its other spellings: the plain reading of a
    /// figure, for the figure and for each single word that says it another way. A word
    /// that is the plain reading itself has no other form.
    pub(super) fn alias(&self, text: &str) -> Option<String> {
        if let Some(figure) = self.figures.get(text) {
            return figure.readings.first().cloned();
        }
        let said = word_text(text);
        (self.spoken.get(&said).filter(|plain| **plain != said)).cloned()
    }
}

/// ICU's number rules for one language.
struct Rules {
    locale: CString,
    speller: Format,
    /// The rule sets that say a count, the plain one first.
    cardinals: Vec<String>,
    ordinals: Vec<String>,
    /// The rule sets that say a year.
    years: Vec<String>,
    decimal: Format,
    /// Writes an ordinal in digits: "3rd", "3.", "3.º", "3-й".
    endings: Option<Format>,
    ending_sets: Vec<String>,
    /// The start of a year said in two halves, with the rule set that says it so.
    halves: Option<Option<(String, String)>>,
    /// The currencies written with each sign.
    currencies: HashMap<char, Vec<String>>,
}

impl Rules {
    fn open(locale: &CStr) -> Option<Self> {
        let speller = Format::open(SPELL_OUT, locale)?;
        let decimal = Format::open(DECIMAL, locale)?;
        // Some languages leave a four-digit number ungrouped. A recognizer may not.
        decimal.set(MINIMUM_GROUPING_DIGITS, 1);
        let sets = speller.rule_sets();
        let (ordinals, counts): (Vec<_>, Vec<_>) =
            sets.into_iter().partition(|set| set.contains("ordinal"));
        let (years, mut cardinals): (Vec<_>, Vec<_>) =
            counts.into_iter().partition(|set| set.contains("year"));
        cardinals.extend(years.iter().cloned());
        let endings = Format::open(ORDINAL, locale);
        let ending_sets = endings.as_ref().map(Format::rule_sets).unwrap_or_default();
        (!cardinals.is_empty()).then_some(Self {
            locale: locale.to_owned(),
            speller,
            cardinals,
            ordinals,
            years,
            decimal,
            endings,
            ending_sets,
            halves: None,
            currencies: HashMap::new(),
        })
    }

    /// A value in each of the given rule sets, as `word_text` writes words. A rule set
    /// that falls back to digits says nothing.
    fn said(speller: &mut Format, sets: &[String], value: f64) -> Vec<String> {
        let mut readings = Vec::new();
        for set in sets {
            let reading = speller.say(value, set).map(|said| word_text(&said));
            if let Some(reading) = reading
                && !reading.is_empty()
                && !reading.chars().any(|character| digit(character).is_some())
                && !readings.contains(&reading)
            {
                readings.push(reading);
            }
        }
        readings
    }

    fn figure(&mut self, word: &str) -> Option<Figure> {
        let token = plain(word.trim());
        if token.chars().any(char::is_whitespace) {
            return None;
        }
        // A quote, a bracket or a currency sign can stand before the figure.
        let body = token.trim_start_matches(|character| {
            matches!(character, '"' | '\'')
                || matches!(
                    get_general_category(character),
                    GeneralCategory::OpenPunctuation
                        | GeneralCategory::InitialPunctuation
                        | GeneralCategory::FinalPunctuation
                        | GeneralCategory::CurrencySymbol
                )
        });
        let core = body.trim_end_matches(|character| !crate::transcript::is_word_char(character));
        if !core
            .chars()
            .next()
            .is_some_and(crate::transcript::is_word_char)
        {
            return None;
        }
        let mut figure = Figure::default();
        let mut value = None;
        if core.chars().any(char::is_alphabetic) {
            figure.readings = self.ordinal(core, core, true)?;
            figure.ordinal = true;
        } else if let Some((hour, minute)) = core.split_once(':') {
            figure.readings = self.clock(hour, minute)?;
        } else {
            let number = self.decimal(core)?;
            figure.readings = Self::said(&mut self.speller, &self.cardinals, number);
            // "3." is the third in German and can also be a three that ends a sentence.
            let mark = body[core.len()..].chars().next().map_or(0, char::len_utf8);
            if mark > 0
                && core.bytes().all(|byte| byte.is_ascii_digit())
                && let Some(ordinal) = self.ordinal(core, &body[..core.len() + mark], false)
            {
                figure.readings.extend(ordinal);
            }
            value = Some(number);
        }
        if let Some(value) = value {
            figure.names = self.names(&token, value);
        }
        (!figure.readings.is_empty()).then_some(figure)
    }

    /// The value of digits with the language's separators. Only what the language itself
    /// writes for a value counts, so "007" and, in English, "1,25" are not numbers.
    fn decimal(&self, core: &str) -> Option<f64> {
        let digits = |part: &str| -> String {
            (part.chars().filter(char::is_ascii_digit)).collect::<String>()
        };
        if digits(core).len() > 15 {
            return None;
        }
        let mut candidates = vec![(digits(core), 0)];
        if let Some((at, separator)) = (core.char_indices()).rfind(|(_, c)| !c.is_ascii_digit()) {
            let fraction = &core[at + separator.len_utf8()..];
            candidates.push((
                format!("{}.{fraction}", digits(&core[..at])),
                i32::try_from(fraction.len()).ok()?,
            ));
        }
        for (number, fraction) in candidates {
            let Ok(value) = number.parse::<f64>() else {
                continue;
            };
            self.decimal.set(FRACTION_DIGITS, fraction);
            for grouped in [0, 1] {
                self.decimal.set(GROUPING_USED, grouped);
                if (self.decimal.write(value)).is_some_and(|written| plain(&written) == core) {
                    return (value <= LARGEST).then_some(value);
                }
            }
        }
        None
    }

    /// The readings of an ordinal written in digits, when `written` is how the language
    /// writes the ordinal of the number in `core`. An ending with letters, as in "3rd", is
    /// compared as words are. An ending that is only a mark, as in "3.", has to be exact.
    fn ordinal(&mut self, core: &str, written: &str, lettered: bool) -> Option<Vec<String>> {
        let mut runs = core
            .split(|character: char| !character.is_ascii_digit())
            .filter(|run| !run.is_empty());
        let (Some(digits), None) = (runs.next(), runs.next()) else {
            return None;
        };
        let value = digits
            .parse::<f64>()
            .ok()
            .filter(|&value| value <= LARGEST)?;
        let endings = self.endings.as_mut()?;
        let found = self.ending_sets.iter().any(|set| {
            (endings.say(value, set)).is_some_and(|ordinal| {
                let ordinal = plain(&ordinal);
                if lettered {
                    word_text(&ordinal) != digits && word_text(&ordinal) == word_text(written)
                } else {
                    word_text(&ordinal) == digits && ordinal != digits && ordinal == written
                }
            })
        });
        let readings = Self::said(&mut self.speller, &self.ordinals, value);
        (found && !readings.is_empty()).then_some(readings)
    }

    /// A time of day is said as its two figures side by side.
    fn clock(&mut self, hour: &str, minute: &str) -> Option<Vec<String>> {
        let digits =
            |part: &str| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit());
        if !digits(hour) || !digits(minute) || minute.len() != 2 {
            return None;
        }
        let (hour, minute) = (hour.parse::<u32>().ok()?, minute.parse::<u32>().ok()?);
        if hour > 24 || minute > 59 || (hour == 24 && minute != 0) {
            return None;
        }
        let hours = Self::said(&mut self.speller, &self.cardinals, f64::from(hour));
        if minute == 0 {
            return Some(hours);
        }
        let mut minutes: Vec<_> = self.half(minute).into_iter().collect();
        for reading in Self::said(&mut self.speller, &self.cardinals, f64::from(minute)) {
            if !minutes.contains(&reading) {
                minutes.push(reading);
            }
        }
        Some(
            minutes
                .iter()
                .flat_map(|minute| hours.iter().map(move |hour| format!("{hour}{minute}")))
                .collect(),
        )
    }

    /// The second half of a year said in two halves, which is how such a language also
    /// says the minutes of an hour: "oh five" in "nineteen oh five" and "four oh five".
    /// A language reads a year so if the second half of 1930 is its plain thirty.
    fn half(&mut self, number: u32) -> Option<String> {
        if self.halves.is_none() {
            let plain = Self::said(&mut self.speller, &self.cardinals[..1], 30.0).pop();
            let mut found = None;
            for set in &self.years {
                let mut say =
                    |value: f64| self.speller.say(value, set).map(|said| word_text(&said));
                let (Some(century), Some(year)) = (say(19.0), say(1930.0)) else {
                    continue;
                };
                if !century.is_empty() && year.strip_prefix(century.as_str()) == plain.as_deref() {
                    found = Some((set.clone(), century));
                    break;
                }
            }
            self.halves = Some(found);
        }
        let (set, century) = self.halves.as_ref()?.as_ref()?;
        let year = self.speller.say(f64::from(1900 + number), set)?;
        let half = word_text(&year).strip_prefix(century.as_str())?.to_owned();
        (!half.is_empty()).then_some(half)
    }

    /// A value as a unit or currency pattern writes it, such as "5 percent".
    fn written(&self, value: f64, skeleton: &str) -> Option<String> {
        let (skeleton, length) = units(skeleton)?;
        let mut status = 0;
        // SAFETY: the skeleton is passed with its length and the locale is terminated. Both
        // objects are checked before use and closed once.
        unsafe {
            let formatter = unumf_openForSkeletonAndLocale(
                skeleton.as_ptr(),
                length,
                self.locale.as_ptr(),
                &mut status,
            );
            if formatter.is_null() {
                return None;
            }
            let result = unumf_openResult(&mut status);
            let mut written = None;
            if !result.is_null() {
                unumf_formatDouble(formatter, value, result, &mut status);
                if status <= 0 {
                    written = text(|buffer, capacity, status| {
                        unumf_resultToString(result, buffer, capacity, status)
                    });
                }
                unumf_closeResult(result);
            }
            unumf_close(formatter);
            written
        }
    }

    /// The currencies in use that the language writes with one sign.
    fn currencies(&mut self, sign: char) -> &[String] {
        if !self.currencies.contains_key(&sign) {
            let mut codes = Vec::new();
            let mut status = 0;
            // SAFETY: the enumeration is checked, read until it ends and closed once. Each
            // code is a terminated string that is copied before the next call.
            unsafe {
                let list = ucurr_openISOCurrencies(CURRENCIES_IN_USE, &mut status);
                while !list.is_null() && status <= 0 {
                    let code = uenum_next(list, ptr::null_mut(), &mut status);
                    if code.is_null() {
                        break;
                    }
                    codes.push(CStr::from_ptr(code).to_string_lossy().into_owned());
                }
                if !list.is_null() {
                    uenum_close(list);
                }
            }
            codes.retain(|code| {
                (self.written(1.0, &format!("currency/{code} unit-width-narrow"))).is_some_and(
                    |written| {
                        let mut signs = written.nfkc().filter(|&c| is_currency(c));
                        signs.next() == Some(sign) && signs.next().is_none()
                    },
                )
            });
            self.currencies.insert(sign, codes);
        }
        &self.currencies[&sign]
    }

    /// What can be said after a number for the sign it is written with. A name of several
    /// words is said in full, or by the words that open or close it: "dollars" for
    /// "US dollars", "dólares" for "dólares estadounidenses".
    fn names(&mut self, token: &str, value: f64) -> Vec<String> {
        let mut skeletons = Vec::new();
        for sign in token.chars().filter(|&character| is_currency(character)) {
            let codes = self.currencies(sign);
            skeletons.extend(codes.iter().map(|code| format!("currency/{code}")));
        }
        for unit in ["percent", "permille"] {
            let sign: String = (self.written(1.0, unit).iter())
                .flat_map(|written| plain(written).chars().collect::<Vec<_>>())
                .filter(|&c| digit(c).is_none() && !c.is_whitespace())
                .collect();
            if !sign.is_empty() && token.contains(&sign) {
                skeletons.push(unit.to_owned());
            }
        }
        // A whole amount is named as a count: "1 US dollar" and not "1.00 US dollars".
        let precision = if value.fract() == 0.0 {
            " precision-integer"
        } else {
            ""
        };
        let mut names = Vec::new();
        for skeleton in skeletons {
            let pattern = format!("{skeleton} unit-width-full-name{precision}");
            let Some(written) = self.written(value, &pattern) else {
                continue;
            };
            let words: Vec<_> = written
                .split_whitespace()
                .filter(|word| !word.chars().any(|character| digit(character).is_some()))
                .map(word_text)
                .filter(|word| !word.is_empty())
                .collect();
            for count in 1..=words.len() {
                for name in [
                    words[..count].concat(),
                    words[words.len() - count..].concat(),
                ] {
                    if !names.contains(&name) {
                        names.push(name);
                    }
                }
            }
        }
        names
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn figure(language: &str, word: &str) -> Option<Figure> {
        let (locale, _) = locale(language)?;
        Rules::open(&locale)?.figure(word)
    }

    fn readings(language: &str, word: &str) -> Vec<String> {
        figure(language, word).map_or_else(Vec::new, |figure| figure.readings)
    }

    #[test]
    fn a_figure_is_read_every_way_its_language_says_it() {
        let english = |word| readings("en", word);
        assert_eq!(english("200"), ["twohundred"]);
        assert!(english("350.").contains(&"threehundredandfifty".to_owned()));
        assert_eq!(english("1,250")[0], "onethousandtwohundredfifty");
        assert!(english("1,250").contains(&"twelvefifty".to_owned()));
        assert!(english("2024.").contains(&"twentytwentyfour".to_owned()));
        assert!(english("1905").contains(&"nineteenohfive".to_owned()));
        assert_eq!(english("12345")[0], "twelvethousandthreehundredfortyfive");
        assert_eq!(english("3.5"), ["threepointfive"]);
        assert_eq!(english("\u{663}\u{665}\u{660}")[0], "threehundredfifty");
        for word in ["007", "1,25", "1,2345", "v2", "-3", ".5", "3 5", "12a4"] {
            assert!(figure("en", word).is_none(), "{word}");
        }
    }

    #[test]
    fn a_time_of_day_is_read_as_its_two_figures() {
        let english = |word| readings("en", word);
        assert_eq!(english("4:00"), ["four"]);
        assert_eq!(english("4:30"), ["fourthirty"]);
        assert_eq!(english("4:05"), ["fourohfive", "fourfive"]);
        assert_eq!(readings("de", "4:30"), ["vierdreißig"]);
        assert_eq!(readings("de", "4:05"), ["vierfünf"]);
        for word in ["4:5", "25:00", "24:01", "4:60"] {
            assert!(figure("en", word).is_none(), "{word}");
        }
    }

    #[test]
    fn an_ordinal_is_read_by_the_ending_its_language_writes() {
        let ordinal = |language, word| figure(language, word).filter(|figure| figure.ordinal);
        assert_eq!(ordinal("en", "3rd.").unwrap().readings, ["third"]);
        assert_eq!(ordinal("en", "21st").unwrap().readings, ["twentyfirst"]);
        assert_eq!(ordinal("en", "100th").unwrap().readings, ["onehundredth"]);
        for word in ["-3rd", "3 rd", "3st", "21th", "3rd4"] {
            assert!(figure("en", word).is_none(), "{word}");
        }
        assert!(
            ordinal("fr", "1er")
                .unwrap()
                .readings
                .contains(&"premier".to_owned())
        );
        assert!(
            ordinal("es", "1.ª")
                .unwrap()
                .readings
                .contains(&"primera".to_owned())
        );
        assert!(
            ordinal("ru", "3-й")
                .unwrap()
                .readings
                .contains(&"третий".to_owned())
        );
        // A German "3." is the third, and it is a three before a full stop.
        let german = figure("de", "3.").unwrap();
        assert!(!german.ordinal);
        assert_eq!(german.readings[0], "drei");
        assert!(german.readings.contains(&"dritten".to_owned()));
        assert_eq!(readings("en", "3."), ["three"]);
    }

    #[test]
    fn every_language_reads_its_own_figures() {
        assert_eq!(readings("de", "250")[0], "zweihundertfünfzig");
        assert_eq!(readings("de", "1.250")[0], "eintausendzweihundertfünfzig");
        assert_eq!(readings("de", "1,25")[0], "einskommazweifünf");
        assert!(readings("de", "1").contains(&"eine".to_owned()));
        assert!(readings("es", "1").contains(&"una".to_owned()));
        assert_eq!(readings("es", "1.250")[0], "mildoscientoscincuenta");
        assert!(readings("es", "1.250").contains(&"mildoscientascincuenta".to_owned()));
        assert_eq!(readings("fr", "80")[0], "quatrevingts");
        assert_eq!(readings("ru", "250")[0], "двестипятьдесят");
        assert!(readings("ru", "250").contains(&"двухсотпятидесяти".to_owned()));
        assert_eq!(readings("ja", "250")[0], "二百五十");
    }

    #[test]
    fn a_sign_is_said_by_the_name_its_language_gives_it() {
        let dollars = figure("en", "$200").unwrap();
        assert!(dollars.says("twohundred"));
        assert!(dollars.says("twohundreddollars"));
        assert!(dollars.says("twohundredusdollars"));
        assert!(dollars.begins("twohundreddoll"));
        assert!(!dollars.says("twohundredpercent"));
        assert!(dollars.includes("dollars"));
        assert!(figure("en", "$1").unwrap().says("onedollar"));
        assert!(!figure("en", "$1").unwrap().says("onedollars"));
        assert!(figure("en", "$2.50").unwrap().says("twopointfivedollars"));
        assert!(figure("en", "50%").unwrap().says("fiftypercent"));
        assert!(figure("de", "50%").unwrap().says("fünfzigprozent"));
        assert!(figure("es", "200$").unwrap().says("doscientosdólares"));
        assert!(figure("fr", "200€").unwrap().says("deuxcentseuros"));
        assert!(!figure("en", "200").unwrap().says("twohundreddollars"));
    }

    #[test]
    fn a_language_is_known_by_its_tag_or_its_name() {
        for declared in ["en", "EN", "en-US", "en_us", "English", "english", "eng"] {
            assert_eq!(language_code(declared).as_deref(), Some("en"), "{declared}");
        }
        assert_eq!(language_code("de-CH").as_deref(), Some("de"));
        assert_eq!(language_code("German").as_deref(), Some("de"));
        for declared in ["", "auto", "klingonese", "en US"] {
            assert_eq!(language_code(declared), None, "{declared}");
        }
    }

    #[test]
    fn an_undeclared_language_is_read_from_the_words() {
        let german = "Wir haben ungefähr 250 Stunden gebraucht, verteilt auf drei oder vier \
            Monate, und am Ende haben wir deutlich mehr bezahlt, als irgendjemand von uns am \
            Anfang erwartet hatte, weil die Werkstatt so lange geschlossen war.";
        let numbers = Numbers::of(None, german.split(' '));
        assert_eq!(numbers.alias("250").as_deref(), Some("zweihundertfünfzig"));
        assert_eq!(numbers.alias("zweihundertfünfzig"), None);
        assert_eq!(numbers.spoken["zweihundertfünfzig"], "zweihundertfünfzig");
        // Too few words to tell: the recognizer's language.
        let numbers = Numbers::of(None, ["250", "Stunden"].into_iter());
        assert_eq!(numbers.alias("250").as_deref(), Some("twohundredfifty"));
        let numbers = Numbers::of(Some("auto"), ["250"].into_iter());
        assert_eq!(numbers.alias("250").as_deref(), Some("twohundredfifty"));
    }

    #[test]
    fn a_word_that_says_a_figure_of_the_session_meets_it() {
        let numbers = Numbers::of(Some("es"), ["1", "3.ª", "casa"].into_iter());
        assert_eq!(numbers.alias("1").as_deref(), Some("uno"));
        assert_eq!(numbers.alias(" una,").as_deref(), Some("uno"));
        assert_eq!(numbers.alias("3.ª"), numbers.alias("tercera"));
        assert_eq!(numbers.alias("casa"), None);
        assert_eq!(numbers.alias("dos"), None);
        assert!(
            Numbers::of(Some("es"), ["casa"].into_iter())
                .spoken
                .is_empty()
        );
    }
}
