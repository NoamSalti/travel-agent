import random
import re
import time

import streamlit as st
from google import genai
from google.genai import errors, types

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
PREFERRED_MODEL = st.secrets.get("MODEL_NAME", "gemini-3.8-flash")
MAX_ROUNDS = 4                 # each round tries every live model once
BASE_DELAY = 2.0               # seconds between rounds, doubles (2, 4, 8)
MAX_DELAY = 15.0
MAX_FALLBACK_MODELS = 6
DEAD_TTL = 30 * 60             # remember exhausted / unavailable models for 30 minutes
REQUEST_TIMEOUT_MS = 25_000    # a hung call fails fast instead of blocking for minutes
MAX_OUTPUT_TOKENS = 8192       # generous: "thinking" tokens count against this limit too

CREDIT = " תודה ל@noam_salti על כל הפרטים והתמונות המדהימות! אם יש שאלות, עקבו ותכתבו לה!"
CAROUSEL_HEADER = "הצעה לתמונת קרוסלה ראשונה:"

SYSTEM_INSTRUCTION = f"""
אתה קופירייטר מומחה לטיולים, הכותב פוסטים עבור קהילת "מטיילים מצייצים" (@travelers_tweets).
הטון: חם, מלהיב, אישי ונגיש, בעברית טבעית.

מבנה הפוסט (חובה, בסדר הזה):
1. Hook: משפט פתיחה שמושך את הקהל ויוצר סקרנות, בלי לציין את שם היעד.
2. חשיפה: מיד אחרי ה-Hook, חשוף את שם היעד.
3. תוכן מרכזי: הייחודיות, מה רואים (דגשים), איך מגיעים, עונה מומלצת, וטיפ זהב אחד לפחות.
   ניתן להוסיף עלויות אם הן ידועות. אל תמציא עובדות, מחירים או שעות פתיחה שאינך בטוח בהם.
4. CTA: משפט שגורר לפעולה (שמירה, שיתוף, תיוג חבר/ה לטיול).
5. קרדיט: סיים את גוף הפוסט בדיוק במשפט הבא, ללא שינוי:
   {CREDIT}

חובות נוספות:
* שלב אמוג'ים רלוונטיים לאורך הטקסט.
* אסור להכניס קישורים או כתובות אתר (URLs) בשום צורה.
* אחרי הקרדיט, הוסף בסוף תחת הכותרת "{CAROUSEL_HEADER}" רעיון לטקסט על התמונה הראשונה
  בקרוסלה, במבנה: "טקסט קטן:" ו"טקסט גדול:".
"""


# ----------------------------------------------------------------------------
# Client & model discovery
# ----------------------------------------------------------------------------
@st.cache_resource
def get_client():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


def discover_models(_client, preferred: str) -> list[str]:
    """Return an ordered list of text-capable Flash models this key can actually use.

    Hardcoded names cause 404s when access changes, so we ask the API what is
    available. The preferred model always goes first.
    """
    if st.session_state.get("models"):
        return st.session_state["models"]
    skip = ("image", "tts", "live", "audio", "embedding", "robotics", "computer-use", "native")
    found = []
    try:
        for m in _client.models.list():
            name = (m.name or "").replace("models/", "")
            actions = getattr(m, "supported_actions", None) or []
            if "generateContent" not in actions:
                continue
            if not ("flash" in name or name.startswith("gemma")) or any(s in name for s in skip):
                continue
            found.append(name)
    except Exception:
        pass  # fall back to the preferred model only

    # Order: stable Flash > Flash-Lite > Gemma > preview/experimental > "-latest" aliases.
    # Aliases go last because they usually point at the same overloaded model.
    def rank(n: str):
        return (
            n.endswith("-latest"),
            any(t in n for t in ("preview", "omni", "exp")),
            n.startswith("gemma"),
            "lite" in n,
            _neg(n),
        )

    found = sorted(set(found), key=rank)
    ordered = [preferred] + [n for n in found if n != preferred]
    ordered = ordered[: MAX_FALLBACK_MODELS + 1]
    # Only remember a successful listing, so a failed lookup is retried next time
    if found:
        st.session_state["models"] = ordered
    return ordered


def _neg(s: str) -> tuple:
    # sort strings descending inside a tuple sort key
    return tuple(-ord(c) for c in s)


# ----------------------------------------------------------------------------
# Error classification
# ----------------------------------------------------------------------------
def classify(e: errors.APIError) -> str:
    """Return one of: 'retry', 'rate', 'daily', 'skip', 'fatal'."""
    code = getattr(e, "code", None)
    msg = str(e)
    if code == 429:
        if "PerDay" in msg or "per day" in msg.lower():
            return "daily"      # daily quota gone: retrying this model is pointless
        return "rate"           # per-minute limit: wait, then retry
    if code in (500, 502, 503, 504):
        return "retry"
    if code in (400, 404):
        return "skip"           # model unavailable / doesn't support this request
    return "fatal"              # 401 / 403 etc. (bad key or permissions)


def suggested_wait(e: errors.APIError, attempt: int) -> float:
    m = re.search(r"retry in ([\d.]+)s", str(e), re.IGNORECASE) or re.search(
        r"'retryDelay': '(\d+)s'", str(e)
    )
    if m:
        return min(float(m.group(1)) + 1, 60.0)
    delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
    return delay + random.uniform(0, 1)  # jitter


# ----------------------------------------------------------------------------
# Generation with retry + fallback
# ----------------------------------------------------------------------------
def generate_post(client, destination: str, status, log: list) -> tuple[str, str]:
    """Round-robin over models: one try per live model per round, then back off.

    Demand spikes (503) usually hit one model at a time and last from seconds to
    a few minutes, so rotating models and waiting between rounds beats hammering
    a single model.
    """
    models = discover_models(client, PREFERRED_MODEL)
    log.append("מודלים שנבדקו: " + ", ".join(models))
    request = f"כתוב פוסט על היעד: {destination}"

    def build(model: str):
        kwargs = dict(
            temperature=0.9,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
        )
        if model.startswith("gemma"):
            # Gemma models don't accept a system instruction: put it in the prompt
            return f"{SYSTEM_INSTRUCTION}\n\n{request}", types.GenerateContentConfig(**kwargs)
        return request, types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION, **kwargs)

    # Models that were exhausted / unavailable recently are skipped for DEAD_TTL
    persisted = st.session_state.setdefault("dead_models", {})
    dead = {m for m, expiry in persisted.items() if expiry > time.time()}
    if all(m in dead for m in models):
        persisted.clear()
        dead = set()
    last_error = None

    for rnd in range(MAX_ROUNDS):
        live = [m for m in models if m not in dead]
        if not live:
            break
        wait_hint = 0.0
        for model in live:
            try:
                status.update(label=f"סבב {rnd + 1}/{MAX_ROUNDS}: מנסה את {model}...")
                contents, config = build(model)
                resp = client.models.generate_content(model=model, contents=contents, config=config)
                if resp.text and resp.text.strip():
                    finish = str(getattr(resp.candidates[0], "finish_reason", "")) if resp.candidates else ""
                    if "MAX_TOKENS" in finish:
                        # Truncated post: never show it, try another model instead
                        log.append(f"סבב {rnd + 1} | {model}: תשובה נחתכה (MAX_TOKENS)")
                        continue
                    return resp.text, model
                log.append(f"סבב {rnd + 1} | {model}: תשובה ריקה")
            except errors.APIError as e:
                last_error = e
                log.append(f"סבב {rnd + 1} | {model}: {e.code} {getattr(e, 'status', '')}")
                kind = classify(e)
                if kind == "fatal":
                    raise
                if kind in ("daily", "skip"):
                    dead.add(model)
                    persisted[model] = time.time() + DEAD_TTL
                elif kind == "rate":
                    wait_hint = max(wait_hint, suggested_wait(e, rnd))
            except Exception as e:  # timeouts / network hiccups: treat as transient
                last_error = e
                log.append(f"סבב {rnd + 1} | {model}: {type(e).__name__}")

        if rnd < MAX_ROUNDS - 1 and any(m not in dead for m in models):
            wait = max(wait_hint, min(BASE_DELAY * (2 ** rnd), MAX_DELAY) + random.uniform(0, 1))
            status.update(label=f"השרתים עמוסים, ממתין {wait:.0f} שניות ומנסה שוב...")
            time.sleep(wait)

    raise RuntimeError(
        "כל המודלים הזמינים עמוסים או שמכסת ה-API נוצלה. נסו שוב בעוד כמה דקות, "
        f"או מחר אם המכסה היומית נגמרה.\n\nשגיאה אחרונה: {last_error}"
    )


# ----------------------------------------------------------------------------
# Post-processing guarantees (don't trust the model blindly)
# ----------------------------------------------------------------------------
URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)


def enforce_rules(text: str) -> str:
    text = URL_RE.sub("", text)
    if CREDIT not in text:
        if CAROUSEL_HEADER in text:
            body, tail = text.split(CAROUSEL_HEADER, 1)
            text = f"{body.rstrip()}\n\n{CREDIT}\n\n{CAROUSEL_HEADER}{tail}"
        else:
            text = f"{text.rstrip()}\n\n{CREDIT}"
    return text.strip()


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Travel Content Agent", page_icon="✈️")
st.markdown(
    """
    <style>
    .rtl {direction: rtl; text-align: right;}
    textarea {direction: rtl;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("✈️ Travel Content Agent")
st.markdown("מחולל פוסטים אוטומטי בסגנון מטיילים מצייצים")

try:
    client = get_client()
except Exception:
    st.error("שגיאה: חסר מפתח API. יש להגדיר GEMINI_API_KEY בהגדרות השרת (secrets).")
    st.stop()

if "posts" not in st.session_state:
    st.session_state.posts = {}  # destination -> (text, model); avoids re-spending quota

destination = st.text_input(
    "לאיזה יעד נכתוב את הפוסט היום?",
    placeholder="לדוגמה: קאו סוק בתאילנד, נחל פרת, ג'ג'ו איילנד...",
).strip()

col1, col2 = st.columns([1, 1])
generate = col1.button("ייצר פוסט 🪄", type="primary")
regenerate = col2.button("ייצר מחדש 🔄", help="משתמש בעוד קריאת API")

if generate or regenerate:
    if not destination:
        st.warning("אנא הזן יעד כדי להתחיל.")
    else:
        key = destination.lower()
        if generate and key in st.session_state.posts:
            st.info("הפוסט כבר נוצר ליעד הזה, מציג אותו מהזיכרון (חוסך מכסה).")
        else:
            log: list = []
            with st.status("הסוכן כותב את הפוסט...", expanded=False) as status:
                try:
                    raw, used_model = generate_post(client, destination, status, log)
                    st.session_state.posts[key] = (enforce_rules(raw), used_model)
                    status.update(label="הפוסט מוכן!", state="complete")
                except errors.APIError as e:
                    status.update(label="שגיאה", state="error")
                    st.error(f"שגיאת API ({e.code}): {e.message}")
                except Exception as e:
                    status.update(label="שגיאה", state="error")
                    st.error(str(e))
            if log and key not in st.session_state.posts:
                with st.expander("פרטי הניסיונות (לדיבאג)"):
                    st.code("\n".join(log), language=None)

entry = st.session_state.posts.get(destination.lower()) if destination else None
if entry:
    post, used_model = entry
    st.success("הפוסט מוכן! 🤖")
    st.markdown(f'<div class="rtl">\n\n{post}\n\n</div>', unsafe_allow_html=True)
    st.text_area("להעתקה", post, height=300)
    st.caption(f"נוצר עם: {used_model}")
