import streamlit as st
from google import genai
import time

st.set_page_config(page_title="Travel Content Agent", page_icon="✈️")
st.title("✈️ Travel Content Agent")
st.markdown("מחולל פוסטים אוטומטי בסגנון 'מטיילים מצייצים'")

try:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    st.error("שגיאה: חסר מפתח API. יש להגדיר GEMINI_API_KEY בהגדרות השרת.")
    st.stop()

destination = st.text_input("לאיזה יעד נכתוב את הפוסט היום?", placeholder="לדוגמה: קאו סוק בתאילנד, נחל פרת, ג'ג'ו איילנד...")

if st.button("ייצר פוסט 🪄"):
    if destination:
        with st.spinner("הסוכן כותב את הפוסט..."):
            prompt = f"""
            אתה קופירייטר מומחה לטיולים הכותב פוסטים עבור קהילת "מטיילים מצייצים".
            כתוב פוסט מרתק על היעד: {destination}.
            
            חובה לעמוד בחוקי הברזל הבאים:
            1. משפט פתיחה: חייב להיות משפט שמושך את הקהל (Hook) ויוצר סקרנות (ללא ציון שם המקום עדיין).
            2. חשיפה: מיד לאחר הפתיח, חשוף את שם המקום.
            3. תוכן מרכזי: הסבר מקיף שיכול לכלול את הייחודיות, מה רואים, איך מגיעים, עונה טובה, עלויות או טיפ זהב.
            4. הנעה לפעולה (CTA): הפוסט חייב להסתיים במשפט שגורר לפעולה.
            5. קרדיט: סיים תמיד עם המשפט הבא בדיוק: "אם יש שאלות, עקבו ותכתבו ל-@Noam_Salti! תודה ל@noam_salti על כל הפרטים והתמונות המדהימות!"
            
            נקודות חובה נוספות:
            * שלב אמוג'ים רלוונטיים לאורך הטקסט.
            * הפוסט אסור שיכיל קישורים (URLs).
            * הוסף בסוף הפוסט (תחת הכותרת "הצעה לתמונת קרוסלה ראשונה:") רעיון לטקסט שיופיע על התמונה הראשונה, במבנה של טקסט קטן וטקסט גדול.
            """
            
            # מנגנון ניסיון חוזר אוטומטי במקרה של עומס בשרתים (503)
            success = False
            for attempt in range(3):
                try:
                    response = client.models.generate_content(
                        model='gemini-3.8-flash',
                        contents=prompt
                    )
                    st.success("הפוסט מוכן! 🤖")
                    st.markdown(response.text)
                    success = True
                    break
                except Exception as e:
                    if "503" in str(e) and attempt < 2:
                        time.sleep(3)  # המתנה של 3 שניות וניסיון נוסף
                        continue
                    else:
                        st.error(f"התרחשה שגיאה מול ה-API: {e}")
                        break
    else:
        st.warning("אנא הזן יעד כדי להתחיל.")
