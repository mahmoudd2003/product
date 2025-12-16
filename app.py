import os
import json
import time
import hashlib
import streamlit as st
import pandas as pd
from openai import OpenAI

st.set_page_config(page_title="مولّد عناوين وأوصاف المنتجات", layout="wide")
st.title("مولّد عناوين وأوصاف المنتجات (قائمة مباشرة)")
st.caption("أدخل قائمة منتجات (كل سطر منتج) ← يولّد العنوان والوصف مباشرة")

# =========================
# API KEY
# =========================
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("ضع OPENAI_API_KEY في Streamlit Secrets (Manage app → Settings → Secrets).")
    st.stop()

client = OpenAI(api_key=api_key)

# =========================
# SETTINGS
# =========================
MODEL = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0)
temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)

SYSTEM_PROMPT = """
أنت خبير محتوى منتجات لسوبرماركت عربي كبير.

المطلوب:
- إنشاء عنوان SEO-friendly ووصف عربي بشري لكل منتج.
- ممنوع استخدام عبارات عامة مكررة مثل: "مناسب للاستخدام اليومي بمواصفات واضحة".
- لا تفترض ادعاءات غير مؤكدة (الأفضل، يعالج، يحسن الصحة...).
- الأسلوب: واضح، مباشر، عملي، مثل وصف سوبرماركت محترف.

أعد الناتج كـ JSON فقط.
"""

# schema خام (Structured Outputs)
JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "title": {"type": "string", "minLength": 10, "maxLength": 95},
        "description": {"type": "string", "minLength": 120, "maxLength": 900},
    },
    "required": ["title", "description"],
}

def stable_key(text: str):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def build_user_input(product_name: str) -> str:
    return f"""
اسم المنتج:
{product_name}

أرجع JSON فقط بهذا الشكل:
{{
  "title": "…",
  "description": "…"
}}

القواعد:
1) title: عنوان عربي SEO-friendly بصيغة طبيعية (نوع + ماركة + خاصية + حجم إن وجد).
2) description:
   - 2 إلى 4 جمل مفيدة ومحددة (بدون جمل عامة مكررة)
   - ثم "الاستخدامات:" (3 نقاط)
   - ثم "المواصفات:" (نقاط مختصرة)
"""

def call_openai(product_name: str, retries: int = 6):
    """
    1) جرّب Structured Outputs (json_schema)
    2) إذا فشل بسبب دعم/صيغة، استخدم JSON mode كبديل (json_object)
    مع عرض الخطأ الحقيقي في الواجهة.
    """
    last_err = None
    user_input = build_user_input(product_name)

    for attempt in range(retries):
        try:
            resp = client.responses.create(
                model=MODEL,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_input},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "strict": True,
                        "schema": JSON_SCHEMA,
                    }
                },
                temperature=temperature,
            )
            return json.loads(resp.output_text)

        except Exception as e:
            last_err = e
            time.sleep(1.2 * (2 ** attempt))

    # Fallback: JSON mode
    for attempt in range(retries):
        try:
            resp = client.responses.create(
                model=MODEL,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_input},
                ],
                text={"format": {"type": "json_object"}},
                temperature=temperature,
            )
            return json.loads(resp.output_text)
        except Exception as e:
            last_err = e
            time.sleep(1.2 * (2 ** attempt))

    raise last_err

# =========================
# UI
# =========================
st.subheader("📝 أدخل قائمة المنتجات")
products_text = st.text_area(
    "كل سطر = منتج واحد",
    height=220,
    placeholder="مثال:\nالمراعي حليب كامل الدسم 1 لتر\nنيفيا لوشن جسم ألوفيرا 400 مل"
)

col1, col2 = st.columns([1, 1])
with col1:
    run = st.button("🚀 توليد")
with col2:
    st.info("ابدأ بـ 5–20 منتج للتأكد من الإعدادات.")

if run:
    products = [p.strip() for p in (products_text or "").splitlines() if p.strip()]
    if not products:
        st.warning("أدخل منتجًا واحدًا على الأقل.")
        st.stop()

    results = []
    cache = {}

    prog = st.progress(0.0)
    status = st.empty()

    total = len(products)
    for i, product in enumerate(products, start=1):
        key = stable_key(product)

        try:
            if key in cache:
                data = cache[key]
            else:
                data = call_openai(product)
                cache[key] = data

            results.append({
                "raw_name": product,
                "generated_title": data.get("title", "").strip(),
                "generated_description": data.get("description", "").strip(),
            })

        except Exception as e:
            st.error("❌ فشل استدعاء OpenAI. هذا هو الخطأ الحقيقي (كما ورد من السيرفر):")
            st.code(str(e))
            st.stop()

        prog.progress(i / total)
        status.write(f"تمت معالجة {i}/{total}")

    df = pd.DataFrame(results)
    st.success("✅ تم توليد النتائج")
    st.dataframe(df, use_container_width=True)

    csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "⬇️ تنزيل النتائج CSV",
        data=csv_bytes,
        file_name="products_generated.csv",
        mime="text/csv",
    )
