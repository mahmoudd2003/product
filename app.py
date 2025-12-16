import os
import json
import time
import hashlib
import streamlit as st
import pandas as pd
from openai import OpenAI

# =========================
# PAGE
# =========================
st.set_page_config(page_title="مولّد عناوين وأوصاف المنتجات", layout="wide")
st.title("مولّد عناوين وأوصاف المنتجات")
st.caption("أدخل قائمة منتجات → يولّد العنوان والوصف مباشرة (بدون CSV)")

# =========================
# API KEY
# =========================
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("ضع OPENAI_API_KEY في Streamlit Secrets")
    st.stop()

client = OpenAI(api_key=api_key)

# =========================
# SETTINGS
# =========================
MODEL = "gpt-4.1-mini"
temperature = 0.7

SYSTEM_PROMPT = """أنت خبير محتوى منتجات لسوبرماركت عربي كبير.

المطلوب:
- إنشاء عنوان SEO-friendly ووصف عربي بشري لكل منتج.
- عدم استخدام جمل عامة مكررة مثل:
  "مناسب للاستخدام اليومي بمواصفات واضحة".
- لا تفترض ادعاءات (الأفضل، يعالج، يحسن الصحة...).
- الأسلوب: واضح، مباشر، عملي، مثل وصف سوبرماركت محترف.
"""

JSON_SCHEMA = {
    "name": "product_content",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "title": {"type": "string", "minLength": 10, "maxLength": 95},
            "description": {"type": "string", "minLength": 120, "maxLength": 900}
        },
        "required": ["title", "description"]
    }
}

# =========================
# HELPERS
# =========================
def norm(s):
    return str(s).strip() if s else ""

def stable_key(text: str):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def build_user_input(product_name: str) -> str:
    return f"""
اسم المنتج:
{product_name}

المطلوب:
1) title: عنوان عربي SEO-friendly بصيغة طبيعية (نوع المنتج + الماركة + الخاصية + الحجم إن وُجد).
2) description:
   - 2 إلى 4 جمل مفيدة ومحددة
   - ثم "الاستخدامات:" (3 نقاط)
   - ثم "المواصفات:" (نقاط مختصرة)
"""

def call_openai(product_name: str):
    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_input(product_name)},
        ],
        text={"format": {"type": "json_schema", "json_schema": JSON_SCHEMA}},
        temperature=temperature,
    )
    return json.loads(resp.output_text)

# =========================
# UI INPUT
# =========================
st.subheader("📝 أدخل قائمة المنتجات")
st.caption("كل سطر = منتج واحد")

products_text = st.text_area(
    "مثال:\nالمراعي حليب كامل الدسم 1 لتر\nنيفيا لوشن جسم ألوفيرا 400 مل",
    height=220
)

run = st.button("🚀 توليد العناوين والأوصاف")

# =========================
# PROCESS
# =========================
if run:
    products = [p.strip() for p in products_text.splitlines() if p.strip()]

    if not products:
        st.warning("أدخل منتجًا واحدًا على الأقل")
        st.stop()

    results = []
    cache = {}

    prog = st.progress(0.0)
    status = st.empty()

    total = len(products)

    for i, p in enumerate(products, start=1):
        key = stable_key(p)
        if key in cache:
            data = cache[key]
        else:
            data = call_openai(p)
            cache[key] = data

        results.append({
            "raw_name": p,
            "generated_title": data["title"],
            "generated_description": data["description"]
        })

        prog.progress(i / total)
        status.write(f"تمت معالجة {i}/{total}")

    df = pd.DataFrame(results)

    st.success("✅ تم توليد المحتوى")
    st.dataframe(df, use_container_width=True)

    # Optional download
    csv = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "⬇️ تنزيل النتائج CSV (اختياري)",
        data=csv,
        file_name="products_generated.csv",
        mime="text/csv",
    )
