import os
import json
import time
import hashlib

import pandas as pd
import streamlit as st
from openai import OpenAI

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="CSV → OpenAI → CSV (Titles & Descriptions)", layout="wide")
st.title("توليد عناوين وأوصاف المنتجات من CSV باستخدام OpenAI")

st.caption("ارفع ملف CSV → اختر الأعمدة → ولّد العنوان والوصف → نزّل CSV الناتج")

# ---------------------------
# Secrets / API Key
# ---------------------------
# Streamlit Community Cloud: ضع المفتاح في Secrets (Manage app → Settings → Secrets)
# محليًا: ضع .streamlit/secrets.toml ولا ترفعه على GitHub
api_key = None
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("مفتاح OPENAI_API_KEY غير موجود. أضفه في Streamlit Secrets أو كمتغير بيئة.")
    st.stop()

client = OpenAI(api_key=api_key)

# ---------------------------
# Settings
# ---------------------------
MODEL = st.selectbox("اختر الموديل", ["gpt-4.1-mini", "gpt-4.1"], index=0)
temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)

SYSTEM_PROMPT = """أنت خبير محتوى منتجات لسوبرماركت عربي كبير.
المطلوب: كتابة عنوان ووصف عربيين أصليين وقابلين للفهرسة.
ممنوع استخدام عبارات عامة نمطية ومتكررة مثل: "مناسب للاستخدام اليومي بمواصفات واضحة".
لا تفترض ادعاءات غير مؤكدة (مثل: الأفضل/الأجود/يعالج/يحسن الصحة) إن لم تكن موجودة في البيانات.
الأسلوب: واضح، بشري، مباشر، بدون مبالغة.
"""

JSON_SCHEMA = {
    "name": "product_content",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "title": {"type": "string", "minLength": 10, "maxLength": 95},
            "description": {"type": "string", "minLength": 140, "maxLength": 1300},
        },
        "required": ["title", "description"],
    },
}

# ---------------------------
# Helpers
# ---------------------------
def norm(x):
    return str(x).strip() if x is not None else ""

def stable_key(row: dict) -> str:
    base = "|".join([
        norm(row.get("raw_name") or row.get("name")),
        norm(row.get("brand")),
        norm(row.get("product_type")),
        norm(row.get("feature")),
        norm(row.get("size")),
        norm(row.get("unit")),
    ])
    return hashlib.md5(base.encode("utf-8")).hexdigest()

def build_user_input(row: dict) -> str:
    raw_name = norm(row.get("raw_name") or row.get("name"))

    parts = [f"اسم خام: {raw_name}"]

    optional_fields = [
        ("brand", "الماركة"),
        ("product_type", "نوع المنتج"),
        ("feature", "الخاصية/النكهة"),
        ("size", "الحجم"),
        ("unit", "الوحدة"),
        ("country", "بلد المنشأ"),
        ("storage", "التخزين"),
        ("shelf_life", "الصلاحية"),
        ("ingredients", "المكونات"),
        ("uses", "الاستخدامات (إن وُجدت)"),
    ]

    for k, label in optional_fields:
        v = norm(row.get(k))
        if v:
            parts.append(f"{label}: {v}")

    parts.append("""
المطلوب:
1) title: عنوان SEO-friendly بالعربية بصيغة طبيعية:
   (نوع المنتج + الماركة + الخاصية + الحجم/الوحدة) عند توفرها.
2) description:
   - 2 إلى 4 جمل مفيدة ومحددة (بدون جمل عامة مكررة)
   - ثم "الاستخدامات:" 3 نقاط (إن لم تتوفر، استنتج استخدامات منطقية بدون ادعاءات)
   - ثم "المواصفات:" نقاط قصيرة (الماركة/النوع/الحجم/المنشأ/التخزين إن توفر)
""")

    return "\n".join(parts).strip()

def call_openai_structured(user_input: str, max_retries: int = 6) -> dict:
    for attempt in range(max_retries):
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
                        "json_schema": JSON_SCHEMA
                    }
                },
                temperature=temperature,
            )
            return json.loads(resp.output_text)
        except Exception as e:
            wait = 1.5 * (2 ** attempt)
            time.sleep(wait)
    raise RuntimeError("Failed after retries")

# ---------------------------
# Upload CSV
# ---------------------------
uploaded = st.file_uploader("ارفع ملف CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded, encoding="utf-8-sig")
    st.success(f"تم تحميل الملف. عدد الصفوف: {len(df):,}")

    st.subheader("تحديد الأعمدة")
    cols = list(df.columns)

    col_name = st.selectbox("عمود اسم المنتج (raw_name أو name)", cols, index=0)
    col_brand = st.selectbox("عمود الماركة (اختياري)", [""] + cols, index=0)
    col_type = st.selectbox("عمود نوع المنتج (اختياري)", [""] + cols, index=0)
    col_feature = st.selectbox("عمود الخاصية/النكهة (اختياري)", [""] + cols, index=0)
    col_size = st.selectbox("عمود الحجم (اختياري)", [""] + cols, index=0)
    col_unit = st.selectbox("عمود الوحدة (اختياري)", [""] + cols, index=0)
    col_country = st.selectbox("عمود بلد المنشأ (اختياري)", [""] + cols, index=0)
    col_storage = st.selectbox("عمود التخزين (اختياري)", [""] + cols, index=0)
    col_uses = st.selectbox("عمود الاستخدامات (اختياري)", [""] + cols, index=0)

    st.subheader("تشغيل")
    limit = st.number_input("اختبار على أول N صف (0 = كل الملف)", min_value=0, value=20, step=10)
    batch_hint = st.info("نصيحة: ابدأ بـ 20–50 صف للتأكد من الجودة، ثم ارفع العدد تدريجيًا.")

    run = st.button("🚀 توليد العناوين والأوصاف")

    if run:
        work_df = df.copy()
        if limit and limit > 0:
            work_df = work_df.head(int(limit))

        # تجهيز أعمدة الخرج
        out_titles = []
        out_descs = []

        # كاش داخل الجلسة لتقليل الاستهلاك إذا تكررت منتجات
        if "cache" not in st.session_state:
            st.session_state["cache"] = {}
        cache = st.session_state["cache"]

        prog = st.progress(0)
        status = st.empty()

        total = len(work_df)
        for i, (_, r) in enumerate(work_df.iterrows(), start=1):
            row = {
                "raw_name": norm(r.get(col_name)),
                "brand": norm(r.get(col_brand)) if col_brand else "",
                "product_type": norm(r.get(col_type)) if col_type else "",
                "feature": norm(r.get(col_feature)) if col_feature else "",
                "size": norm(r.get(col_size)) if col_size else "",
                "unit": norm(r.get(col_unit)) if col_unit else "",
                "country": norm(r.get(col_country)) if col_country else "",
                "storage": norm(r.get(col_storage)) if col_storage else "",
                "uses": norm(r.get(col_uses)) if col_uses else "",
            }

            k = stable_key(row)
            if k in cache:
                data = cache[k]
            else:
                user_input = build_user_input(row)
                data = call_openai_structured(user_input)
                cache[k] = data

            out_titles.append(data["title"].strip())
            out_descs.append(data["description"].strip())

            prog.progress(i / total)
            status.write(f"تمت معالجة {i}/{total}")

        work_df["generated_title"] = out_titles
        work_df["generated_description"] = out_descs

        st.success("✅ انتهينا!")
        st.dataframe(work_df.head(20), use_container_width=True)

        out_csv = work_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "⬇️ تنزيل CSV الناتج",
            data=out_csv,
            file_name="products_out.csv",
            mime="text/csv",
        )

else:
    st.info("ارفع ملف CSV للبدء.")
