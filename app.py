import os
import json
import time
import hashlib
import pandas as pd
import streamlit as st
from openai import OpenAI

# =========================
# PAGE
# =========================
st.set_page_config(page_title="Product Content Generator", layout="wide")
st.title("مولّد عناوين وأوصاف المنتجات (CSV + إدخال مباشر)")

st.caption("اختر طريقة الإدخال → ولّد العنوان والوصف → نزّل CSV الناتج")

# =========================
# API KEY (Streamlit Secrets)
# =========================
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY غير موجود. أضفه في Streamlit Secrets أو كمتغير بيئة.")
    st.stop()

client = OpenAI(api_key=api_key)

# =========================
# SETTINGS
# =========================
MODEL = st.selectbox("الموديل", ["gpt-4.1-mini", "gpt-4.1"], index=0)
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

# =========================
# HELPERS
# =========================
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
                text={"format": {"type": "json_schema", "json_schema": JSON_SCHEMA}},
                temperature=temperature,
            )
            return json.loads(resp.output_text)
        except Exception as e:
            wait = 1.5 * (2 ** attempt)
            time.sleep(wait)
    raise RuntimeError("Failed after retries")

def to_dataframe_from_list(lines: str) -> pd.DataFrame:
    items = []
    for ln in (lines or "").splitlines():
        ln = ln.strip()
        if ln:
            items.append({"raw_name": ln})
    return pd.DataFrame(items)

def to_dataframe_from_text_csv(csv_text: str) -> pd.DataFrame:
    # يعتمد على pandas لقراءة نص CSV من مربع النص
    import io
    return pd.read_csv(io.StringIO(csv_text), encoding="utf-8")

# =========================
# INPUT MODE
# =========================
mode = st.radio(
    "اختر طريقة إدخال المنتجات",
    ["إدخال مباشر (كل سطر منتج)", "لصق CSV كنص", "رفع ملف CSV"],
    horizontal=True
)

df = None

if mode == "إدخال مباشر (كل سطر منتج)":
    st.subheader("الصق المنتجات هنا")
    st.caption("كل سطر = اسم منتج خام. مثال: المراعي حليب كامل الدسم 1 لتر")
    text = st.text_area("قائمة المنتجات", height=220, placeholder="اكتب/الصق المنتجات هنا...")
    df = to_dataframe_from_list(text)

elif mode == "لصق CSV كنص":
    st.subheader("الصق محتوى CSV هنا")
    st.caption("الصق CSV كامل مع الهيدر (مثال: raw_name,brand,product_type,size,unit ...)")
    csv_text = st.text_area("CSV نصّي", height=260, placeholder="raw_name,brand,product_type,size,unit\n...")
    if csv_text.strip():
        try:
            df = to_dataframe_from_text_csv(csv_text)
        except Exception as e:
            st.error(f"خطأ في قراءة CSV النصي: {e}")
            df = None

else:
    uploaded = st.file_uploader("ارفع ملف CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded, encoding="utf-8-sig")

# =========================
# COLUMN MAPPING
# =========================
if df is not None:
    st.divider()
    st.success(f"عدد المنتجات المحمّلة: {len(df):,}")

    if len(df) == 0:
        st.warning("لا يوجد أي منتجات بعد. أضف منتجات ثم تابع.")
        st.stop()

    cols = list(df.columns)

    st.subheader("ربط الأعمدة (Column Mapping)")
    col_name = st.selectbox("عمود اسم المنتج", cols, index=cols.index("raw_name") if "raw_name" in cols else 0)

    def opt_col(label):
        return st.selectbox(label, [""] + cols, index=0)

    col_brand   = opt_col("عمود الماركة (اختياري)")
    col_type    = opt_col("عمود نوع المنتج (اختياري)")
    col_feature = opt_col("عمود الخاصية/النكهة (اختياري)")
    col_size    = opt_col("عمود الحجم (اختياري)")
    col_unit    = opt_col("عمود الوحدة (اختياري)")
    col_country = opt_col("عمود بلد المنشأ (اختياري)")
    col_storage = opt_col("عمود التخزين (اختياري)")
    col_uses    = opt_col("عمود الاستخدامات (اختياري)")

    st.subheader("التشغيل")
    limit = st.number_input("اختبار على أول N صف (0 = كل المنتجات)", min_value=0, value=min(20, len(df)), step=10)
    st.caption("ابدأ بـ 20–50 منتج للتأكد من الجودة ثم زد العدد تدريجيًا.")

    run = st.button("🚀 توليد العنوان والوصف")

    if run:
        work_df = df.copy()
        if limit and limit > 0:
            work_df = work_df.head(int(limit))

        titles, descs = [], []

        if "cache" not in st.session_state:
            st.session_state["cache"] = {}
        cache = st.session_state["cache"]

        prog = st.progress(0.0)
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
                data = call_openai_structured(build_user_input(row))
                cache[k] = data

            titles.append(data["title"].strip())
            descs.append(data["description"].strip())

            prog.progress(i / total)
            status.write(f"تمت معالجة {i}/{total}")

        work_df["generated_title"] = titles
        work_df["generated_description"] = descs

        st.success("✅ تم إنشاء العناوين والأوصاف!")
        st.dataframe(work_df.head(30), use_container_width=True)

        out_csv = work_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "⬇️ تنزيل CSV الناتج",
            data=out_csv,
            file_name="products_out.csv",
            mime="text/csv",
        )
