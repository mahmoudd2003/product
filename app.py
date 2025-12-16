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
st.title("مولّد عناوين وأوصاف المنتجات (قائمة مباشرة)")
st.caption("ألصق قائمة منتجات (كل سطر منتج) → توليد سريع Batch=30 → تنزيل النتائج")

# =========================
# API KEY
# =========================
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("ضع OPENAI_API_KEY في Streamlit Secrets: Manage app → Settings → Secrets")
    st.stop()

client = OpenAI(api_key=api_key)

# =========================
# SETTINGS
# =========================
MODEL = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0)
temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)

BATCH_SIZE = 30  # كما طلبت

SYSTEM_PROMPT = """
أنت خبير محتوى منتجات لسوبرماركت عربي كبير.

المطلوب:
- إنشاء عنوان SEO-friendly ووصف عربي بشري لكل منتج.
- ممنوع استخدام عبارات عامة مكررة مثل: "مناسب للاستخدام اليومي بمواصفات واضحة".
- لا تفترض ادعاءات غير مؤكدة (الأفضل، يعالج، يحسن الصحة...).
- الأسلوب: واضح، مباشر، عملي، مثل وصف سوبرماركت محترف.
- أعد النتائج كـ JSON فقط.
"""

# =========================
# STRUCTURED OUTPUT SCHEMA (Batch)
# =========================
BATCH_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "items": {
            "type": "array",
            "minItems": 1,
            "maxItems": 30,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "raw_name": {"type": "string", "minLength": 2, "maxLength": 300},
                    "title": {"type": "string", "minLength": 10, "maxLength": 95},
                    "description": {"type": "string", "minLength": 120, "maxLength": 900},
                },
                "required": ["raw_name", "title", "description"],
            },
        }
    },
    "required": ["items"],
}

# =========================
# HELPERS
# =========================
def stable_key(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def build_batch_user_input(product_names: list[str]) -> str:
    # نجعل الطلب واضح جدًا + يقلل التكرار
    joined = "\n".join([f"- {p}" for p in product_names])
    return f"""
هذه قائمة منتجات. أعد JSON فقط بالشكل:
{{
  "items": [
    {{"raw_name":"..","title":"..","description":".."}}
  ]
}}

القواعد:
1) title: عنوان عربي SEO-friendly بصيغة طبيعية (نوع + ماركة + خاصية + حجم إن وُجد).
2) description:
   - 2 إلى 4 جمل مفيدة ومحددة (بدون جمل عامة مكررة)
   - ثم "الاستخدامات:" (3 نقاط)
   - ثم "المواصفات:" (نقاط مختصرة)
3) لا ادعاءات غير مؤكدة.
4) اجعل الصياغة مختلفة قدر الإمكان بين المنتجات.

المنتجات:
{joined}
""".strip()

def call_openai_batch(product_names: list[str], retries: int = 6):
    user_input = build_batch_user_input(product_names)
    last_err = None

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
                        "schema": BATCH_SCHEMA,
                    }
                },
                temperature=temperature,
            )
            data = json.loads(resp.output_text)
            return data["items"]
        except Exception as e:
            last_err = e
            time.sleep(1.2 * (2 ** attempt))

    raise last_err

# =========================
# UI INPUT
# =========================
st.subheader("📝 أدخل قائمة المنتجات")
st.caption("كل سطر = منتج واحد. (السرعة تعتمد على عدد الأسطر)")

products_text = st.text_area(
    "Paste هنا",
    height=240,
    placeholder="مثال:\nالمراعي حليب كامل الدسم 1 لتر\nنيفيا لوشن جسم ألوفيرا 400 مل\n..."
)

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    run = st.button("🚀 توليد (Batch=30)")
with col2:
    limit = st.number_input("اختبار على أول N (0 = الكل)", min_value=0, value=30, step=10)
with col3:
    st.info("نصيحة: جرّب 30–60 منتج أولًا، ثم زد العدد تدريجيًا.")

# =========================
# RUN
# =========================
if run:
    products = [p.strip() for p in (products_text or "").splitlines() if p.strip()]
    if not products:
        st.warning("أدخل منتجًا واحدًا على الأقل.")
        st.stop()

    if limit and limit > 0:
        products = products[: int(limit)]

    # Cache داخل الجلسة
    if "cache" not in st.session_state:
        st.session_state["cache"] = {}
    cache = st.session_state["cache"]

    # نفصل المنتجات إلى:
    # - منتجات موجودة بالكاش
    # - منتجات جديدة تحتاج توليد
    results = []
    to_generate = []
    for p in products:
        k = stable_key(p)
        if k in cache:
            results.append({
                "raw_name": p,
                "generated_title": cache[k]["title"],
                "generated_description": cache[k]["description"],
            })
        else:
            to_generate.append(p)

    total = len(products)
    done = len(results)

    prog = st.progress(done / total if total else 0.0)
    status = st.empty()

    # توليد على دفعات 30
    try:
        for batch in chunk_list(to_generate, BATCH_SIZE):
            status.write(f"جاري توليد دفعة: {len(batch)} منتج...")

            items = call_openai_batch(batch)

            # نُرجع النتائج بنفس raw_name (في حال تغيّرت الترتيبات)
            for it in items:
                raw = (it.get("raw_name") or "").strip()
                title = (it.get("title") or "").strip()
                desc = (it.get("description") or "").strip()

                if not raw:
                    continue

                k = stable_key(raw)
                cache[k] = {"title": title, "description": desc}

                results.append({
                    "raw_name": raw,
                    "generated_title": title,
                    "generated_description": desc,
                })

                done += 1
                prog.progress(min(done / total, 1.0))
                status.write(f"تمت معالجة {done}/{total}")

        # ترتيب النتائج حسب ترتيب الإدخال الأصلي
        order = {p: i for i, p in enumerate(products)}
        results.sort(key=lambda x: order.get(x["raw_name"], 10**9))

        df = pd.DataFrame(results)
        st.success("✅ تم توليد العناوين والأوصاف بسرعة (Batch=30)")
        st.dataframe(df, use_container_width=True)

        csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "⬇️ تنزيل النتائج CSV",
            data=csv_bytes,
            file_name="products_generated.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error("❌ فشل استدعاء OpenAI. هذا هو الخطأ الحقيقي:")
        st.code(str(e))
        st.stop()
