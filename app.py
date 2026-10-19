import json
import re
from concurrent.futures import ThreadPoolExecutor

import ee
import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

# ======================= إعدادات يمكنك تعديلها =======================
PROJECT = "Desertification-map"          # Project ID مال Google Cloud
Y0, Y1 = 2000, 2026                      # سنوات التحليل
BASE = (2000, 2004)                      # فترة المقارنة المرجعية
TEAM = "BDULLAH BSHAR SALEH"
SUPERVISOR = "CRISTIANO RONALDO"
MARSH = [46.4, 30.4, 47.95, 31.9]        # مستطيل تقريبي للأهوار [غرب، جنوب، شرق، شمال]
# =====================================================================

st.set_page_config(page_title="مرصد التصحر في العراق", page_icon="🌍", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;800&display=swap');
html, body, [class*="st-"], .stMarkdown, button, input { font-family: 'Cairo', sans-serif; }
.stApp { background: linear-gradient(180deg,#f4f8ff 0%,#ffffff 40%); }
.block-container { padding-top: 1.2rem; max-width: 1400px; }
.stMarkdown, [data-testid="stSidebar"] label, [data-testid="stSidebar"] h3 { direction: rtl; text-align: right; }
.hero { background: linear-gradient(120deg,#0b1f4b 0%,#1e40af 55%,#0ea5a4 100%); color:#fff;
  padding: 26px 32px; border-radius: 20px; margin-bottom: 18px; direction: rtl;
  box-shadow: 0 10px 30px rgba(11,31,75,.25); }
.hero h1 { margin:0; font-size: 2.2rem; font-weight: 800; color:#fff; }
.hero .sub { color:#fde68a; font-weight:600; margin-top:6px; font-size:1.05rem; }
.hero .meta { color:#dbeafe; margin-top:10px; font-size:.9rem; }
.kpi { background:#fff; border-radius:16px; padding:16px 18px; direction: rtl; text-align:right;
  border-right: 6px solid var(--c); box-shadow: 0 4px 16px rgba(15,23,42,.08); }
.kpi .l { color:#64748b; font-size:.85rem; font-weight:600; }
.kpi .v { font-size:1.9rem; font-weight:800; color:#0f172a; line-height:1.3; }
.kpi .d { font-size:.85rem; font-weight:600; }
.legend { background:#fff; border-radius:14px; padding:12px 16px; margin-top:10px; direction: rtl;
  box-shadow: 0 4px 14px rgba(15,23,42,.07); }
.legend .t { font-weight:800; color:#0b1f4b; margin-bottom:6px; }
.legend .row { display:inline-flex; align-items:center; margin: 3px 14px 3px 0; font-size:.88rem; }
.legend .sw { width:18px; height:18px; border-radius:5px; margin-left:8px; border:1px solid #cbd5e1; display:inline-block; }
.card { background:#fff; border-radius:16px; padding:16px 20px; direction: rtl; text-align:right;
  box-shadow: 0 4px 16px rgba(15,23,42,.07); line-height:1.9; }
.stTabs [data-baseweb="tab"] { font-weight:700; font-size:1rem; }
</style>
""", unsafe_allow_html=True)


# ------------------------------ Earth Engine ------------------------------
@st.cache_resource
def init_ee():
    try:
        if "gee_service_account" in st.secrets:      # على الموقع الدائم
            info = dict(st.secrets["gee_service_account"])
            creds = ee.ServiceAccountCredentials(info["client_email"], key_data=json.dumps(info))
            ee.Initialize(creds, project=info.get("project_id", PROJECT))
            return True
    except Exception:
        pass
    ee.Initialize(project=PROJECT)                   # على جهازك
    return True


init_ee()

AR = {"anbar": "الأنبار", "basra": "البصرة", "muthann": "المثنى", "qadis": "القادسية", "najaf": "النجف",
      "arbil": "أربيل", "erbil": "أربيل", "sulay": "السليمانية", "tamim": "كركوك", "kirkuk": "كركوك",
      "babil": "بابل", "baghdad": "بغداد", "dahuk": "دهوك", "duhok": "دهوك", "dhi": "ذي قار",
      "diyala": "ديالى", "karbal": "كربلاء", "maysan": "ميسان", "ninaw": "نينوى",
      "salah": "صلاح الدين", "wasit": "واسط"}


def ar(name):
    n = re.sub("[^a-z]", "", name.lower())
    return next((v for k, v in AR.items() if k in n), name)


@st.cache_resource
def governorates():
    for ds in ("FAO/GAUL_SIMPLIFIED_500m/2015/level1", "FAO/GAUL/2015/level1"):
        fc = ee.FeatureCollection(ds).filter(ee.Filter.eq("ADM0_NAME", "Iraq"))
        names = fc.aggregate_array("ADM1_NAME").getInfo()
        if names:
            return fc, names
    raise RuntimeError("تعذر تحميل حدود المحافظات")


IRAQ, MARSH_KEY = "كل العراق", "الأهوار الجنوبية (تقريبي)"
REGIONS = [IRAQ, MARSH_KEY] + sorted({ar(n) for n in governorates()[1]})


def region(key):
    fc, names = governorates()
    if key == IRAQ:
        return fc.geometry()
    if key == MARSH_KEY:
        return ee.Geometry.Rectangle(MARSH)
    n = next(n for n in names if ar(n) == key)
    return fc.filter(ee.Filter.eq("ADM1_NAME", n)).geometry()


@st.cache_data(show_spinner=False)
def bounds(key):
    c = region(key).bounds().coordinates().getInfo()[0]
    xs, ys = [p[0] for p in c], [p[1] for p in c]
    return [[min(ys), min(xs)], [max(ys), max(xs)]]


# ------------------------------ بناء الصور ------------------------------
def _or_empty(coll, img, band):
    return ee.Image(ee.Algorithms.If(coll.size().gt(0), img, ee.Image(0).rename(band).selfMask()))


def ndvi(y):
    c = ee.ImageCollection("MODIS/061/MOD13A2").filterDate(f"{y}-01-01", f"{y+1}-01-01").select("NDVI")
    return _or_empty(c, c.max().multiply(0.0001).rename("NDVI"), "NDVI")


def mndwi(y):
    def f(i):
        g, s = i.select("sur_refl_b04"), i.select("sur_refl_b06")
        ok = i.select("StateQA").bitwiseAnd(3).eq(0).And(g.gt(0)).And(s.gt(0))
        return g.subtract(s).divide(g.add(s)).rename("MNDWI").updateMask(ok)
    c = ee.ImageCollection("MODIS/061/MOD09A1").filterDate(f"{y}-01-01", f"{y+1}-01-01").map(f)
    return _or_empty(c, c.median(), "MNDWI")


def lst(y):
    c = ee.ImageCollection("MODIS/061/MOD11A2").filterDate(f"{y}-06-01", f"{y}-10-01").select("LST_Day_1km")
    return _or_empty(c, c.mean().multiply(0.02).subtract(273.15).rename("LST"), "LST")


def rain(y):
    c = ee.ImageCollection("UCSB-CHG/CHIRPS/PENTAD").filterDate(f"{y-1}-10-01", f"{y}-10-01").select("precipitation")
    return _or_empty(c, c.sum().rename("RAIN"), "RAIN")


def avg(fn, years):
    return ee.ImageCollection([fn(y) for y in years]).mean()


def yrs(y, w=3):
    return range(max(Y0, y - w + 1), y + 1)


def base(fn):
    return avg(fn, range(BASE[0], BASE[1] + 1))


def risk(y):
    bn, cn = base(ndvi), avg(ndvi, yrs(y))
    bl, cl = base(lst), avg(lst, yrs(y))
    loss = bn.subtract(cn).divide(0.15).clamp(0, 1).multiply(65)
    heat = cl.subtract(bl).divide(3).clamp(0, 1).multiply(35)
    return loss.add(heat).updateMask(bn.gt(0.12)).rename("RISK")


def water_layer(y):
    m = mndwi(y)
    return m.updateMask(m.gt(-0.1))


CHANGE = "تغير الغطاء النباتي (بين سنتين)"
LAYERS = {
    "الغطاء النباتي (NDVI)": dict(
        fn=lambda y: ndvi(y),
        vis=dict(min=0, max=0.8, palette=["#8c510a", "#d8b365", "#f6e8a0", "#a6d96a", "#1a9850", "#00441b"]),
        legend=[("#8c510a", "أقل من 0.15", "صحراء وتربة جرداء"), ("#d8b365", "0.15 – 0.3", "نبات نادر ومتناثر"),
                ("#f6e8a0", "0.3 – 0.45", "شجيرات ومراعي"), ("#a6d96a", "0.45 – 0.6", "زراعة وغطاء متوسط"),
                ("#1a9850", "0.6 – 0.75", "زراعة مروية كثيفة"), ("#00441b", "أكثر من 0.75", "غطاء كثيف جداً (قصب وبساتين)")]),
    "المسطحات المائية": dict(
        fn=water_layer,
        vis=dict(min=-0.1, max=0.5, palette=["#c6dbef", "#6baed6", "#2171b5", "#08306b"]),
        legend=[("#c6dbef", "قريب من 0", "رطوبة / ماء ضحل / أهوار"), ("#6baed6", "0.1 – 0.25", "ماء مفتوح"),
                ("#08306b", "أكثر من 0.3", "ماء عميق (بحيرات وأنهار كبيرة)"), ("#ffffff", "شفاف", "يابسة")]),
    "حرارة سطح الأرض الصيفية": dict(
        fn=lambda y: lst(y),
        vis=dict(min=35, max=55, palette=["#313695", "#74add1", "#ffffbf", "#f46d43", "#a50026"]),
        legend=[("#313695", "أقل من 38°م", "بارد نسبياً (جبال / مياه)"), ("#74add1", "38 – 42°م", "معتدل"),
                ("#ffffbf", "42 – 46°م", "حار"), ("#f46d43", "46 – 50°م", "حار جداً"),
                ("#a50026", "أكثر من 52°م", "شديد (صحراء وتربة جرداء)")]),
    "مؤشر مخاطر التصحر": dict(
        fn=lambda y: risk(y),
        vis=dict(min=0, max=100, palette=["#1a9850", "#fee08b", "#fdae61", "#d73027", "#7f0000"]),
        legend=[("#1a9850", "0 – 20", "خطر منخفض جداً"), ("#fee08b", "20 – 40", "خطر منخفض"),
                ("#fdae61", "40 – 60", "خطر متوسط"), ("#d73027", "60 – 80", "خطر عالٍ"),
                ("#7f0000", "80 – 100", "خطر حرج"), ("#ffffff", "شفاف", "صحراء أصلاً (لا يُقاس عليها)")]),
    CHANGE: dict(
        fn=None,
        vis=dict(min=-0.2, max=0.2, palette=["#a50026", "#f46d43", "#ffffbf", "#78c679", "#006837"]),
        legend=[("#a50026", "−0.2 فأقل", "تدهور شديد"), ("#f46d43", "−0.1", "تدهور"), ("#ffffbf", "0", "مستقر"),
                ("#78c679", "+0.1", "تحسن"), ("#006837", "+0.2 فأكثر", "تحسن كبير")]),
}


@st.cache_data(ttl=3000, show_spinner=False)
def tile_url(key, layer, y, y2):
    L = LAYERS[layer]
    if layer == CHANGE:
        img = avg(ndvi, yrs(y2)).subtract(avg(ndvi, yrs(y)))
    else:
        img = L["fn"](y)
    return img.clip(region(key)).getMapId(L["vis"])["tile_fetcher"].url_format


def draw(key, layer, y, y2, k, h=560):
    m = folium.Map(tiles=None, control_scale=True)
    folium.TileLayer("CartoDB positron", name="خريطة فاتحة").add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="خريطة داكنة", show=False).add_to(m)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     attr="Esri", name="قمر صناعي", show=False).add_to(m)
    folium.TileLayer(tiles=tile_url(key, layer, y, y2), attr="Google Earth Engine", name=layer,
                     overlay=True, opacity=0.85).add_to(m)
    m.fit_bounds(bounds(key))
    folium.LayerControl(collapsed=True).add_to(m)
    st_folium(m, height=h, use_container_width=True, returned_objects=[], key=k)


# ------------------------------ الإحصاءات ------------------------------
def _one(geom, y):
    kw = dict(geometry=geom, bestEffort=True, maxPixels=1e10, tileScale=4)
    d1 = ndvi(y).addBands(lst(y)).reduceRegion(reducer=ee.Reducer.mean(), scale=1000, **kw)
    d2 = rain(y).reduceRegion(reducer=ee.Reducer.mean(), scale=5566, **kw)
    area = mndwi(y).gt(0).unmask(0).multiply(ee.Image.pixelArea()).rename("WATER")
    d3 = area.reduceRegion(reducer=ee.Reducer.sum(), scale=500, **kw)
    return ee.Dictionary(d1).combine(d2).combine(d3).getInfo()


@st.cache_data(ttl=7 * 86400, persist="disk", show_spinner=False)
def series(key):
    geom = region(key)

    def run(y):
        try:
            r = _one(geom, y)
        except Exception:
            r = {}
        w = r.get("WATER")
        return dict(year=y, ndvi=r.get("NDVI"), lst=r.get("LST"), rain=r.get("RAIN"),
                    water=None if w is None else w / 1e6)

    with ThreadPoolExecutor(max_workers=6) as ex:
        df = pd.DataFrame(list(ex.map(run, range(Y0, Y1 + 1)))).astype(float)
    if df["ndvi"].notna().sum() == 0:
        raise RuntimeError("لم تصل أي بيانات من Earth Engine")
    return df


@st.cache_data(ttl=7 * 86400, persist="disk", show_spinner=False)
def gov_table(y):
    fc, _ = governorates()
    img = (base(ndvi).rename("base").addBands(avg(ndvi, yrs(y)).rename("now"))
           .addBands(risk(y).rename("risk")))
    feats = img.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=1000, tileScale=4).getInfo()["features"]
    rows = [dict(name=ar(f["properties"]["ADM1_NAME"]), base=f["properties"].get("base"),
                 now=f["properties"].get("now"), risk=f["properties"].get("risk")) for f in feats]
    df = pd.DataFrame(rows).astype({"base": float, "now": float, "risk": float})
    df["delta"] = df["now"] - df["base"]
    return df


# ------------------------------ عناصر الواجهة ------------------------------
def kpi(label, value, delta, color):
    st.markdown(f'<div class="kpi" style="--c:{color}"><div class="l">{label}</div>'
                f'<div class="v">{value}</div><div class="d" style="color:{color}">{delta}</div></div>',
                unsafe_allow_html=True)


def legend(layer):
    rows = "".join(f'<span class="row"><span class="sw" style="background:{c}"></span>'
                   f'<b>{v}</b>&nbsp;←&nbsp;{t}</span>' for c, v, t in LAYERS[layer]["legend"])
    st.markdown(f'<div class="legend"><div class="t">🎨 دليل الألوان: {layer}</div>{rows}</div>',
                unsafe_allow_html=True)


def trend_chart(df, col, title, color, unit):
    d = df.dropna(subset=[col])
    p = np.polyfit(d.year, d[col], 1)
    fig = go.Figure()
    fig.add_scatter(x=d.year, y=d[col], mode="lines+markers", name=title, line=dict(color=color, width=3))
    fig.add_scatter(x=d.year, y=np.polyval(p, d.year), mode="lines", name="الاتجاه العام",
                    line=dict(color="#94a3b8", dash="dash"))
    fig.update_layout(title=title, height=330, margin=dict(l=10, r=10, t=50, b=10), template="plotly_white",
                      yaxis_title=unit, legend=dict(orientation="h", y=-0.2), font=dict(family="Cairo"))
    st.plotly_chart(fig)


def pct_trend(df, col):
    d = df.dropna(subset=[col])
    return np.polyfit(d.year, d[col], 1)[0] * 10 / d[col].mean() * 100


def change_txt(now, then, unit="%", absolute=False):
    if pd.isna(now) or pd.isna(then):
        return ""
    ch = now - then if absolute else (now - then) / then * 100
    return f"{'▲' if ch >= 0 else '▼'} {abs(ch):.1f}{unit} عن {Y0}"


# ------------------------------ الشريط الجانبي ------------------------------
with st.sidebar:
    st.markdown("### ⚙️ لوحة التحكم")
    key = st.selectbox("📍 المنطقة", REGIONS)
    layer = st.radio("🗂️ الطبقة", list(LAYERS))
    is_change = layer == CHANGE
    mode = "سنة واحدة" if is_change else st.radio("🖥️ نمط العرض", ["سنة واحدة", "مقارنة قبل وبعد"])
    pair = is_change or mode != "سنة واحدة"
    if pair:
        ya, yb = st.slider("📅 السنتان (من – إلى)", Y0, Y1, (Y0, Y1))
    else:
        y = st.slider("📅 السنة", Y0, Y1, Y1)
    st.caption("بيانات 2026 جزئية حتى آخر صورة متاحة.")

# ------------------------------ الواجهة الرئيسية ------------------------------
st.markdown(f"""
<div class="hero">
  <h1>🌍 مرصد التصحر في العراق</h1>
  <div class="sub">رصد وتحليل {Y1 - Y0 + 1} عاماً ({Y0} – {Y1}) من بيانات الأقمار الصناعية</div>
  <div class="meta">جامعة المستقبل · الكلية التقنية الهندسية · مسابقة التميز التقني — مسار أفضل محلل بيانات<br>
  إعداد: {TEAM} · إشراف: {SUPERVISOR}</div>
</div>""", unsafe_allow_html=True)

try:
    with st.spinner("جارٍ تحليل بيانات الأقمار الصناعية... (أول مرة فقط، بعدها فوري)"):
        df = series(key)
except Exception as e:
    st.error(f"تعذر جلب البيانات من Earth Engine: {e}")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["🗺️ الخريطة التفاعلية", "📈 الاتجاهات", "🏛️ المحافظات", "🧪 المنهجية"])

# ---------- تبويب الخريطة ----------
with tab1:
    ky = yb if pair else y
    r, r0 = df[df.year == ky].iloc[0], df.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("🌿 الغطاء النباتي (NDVI)", f"{r.ndvi:.3f}", change_txt(r.ndvi, r0.ndvi), "#16a34a")
    with c2:
        kpi("💧 المسطحات المائية (كم²)", f"{r.water:,.0f}", change_txt(r.water, r0.water), "#2563eb")
    with c3:
        kpi("🔥 حرارة الصيف (°م)", f"{r.lst:.1f}", change_txt(r.lst, r0.lst, "°م", True), "#dc2626")
    with c4:
        kpi("🌧️ الأمطار (ملم/سنة مائية)", f"{r.rain:,.0f}", change_txt(r.rain, r0.rain), "#7c3aed")
    st.write("")

    if is_change:
        st.markdown(f"**التغير في الغطاء النباتي بين {ya} و{yb}** (متوسط 3 سنوات لكل طرف)")
        draw(key, layer, ya, yb, "m_change")
    elif pair:
        a, b = st.columns(2)
        with a:
            st.markdown(f"#### 📅 {ya}")
            draw(key, layer, ya, ya, "m_a", 520)
        with b:
            st.markdown(f"#### 📅 {yb}")
            draw(key, layer, yb, yb, "m_b", 520)
    else:
        draw(key, layer, y, y, "m_single")
    legend(layer)

# ---------- تبويب الاتجاهات ----------
with tab2:
    d = df.dropna(subset=["ndvi"])
    best, worst = d.loc[d.ndvi.idxmax()], d.loc[d.ndvi.idxmin()]
    corr = d[["ndvi", "rain"]].dropna().corr().iloc[0, 1]
    st.markdown(f"""<div class="card"><b>💡 أبرز النتائج ({key})</b><br>
    • الاتجاه العام للغطاء النباتي: <b>{pct_trend(df, 'ndvi'):+.1f}%</b> لكل عقد.<br>
    • أعلى اخضرار سنة <b>{int(best.year)}</b> (NDVI = {best.ndvi:.3f})، وأدنى اخضرار سنة <b>{int(worst.year)}</b> (NDVI = {worst.ndvi:.3f}).<br>
    • اتجاه حرارة الصيف: <b>{np.polyfit(df.dropna(subset=['lst']).year, df.dropna(subset=['lst']).lst, 1)[0] * 10:+.2f}°م</b> لكل عقد.<br>
    • الارتباط بين الأمطار والغطاء النباتي: <b>{corr:.2f}</b> {'(قوي: الأمطار هي المحرك الأساسي)' if abs(corr) > 0.6 else '(متوسط/ضعيف: عوامل أخرى كالري وإدارة المياه مؤثرة)'}.
    </div>""", unsafe_allow_html=True)
    st.write("")
    a, b = st.columns(2)
    with a:
        trend_chart(df, "ndvi", "الغطاء النباتي (ذروة NDVI السنوية)", "#16a34a", "NDVI")
        trend_chart(df, "lst", "حرارة سطح الأرض الصيفية", "#dc2626", "°م")
    with b:
        trend_chart(df, "water", "مساحة المسطحات المائية (دائمة/شبه دائمة)", "#2563eb", "كم²")
        trend_chart(df, "rain", "الأمطار (السنة المائية أكتوبر–سبتمبر)", "#7c3aed", "ملم")
    st.download_button("⬇️ تحميل البيانات (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"desertification_{key}.csv", mime="text/csv")

# ---------- تبويب المحافظات ----------
with tab3:
    gy = st.slider("سنة المقارنة مع الفترة المرجعية", Y0 + 5, Y1, Y1, key="gy")
    with st.spinner("جارٍ حساب المحافظات..."):
        g = gov_table(gy)
    gg = g.dropna(subset=["risk"]).sort_values("risk")
    fig = go.Figure(go.Bar(x=gg.risk, y=gg.name, orientation="h", marker=dict(
        color=gg.risk, colorscale=[[0, "#1a9850"], [.4, "#fee08b"], [.7, "#d73027"], [1, "#7f0000"]], cmin=0, cmax=100)))
    fig.update_layout(title=f"مؤشر مخاطر التصحر حسب المحافظة ({gy})", height=520, template="plotly_white",
                      margin=dict(l=10, r=10, t=50, b=10), font=dict(family="Cairo"), xaxis_title="درجة الخطر (0–100)")
    st.plotly_chart(fig)
    show = g.sort_values("delta")[["name", "base", "now", "delta", "risk"]].round(3)
    show.columns = ["المحافظة", f"NDVI ({BASE[0]}–{BASE[1]})", f"NDVI (آخر 3 سنوات حتى {gy})", "التغير", "درجة الخطر"]
    st.dataframe(show, hide_index=True)

# ---------- تبويب المنهجية ----------
with tab4:
    st.markdown(f"""<div class="card">
<b>🛰️ مصادر البيانات (مجانية ومفتوحة عبر Google Earth Engine)</b><br>
• <b>NASA MODIS MOD13A2</b>: مؤشر الغطاء النباتي NDVI (دقة 1 كم) — نستخدم <b>أعلى قيمة في السنة</b> لتمثيل ذروة الاخضرار.<br>
• <b>NASA MODIS MOD09A1</b>: مؤشر المياه MNDWI (دقة 500 م) — نستخدم <b>الوسيط السنوي</b>؛ المساحة المائية = البكسلات التي فيها ماء أكثر من نصف السنة.<br>
• <b>NASA MODIS MOD11A2</b>: حرارة سطح الأرض النهارية لأشهر يونيو–سبتمبر.<br>
• <b>CHIRPS</b>: الأمطار (دقة 5 كم) لكل سنة مائية (أكتوبر–سبتمبر).<br>
• <b>FAO GAUL</b>: حدود العراق والمحافظات.<br><br>
<b>🧮 مؤشر مخاطر التصحر (0–100)</b><br>
الدرجة = 65% × (فقدان NDVI مقارنة بالفترة {BASE[0]}–{BASE[1]}، مقيّساً حتى 0.15) + 35% × (ارتفاع حرارة الصيف، مقيّساً حتى 3°م).<br>
نستخدم متوسط آخر 3 سنوات لتقليل أثر السنوات الاستثنائية، ونستثني المناطق الصحراوية أصلاً (NDVI المرجعي أقل من 0.12).
المؤشر <b>استرشادي</b> لترتيب المناطق وليس قياساً رسمياً للتصحر.<br><br>
<b>⚠️ حدود الدراسة</b><br>
• دقة MODIS (500 م – 1 كم) مناسبة للتحليل الوطني لكنها تُقلل دقة المساحات المائية الصغيرة، ومياه الأهوار المغطاة بالقصب قد تُحتسب أقل من حقيقتها.<br>
• مستطيل الأهوار تقريبي ويشمل أراضي محيطة.<br>
• التغيرات تعكس الأمطار والري وإدارة المياه وليس التغير المناخي وحده.<br>
• بيانات {Y1} جزئية.
</div>""", unsafe_allow_html=True)

st.markdown("<div style='text-align:center;color:#94a3b8;margin-top:24px;direction:rtl'>"
            "جامعة المستقبل · الكلية التقنية الهندسية · مسابقة التميز التقني</div>", unsafe_allow_html=True)