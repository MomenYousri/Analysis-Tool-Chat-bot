import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components
import google.generativeai as genai
import os
from dotenv import load_dotenv

# ==========================================
# 🔒 تحميل المتغيرات السرية بأمان من ملف .env
load_dotenv()
MY_API_KEY = os.getenv("GEMINI_API_KEY")
# ==========================================

# إعدادات صفحة الويب
st.set_page_config(page_title="المحلل الآلي الخارق", page_icon="🤖", layout="wide")

@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    return None

def detect_outliers(df, column):
    mean = df[column].mean()
    std = df[column].std()
    threshold = 3
    outliers = df[(df[column] - mean).abs() > threshold * std]
    return outliers

st.title("🤖 المحلل الآلي  + الخبير الذكي")
st.title("🤖 اداه المهندس مؤمن يسري ")
st.markdown("منصة متكاملة لتحليل البيانات، تنظيفها، واستجوابها بالذكاء الاصطناعي.")

uploaded_file = st.file_uploader("ارفع ملف البيانات (CSV / Excel)", type=['csv', 'xlsx'])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    if df is not None:
        st.success("✅ تم استلام البيانات بنجاح!")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 استكشاف سريع", 
            "📊 التقرير الشامل", 
            "🚨 صائد التشوهات", 
            "📈 لوحة الرسوم",
            "💬 اسأل المحلل الذكي"
        ])
        
        with tab1:
            st.subheader("عينة من البيانات")
            st.dataframe(df.head())
            
        with tab2:
            st.subheader("التقرير التحليلي الشامل")
            if st.button("إنشاء التقرير الشامل 🚀"):
                with st.spinner('جاري تحليل البيانات وبناء التقرير...'):
                    # الكود المحدث ليعمل بنجاح وبدون تعارضات
                    pr = ProfileReport(df, explorative=True, minimal=True)
                    report_html = pr.to_html()
                    components.html(report_html, height=1000, scrolling=True)
                    
        with tab3:
            st.subheader("اكتشاف القيم الشاذة")
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                target_col = st.selectbox("اختر العمود لفحصه:", numeric_cols)
                if st.button("ابحث عن التشوهات 🕵️‍♂️"):
                    outliers_df = detect_outliers(df, target_col)
                    if not outliers_df.empty:
                        st.warning(f"تم اكتشاف {len(outliers_df)} قيمة شاذة!")
                        st.dataframe(outliers_df)
                    else:
                        st.success("البيانات سليمة تماماً!")
            else:
                st.warning("لا توجد أعمدة رقمية للبحث فيها.")
                
        with tab4:
            st.subheader("الرسوم التفاعلية المخصصة")
            if len(numeric_cols) >= 2:
                col_x = st.selectbox("المحور السيني (X)", numeric_cols, index=0)
                col_y = st.selectbox("المحور الصادي (Y)", numeric_cols, index=1)
                fig = px.scatter(df, x=col_x, y=col_y, trendline="ols")
                st.plotly_chart(fig, use_container_width=True)
                
        with tab5:
            st.subheader("💬 استجوب بياناتك واطلب نصائح تجارية")
            user_question = st.text_area("اكتب سؤالك هنا للمحلل الآلي:")
            
            if st.button("إرسال السؤال للمحلل الذكي 🧠"):
                if not MY_API_KEY:
                    st.error("❌ لم يتم العثور على مفتاح الـ API. تأكد من إضافته في ملف .env")
                elif not user_question:
                    st.warning("⚠️ من فضلك اكتب سؤالاً أولاً.")
                else:
                    with st.spinner('جاري قراءة البيانات وصياغة الإجابة...'):
                        try:
                            # استخدام المفتاح المحمل من البيئة السرية
                            genai.configure(api_key=MY_API_KEY)
                            
                            # 💡 الحل الهندسي الذكي: البحث عن النماذج المتاحة لحسابك تلقائياً
                            available_models = [
                                m.name for m in genai.list_models() 
                                if 'generateContent' in m.supported_generation_methods
                            ]
                            
                            if not available_models:
                                st.error("❌ حسابك لا يحتوي على نماذج تدعم توليد النصوص حالياً.")
                            else:
                                # اختيار أول موديل مدعوم تلقائياً
                                smart_model_name = available_models[0]
                                model = genai.GenerativeModel(smart_model_name)
                                
                                data_summary = f"""
                                أنت خبير تحليل بيانات. أمامك ملخص لبيانات المستخدم، أجب على سؤاله بدقة.
                                عدد الصفوف: {df.shape[0]} | عدد الأعمدة: {df.shape[1]}
                                أنواع البيانات: {df.dtypes.to_string()}
                                الملخص الإحصائي: {df.describe().to_string()}
                                عينة من البيانات: {df.head().to_string()}
                                
                                سؤال المستخدم: {user_question}
                                """
                                response = model.generate_content(data_summary)
                                st.markdown(f"### 📋 إجابة الخبير الآلي (باستخدام {smart_model_name.split('/')[-1]}):")
                                st.write(response.text)
                                
                        except Exception as e:
                            st.error(f"❌ حدث خطأ. تفاصيل: {str(e)}")