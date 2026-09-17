import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set the page layout to wide for better graph visibility
st.set_page_config(page_title="학생 디지털 생활 EDA", layout="wide")

st.title("학생의 디지털 생활과 학업 성취도의 관계")
st.markdown("디지털 습관이 기말고사 성적에 미치는 영향을 분석합니다.")

# 1. Load and cache the data
@st.cache
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'student_digital_life.csv')

    df = pd.read_csv(file_path) # Ensure this file is in the same directory
    df_cleaned = df.drop(columns=['student_id']).dropna()
    
    # Outlier Removal
    Q1 = df_cleaned['final_exam_score'].quantile(0.25)
    Q3 = df_cleaned['final_exam_score'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df_no_outliers = df_cleaned[(df_cleaned['final_exam_score'] >= lower_bound) & 
                                (df_cleaned['final_exam_score'] <= upper_bound)]
    return df_no_outliers

df = load_data()

# 2. Layout: Correlation Heatmap
st.header("1. 전체 데이터 상관관계 매트릭스")
st.markdown("**핵심 요약:** 학습 시간은 성적에 뚜렷한 긍정적 효과를 주며, 디지털 기기 사용은 전반적으로 부정적인 경향을 나타냅니다.")

fig_heat, ax_heat = plt.subplots(figsize=(12, 8))
numerical_cols = df.select_dtypes(include=[np.number])
sns.heatmap(numerical_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, ax=ax_heat)
st.pyplot(fig_heat)

st.markdown("---")

# 3. Layout: Scatter Plot Grid
st.header("2. 디지털 기기 사용이 성적에 미치는 영향")
st.markdown("**관찰 결과:** 게임 및 스마트폰 사용 시간은 소셜 미디어나 스트리밍 시청에 비해 기말고사 성적에 더 확실한 부정적인 영향을 미칩니다.")

digital_features = ['smartphone_usage_hours', 'social_media_hours', 'gaming_hours', 'streaming_hours']
fig_scatter, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, feature in enumerate(digital_features):
    sns.regplot(data=df, x=feature, y='final_exam_score', 
                scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'}, ax=axes[i])
    axes[i].set_title(f'{feature.replace("_", " ").title()} vs Final Exam Score')
    axes[i].set_xlabel(feature.replace("_", " ").title())
    axes[i].set_ylabel('Final Exam Score')

plt.tight_layout()
st.pyplot(fig_scatter)