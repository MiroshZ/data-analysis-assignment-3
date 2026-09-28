"""Streamlit UI for the data analysis agent."""

from __future__ import annotations

import os
from io import BytesIO

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from analysis_agent import analyze_file, validate_upload
from groq_agent import DEFAULT_GROQ_MODEL, analyze_file_groq


load_dotenv()
st.set_page_config(page_title="Данные → выводы", page_icon="📊", layout="wide")


def get_api_key(name: str) -> str:
    try:
        return st.secrets.get(name, os.getenv(name, ""))
    except Exception:
        return os.getenv(name, "")


def preview(filename: str, content: bytes) -> pd.DataFrame:
    data = BytesIO(content)
    if filename.lower().endswith(".xlsx"):
        return pd.read_excel(data, nrows=10)
    for encoding in ("utf-8-sig", "cp1251"):
        try:
            return pd.read_csv(data, sep=None, engine="python", encoding=encoding, nrows=10)
        except UnicodeDecodeError:
            data.seek(0)
    raise ValueError("Не удалось прочитать кодировку CSV.")


st.title("Данные → выводы")
st.caption("Загрузите таблицу, задайте вопрос и получите анализ, вычисленный ИИ-агентом в Python.")

with st.container(border=True):
    provider = st.radio(
        "API для анализа",
        ["Groq (бесплатный план)", "OpenAI"],
        horizontal=True,
    )
    if provider.startswith("Groq"):
        st.caption("Для Groq таблица после преобразования должна помещаться в 10 КБ текста.")
    uploaded = st.file_uploader("CSV или Excel (.xlsx), до 15 МБ", type=["csv", "xlsx"])
    question = st.text_area(
        "Что узнать из данных?",
        value="Какие главные тенденции, различия и проблемы качества данных?",
        height=90,
    )
    run = st.button("Проанализировать", type="primary", disabled=uploaded is None)

if uploaded is not None:
    raw = uploaded.getvalue()
    try:
        safe_name = validate_upload(uploaded.name, raw)
    except Exception as exc:
        st.error(f"Некорректный файл: {exc}")
        st.stop()
    st.subheader("Предпросмотр")
    try:
        st.dataframe(preview(safe_name, raw), width="stretch", hide_index=True)
        st.caption(f"Файл: {safe_name} · {len(raw) / 1024:.1f} КБ · показаны первые 10 строк")
    except Exception as exc:
        st.warning(f"Предпросмотр недоступен: {exc}. Агент всё равно попробует прочитать файл.")

if run and uploaded is not None:
    is_groq = provider.startswith("Groq")
    key_name = "GROQ_API_KEY" if is_groq else "OPENAI_API_KEY"
    key = get_api_key(key_name)
    if not key:
        st.error(f"Не задан {key_name}. Добавьте его в .env или секреты Streamlit.")
        st.stop()
    with st.spinner("Агент читает данные, запускает Python и проверяет выводы…"):
        try:
            if is_groq:
                result = analyze_file_groq(
                    safe_name,
                    raw,
                    question,
                    model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
                    api_key=key,
                )
            else:
                result = analyze_file(
                    safe_name,
                    raw,
                    question,
                    model=os.getenv("OPENAI_MODEL", "gpt-4.1"),
                    client=OpenAI(api_key=key, timeout=180.0, max_retries=2),
                )
        except Exception as exc:
            st.error(f"Анализ не выполнен: {exc}")
        else:
            st.session_state["analysis_result"] = result
            st.session_state["analysis_file"] = safe_name
            st.session_state["analysis_provider"] = provider

result = st.session_state.get("analysis_result")
if (
    result
    and st.session_state.get("analysis_file") == (uploaded.name if uploaded else None)
    and st.session_state.get("analysis_provider") == provider
):
    st.divider()
    st.subheader("Результат анализа")
    st.markdown(result.markdown)
    for chart_name, chart_bytes in result.charts:
        st.image(chart_bytes, caption=chart_name, width="stretch")
        st.download_button(
            "Скачать график",
            data=chart_bytes,
            file_name=chart_name,
            mime="image/png",
            key=f"download_{chart_name}",
        )
    with st.expander("Как агент считал результат"):
        st.caption(f"Модель: {result.model} · вызовов Python: {len(result.code)}")
        for index, code in enumerate(result.code, 1):
            st.markdown(f"**Вызов {index}**")
            st.code(code, language="python")
    st.download_button(
        "Скачать отчёт Markdown",
        data=result.markdown.encode("utf-8"),
        file_name="analysis_report.md",
        mime="text/markdown",
    )
