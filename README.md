# Данные → выводы

Небольшое веб-приложение для анализа CSV и Excel. Пользователь загружает таблицу,
задаёт вопрос и получает метрики, график, наблюдения и замечания по качеству данных.

## Как работает агент

1. Streamlit принимает исходный файл и показывает только его предпросмотр.
2. Приложение загружает **полный файл** через OpenAI Files API.
3. Responses API запускает модель с инструментом `code_interpreter` и контейнером,
   в который передан ID файла. `tool_choice="required"` требует вызова инструмента.
4. Модель сама пишет и выполняет Python-код для чтения файла, расчётов и графика.
5. Приложение проверяет наличие `code_interpreter_call`, показывает ответ и
   выполненный код. Если модель сослалась на PNG-график, приложение скачивает его
   из контейнера и показывает на странице.

Метрики **не вычисляются приложением заранее и не подставляются в промпт**. Код
предпросмотра не участвует в аналитическом ответе.

## Запуск

Требуются Python 3.10+ и API-ключ OpenAI с доступом к Responses API и Code
Interpreter. Вызовы API и контейнер Code Interpreter оплачиваются по тарифам
вашего аккаунта.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

В `.env` укажите реальный `OPENAI_API_KEY`. При необходимости измените
`OPENAI_MODEL`; по умолчанию используется `gpt-4.1`.

```bash
streamlit run app.py
```

Откройте адрес, который выдаст Streamlit (обычно `http://localhost:8501`). Для
проверки можно загрузить [sample_data/sales.csv](sample_data/sales.csv) и спросить:
«Как менялась выручка по месяцам и регионам?». Приложение принимает CSV и `.xlsx`
до 15 МБ. Файл отправляется в OpenAI API; не загружайте чувствительные данные без
соответствующего разрешения.

## Проверки

```bash
python -m unittest discover -s tests -v
python -m compileall -q app.py analysis_agent.py
```

Тесты проверяют передачу исходного файла агенту, обязательный вызов Python,
загрузку графика и отказ от ответа без вызова инструмента. Полный API-сценарий
нужно проверить с действующим ключом.

## Развёртывание и сдача

Для публичного доступа удобно использовать Streamlit Community Cloud: загрузите
репозиторий на GitHub, укажите `app.py` как главный файл и добавьте
`OPENAI_API_KEY` в Secrets приложения. Ключ нельзя коммитить в репозиторий.
После развёртывания приложите к сдаче URL приложения и URL GitHub-репозитория.
Если публичное размещение недоступно, приложите к отчёту скриншоты результата
реального запуска с ключом, включая развёрнутый блок «Как агент считал результат».

## Источники API

- [OpenAI Code Interpreter](https://developers.openai.com/api/docs/guides/tools-code-interpreter)
- [OpenAI File inputs](https://developers.openai.com/api/docs/guides/file-inputs)
- [Развёртывание в Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
