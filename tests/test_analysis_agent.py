from __future__ import annotations

import unittest
from types import SimpleNamespace as NS

from analysis_agent import analyze_file, validate_upload


class FakeFiles:
    def __init__(self) -> None:
        self.uploaded = None
        self.deleted = None

    def create(self, *, file, purpose):
        self.uploaded = (file.name, file.read(), purpose)
        return NS(id="file-test")

    def delete(self, file_id):
        self.deleted = file_id


class FakeResponses:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


class FakeClient:
    def __init__(self, response):
        self.files = FakeFiles()
        self.responses = FakeResponses(response)
        self.containers = NS(
            files=NS(
                content=NS(retrieve=lambda **kwargs: NS(read=lambda: b"PNG")),
                list=lambda container_id: [
                    NS(id="cfile-test", path="/mnt/data/analysis_chart.png")
                ],
            )
        )


def make_response(with_code=True, with_chart=False):
    calls = (
        [NS(type="code_interpreter_call", code="print(2 + 2)", container_id="cntr-test")]
        if with_code
        else []
    )
    annotations = (
        [
            NS(
                type="container_file_citation",
                container_id="cntr-test",
                file_id="cfile-test",
                filename="analysis_chart.png",
            )
        ]
        if with_chart
        else []
    )
    return NS(
        status="completed",
        output=calls + [NS(type="message", content=[NS(annotations=annotations)])],
        output_text="## Ключевые метрики\n4",
    )


class AnalysisAgentTests(unittest.TestCase):
    def test_rejects_invalid_uploads(self):
        with self.assertRaises(ValueError):
            validate_upload("data.txt", b"data")
        with self.assertRaises(ValueError):
            validate_upload("data.csv", b"")

    def test_sends_original_file_to_code_interpreter_and_returns_chart(self):
        client = FakeClient(make_response(with_chart=True))
        result = analyze_file("sales.csv", b"value\n4\n", "Сумма?", client=client)

        self.assertEqual(client.files.uploaded, ("sales.csv", b"value\n4\n", "user_data"))
        self.assertEqual(client.responses.kwargs["tool_choice"], "required")
        self.assertEqual(
            client.responses.kwargs["tools"][0]["container"]["file_ids"],
            ["file-test"],
        )
        self.assertEqual(result.code, ["print(2 + 2)"])
        self.assertEqual(result.charts, [("analysis_chart.png", b"PNG")])
        self.assertEqual(client.files.deleted, "file-test")

    def test_rejects_an_answer_without_python_execution(self):
        client = FakeClient(make_response(with_code=False))
        with self.assertRaisesRegex(RuntimeError, "Python"):
            analyze_file("sales.csv", b"value\n4\n", "Сумма?", client=client)
        self.assertEqual(client.files.deleted, "file-test")

    def test_finds_chart_when_model_omits_file_citation(self):
        client = FakeClient(make_response())
        result = analyze_file("sales.csv", b"value\n4\n", "Сумма?", client=client)
        self.assertEqual(result.charts, [("analysis_chart.png", b"PNG")])


if __name__ == "__main__":
    unittest.main()
