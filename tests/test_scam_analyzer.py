import os
import unittest
from unittest.mock import patch

import requests

import scam_analyzer


VALID_CONTENT = (
    '{"score":10,"risk":"low","confidence":99,'
    '"reasons":["a","b","c"],"advice":"ok",'
    '"signals":{"urgency":false,"threat":false,'
    '"financial_request":false,"link_present":false,"impersonation":false}}'
)


class _OkResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": VALID_CONTENT}}]}


class _UnsupportedThinkingResponse:
    status_code = 400
    text = '{"error":"unsupported parameter: chat_template_kwargs"}'


class _UnsupportedThinkingError(requests.exceptions.HTTPError):
    def __init__(self):
        super().__init__("bad request")
        self.response = _UnsupportedThinkingResponse()


class AnalyzeWithLlmTests(unittest.TestCase):
    def setUp(self):
        os.environ["LLM_API_KEY"] = "test-key"

    def test_retries_without_thinking_when_parameter_is_unsupported(self):
        calls = []

        def _post(*args, **kwargs):
            calls.append(kwargs["json"])
            if kwargs["json"].get("chat_template_kwargs"):
                raise _UnsupportedThinkingError()
            return _OkResponse()

        with patch("scam_analyzer.requests.post", side_effect=_post):
            result = scam_analyzer.analyze_with_llm("hello")

        self.assertEqual(result["score"], 10)
        self.assertEqual(len(calls), 2)
        self.assertIn("chat_template_kwargs", calls[0])
        self.assertNotIn("chat_template_kwargs", calls[1])

    def test_does_not_retry_for_unrelated_bad_request(self):
        class _OtherBadResponse:
            status_code = 400
            text = '{"error":"unsupported parameter: temperature"}'

        class _OtherBadError(requests.exceptions.HTTPError):
            def __init__(self):
                super().__init__("bad request")
                self.response = _OtherBadResponse()

        with patch("scam_analyzer.requests.post", side_effect=_OtherBadError()):
            with self.assertRaises(RuntimeError) as raised:
                scam_analyzer.analyze_with_llm("hello")

        self.assertIn("status 400", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
