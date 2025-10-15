import requests

requests.get("https://huggingface.co/api/tasks").raise_for_status()