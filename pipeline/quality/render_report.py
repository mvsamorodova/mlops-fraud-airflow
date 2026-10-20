"""HTML-отчёт сохраняется до проверки статуса DAG."""

from html import escape
from pipeline.artifacts import atomic_text
from pipeline.settings import REPORTS_DIR


def render_report(result):
    """Читает результаты правил, пишет HTML и latest.html; повтор атомарно заменяет отчёт."""
    rows = "".join(
        f"<tr><td>{escape(row['rule'])}</td><td>{'PASS' if row['passed'] else 'FAIL'}</td>"
        f"<td>{row['violations']}</td><td>{escape(row['details'])}</td></tr>" for row in result["checks"]
    )
    html = f"""<!doctype html><html lang="ru"><meta charset="utf-8"><title>Data Quality</title>
<style>body{{font:16px system-ui;max-width:1000px;margin:40px auto}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ddd;padding:10px;text-align:left}}</style>
<h1>Data Quality</h1><p>Запуск: {escape(result['run_id'])}</p><p>Строк: {result['rows']}</p>
<table><tr><th>Правило</th><th>Статус</th><th>Нарушений</th><th>Описание</th></tr>{rows}</table></html>"""
    path = REPORTS_DIR / "dq" / f"{result['key']}.html"
    atomic_text(path, html)
    atomic_text(REPORTS_DIR / "dq" / "latest.html", html)
    return {"path": str(path), "passed": all(row["passed"] for row in result["checks"])}


def enforce_quality(report):
    """Читает итог отчёта, завершает задачу ошибкой при FAIL; данные не изменяет."""
    if not report["passed"]:
        raise ValueError(f"Data Quality FAIL: {report['path']}")
    return report["path"]
