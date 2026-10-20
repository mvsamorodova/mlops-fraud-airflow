"""Проверка настоящего Compose-стенда через Airflow REST API и PostgreSQL."""

import base64
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4


def compose(*args):
    return subprocess.check_output(["docker", "compose", *args], text=True).strip()


def api(path, method="GET", data=None):
    request = Request("http://localhost:8081/api/v1/" + path,
                      data=None if data is None else json.dumps(data).encode(), method=method,
                      headers={"Authorization": "Basic " + base64.b64encode(b"admin:admin").decode(),
                               "Content-Type": "application/json"})
    with urlopen(request, timeout=20) as response:
        return json.load(response)


def wait_until(check, timeout=600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            result = check()
            if result:
                return result
        except (HTTPError, URLError, ConnectionError):
            pass
        time.sleep(3)
    raise TimeoutError("Проверка не завершилась вовремя")


def sql(statement):
    return compose("exec", "-T", "postgres", "psql", "-U", "fraud_user", "-d", "fraud_db", "-Atc", statement)


def trigger(dag):
    run_id = "manual__smoke_" + uuid4().hex[:12]
    api(f"dags/{dag}/dagRuns", "POST", {"dag_run_id": run_id})
    def done():
        run = api(f"dags/{dag}/dagRuns/{run_id}")
        if run["state"] == "failed":
            raise RuntimeError(f"DAG {dag} failed: {run_id}")
        return run if run["state"] == "success" else None
    result = wait_until(done)
    print(f"{dag}: success", flush=True)
    return result


def main():
    wait_until(lambda: len(api("dags")["dags"]) == 3)
    for dag in ("supply_labeled_batch", "retrain_catboost", "data_quality"):
        api(f"dags/{dag}", "PATCH", {"is_paused": False})
    wait_until(lambda: int(sql("SELECT count(*) FROM scores")) > 0)
    scorer_id = compose("ps", "-q", "fraud_detector")
    before = sql("SELECT count(*) FROM labeled_transactions")
    runs = [trigger("supply_labeled_batch"), trigger("retrain_catboost"), trigger("data_quality")]
    assert int(sql("SELECT count(*) FROM labeled_transactions")) > int(before)
    for _ in range(3):
        version = Path("artifacts/models/latest.txt").read_text().strip()
        if version != "initial":
            break
        runs.append(trigger("retrain_catboost"))
    assert version != "initial", "Ни один кандидат не принят"
    wait_until(lambda: int(sql(f"SELECT count(*) FROM scores WHERE model_version='{version}'")) > 0)
    assert compose("ps", "-q", "fraud_detector") == scorer_id
    assert subprocess.check_output(["docker", "inspect", "-f", "{{.RestartCount}}", scorer_id], text=True).strip() == "0"
    with urlopen("http://localhost:8082/dq/latest.html") as response:
        html = response.read().decode()
        assert "PASS" in html and "FAIL" not in html
    results = {"manual_runs": runs, "active_version": version, "hot_reload_without_restart": True,
               "scores": sql("SELECT model_version,count(*) FROM scores GROUP BY model_version"), "dq": "PASS"}
    os.environ["DAG_SCHEDULE"] = "*/1 * * * *"
    compose("up", "-d", "--no-deps", "--force-recreate", "airflow")
    wait_until(lambda: api("dags/retrain_catboost").get("schedule_interval") == {"__type": "CronExpression", "value": "*/1 * * * *"})
    def scheduled():
        found = {}
        for dag in ("supply_labeled_batch", "retrain_catboost", "data_quality"):
            data = api(f"dags/{dag}/dagRuns?limit=100&order_by=-execution_date")["dag_runs"]
            success = [run for run in data if run["run_type"] == "scheduled" and run["state"] == "success"]
            if not success:
                return None
            found[dag] = success[0]["dag_run_id"]
        return found
    results["scheduled_runs"] = wait_until(scheduled, timeout=900)
    assert compose("ps", "-q", "fraud_detector") == scorer_id
    os.environ["DAG_SCHEDULE"] = "manual"
    compose("up", "-d", "--no-deps", "--force-recreate", "airflow")
    wait_until(lambda: api("dags/retrain_catboost").get("schedule_interval") is None)
    wait_until(lambda: not any(run["state"] in {"running", "queued"}
                              for dag in ("supply_labeled_batch", "retrain_catboost", "data_quality")
                              for run in api(f"dags/{dag}/dagRuns?limit=100")["dag_runs"]))
    saved_state = sql("SELECT next_batch_month FROM training_state")
    saved_count = int(sql("SELECT count(*) FROM scores"))
    compose("down")
    compose("up", "-d", "--wait", "--wait-timeout", "600")
    assert sql("SELECT next_batch_month FROM training_state") == saved_state
    wait_until(lambda: int(sql("SELECT count(*) FROM scores")) > saved_count)
    results["restart_preserves_state_and_resumes_scoring"] = True
    Path("results").mkdir(exist_ok=True)
    Path("results/smoke.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
