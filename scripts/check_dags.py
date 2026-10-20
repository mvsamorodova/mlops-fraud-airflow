"""Проверка импорта и зависимостей DAG внутри Airflow."""

from airflow.models import DagBag

bag = DagBag("/opt/airflow/dags", include_examples=False)
assert not bag.import_errors, bag.import_errors
expected = {"retrain_catboost": 7, "supply_labeled_batch": 3, "data_quality": 4}
assert set(bag.dags) == set(expected)
for name, count in expected.items():
    dag = bag.dags[name]
    assert len(dag.tasks) == count
    assert not dag.catchup and dag.max_active_runs == 1
    assert sum(len(task.downstream_task_ids) for task in dag.tasks) == count - 1
print("DAG imports and dependencies: OK")
