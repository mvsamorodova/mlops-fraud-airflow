CREATE TABLE IF NOT EXISTS labeled_transactions (
    transaction_id TEXT PRIMARY KEY,
    transaction_time TIMESTAMP NOT NULL,
    merch TEXT, cat_id TEXT, amount DOUBLE PRECISION,
    name_1 TEXT, name_2 TEXT, gender TEXT, street TEXT,
    one_city TEXT, us_state TEXT, post_code INTEGER,
    lat DOUBLE PRECISION, lon DOUBLE PRECISION, population_city INTEGER,
    jobs TEXT, merchant_lat DOUBLE PRECISION, merchant_lon DOUBLE PRECISION,
    target SMALLINT
);
CREATE INDEX IF NOT EXISTS labeled_transactions_time_idx ON labeled_transactions(transaction_time);
CREATE TABLE IF NOT EXISTS training_state (
    id INTEGER PRIMARY KEY,
    window_start DATE NOT NULL,
    window_end DATE NOT NULL,
    next_batch_month DATE NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS pipeline_runs (
    kind TEXT NOT NULL,
    batch_month DATE NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    PRIMARY KEY(kind, batch_month)
);
CREATE TABLE IF NOT EXISTS run_requests (
    kind TEXT NOT NULL,
    run_id TEXT NOT NULL,
    batch_month DATE NOT NULL,
    PRIMARY KEY(kind, run_id),
    FOREIGN KEY(kind, batch_month) REFERENCES pipeline_runs(kind, batch_month)
);
