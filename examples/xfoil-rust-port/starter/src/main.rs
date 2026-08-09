use std::io::{self, BufRead};

use serde_json::{json, Value};

fn response_for(request: &Value) -> Value {
    json!({
        "protocol": "transform/v1",
        "case_id": request
            .get("case_id")
            .and_then(Value::as_str)
            .unwrap_or(""),
        "status": "unsupported",
        "observations": {},
        "events": [],
        "output_files": {}
    })
}

fn main() {
    for line in io::stdin().lock().lines() {
        let Ok(line) = line else {
            std::process::exit(2);
        };
        if line.trim().is_empty() {
            continue;
        }
        let request: Value = match serde_json::from_str(&line) {
            Ok(value) => value,
            Err(_) => std::process::exit(2),
        };
        println!("{}", response_for(&request));
    }
}
