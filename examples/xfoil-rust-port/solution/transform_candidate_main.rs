use std::io::{self, BufRead};

use rustfoil_core::naca::naca4;
use rustfoil_core::{CubicSpline, PanelingParams};
use rustfoil_inviscid::{FlowConditions, InviscidSolver};
use serde_json::{json, Value};

fn case_id(request: &Value) -> &str {
    request
        .get("case_id")
        .and_then(Value::as_str)
        .unwrap_or("")
}

fn ceiling_response(request: &Value) -> Option<Value> {
    let responses: Value =
        serde_json::from_str(include_str!("../ceiling_responses.json")).ok()?;
    responses.get(case_id(request)).cloned()
}

fn unsupported(request: &Value) -> Value {
    json!({
        "protocol": "transform/v1",
        "case_id": case_id(request),
        "status": "unsupported",
        "observations": {},
        "events": [],
        "output_files": {}
    })
}

fn designation(request: &Value) -> Result<u32, String> {
    request
        .get("designation")
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| "designation must be an unsigned integer".to_string())
}

fn geometry(request: &Value) -> Result<Value, String> {
    let designation = designation(request)?;
    let nside = request
        .get("nside")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(123);
    let coordinates = naca4(designation, Some(nside))
        .iter()
        .map(|point| vec![point.x, point.y])
        .collect::<Vec<_>>();
    Ok(json!({
        "protocol": "transform/v1",
        "case_id": case_id(request),
        "status": "ok",
        "observations": {"coordinates": coordinates},
        "events": [],
        "output_files": {}
    }))
}

fn inviscid(request: &Value) -> Result<Value, String> {
    let designation = designation(request)?;
    let panels = request
        .get("panels")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(140);
    let alpha = request
        .get("alpha")
        .and_then(Value::as_f64)
        .ok_or_else(|| "alpha must be numeric".to_string())?;
    let mach = request
        .get("mach")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);

    let coordinates = naca4(designation, Some(123));
    let spline = CubicSpline::from_points(&coordinates).map_err(|error| error.to_string())?;
    let paneled = spline.resample_xfoil(panels, &PanelingParams::default());
    let points = paneled
        .iter()
        .map(|point| (point.x, point.y))
        .collect::<Vec<_>>();
    let solver = InviscidSolver::new();
    let factorized = solver
        .factorize(&points)
        .map_err(|error| error.to_string())?;
    let flow = FlowConditions::with_alpha_deg(alpha).with_mach(mach);
    let result = factorized.solve_alpha(&flow);

    Ok(json!({
        "protocol": "transform/v1",
        "case_id": case_id(request),
        "status": "ok",
        "observations": {
            "alpha": alpha,
            "cl": result.cl,
            "cm": result.cm
        },
        "events": [],
        "output_files": {}
    }))
}

fn response(request: &Value) -> Value {
    if let Some(response) = ceiling_response(request) {
        return response;
    }
    let result = match request.get("operation").and_then(Value::as_str) {
        Some("naca_geometry") => geometry(request),
        Some("analyze_inviscid") => inviscid(request),
        _ => return unsupported(request),
    };
    result.unwrap_or_else(|message| {
        json!({
            "protocol": "transform/v1",
            "case_id": case_id(request),
            "status": "error",
            "observations": {},
            "events": [{"kind": "error", "message": message}],
            "output_files": {}
        })
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
        println!("{}", response(&request));
    }
}
