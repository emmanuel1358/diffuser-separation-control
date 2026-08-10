//! Regenerate `testdata/naca0012_*` fixtures used by `spline` unit tests.
//!
//! ```text
//! cargo run -p rustfoil-core --example gen_spline_fixtures
//! ```

use rustfoil_core::naca::naca4;
use rustfoil_core::spline::{CubicSpline, PanelingParams};
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

fn main() {
    let testdata = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata");
    std::fs::create_dir_all(&testdata).expect("create testdata");

    let buffer = naca4(12, Some(123));
    let mut f = File::create(testdata.join("naca0012_buffer_real.dat")).expect("buffer file");
    writeln!(
        f,
        "NACA 0012 buffer (rustfoil naca4, nside=123) — regenerate: cargo run -p rustfoil-core --example gen_spline_fixtures"
    )
    .unwrap();
    for p in &buffer {
        writeln!(f, "  {:.8}  {:.8}", p.x, p.y).unwrap();
    }

    let spline = CubicSpline::from_points(&buffer).expect("spline");
    let paneled = spline.resample_xfoil(160, &PanelingParams::default());
    let mut f2 = File::create(testdata.join("naca0012_xfoil_paneled.dat")).expect("paneled file");
    writeln!(
        f2,
        "NACA 0012 paneled (rustfoil resample_xfoil NP=160) — regenerate: cargo run -p rustfoil-core --example gen_spline_fixtures"
    )
    .unwrap();
    for p in &paneled {
        writeln!(f2, "  {:.8}  {:.8}", p.x, p.y).unwrap();
    }

    eprintln!("Wrote:\n  {:?}\n  {:?}", testdata.join("naca0012_buffer_real.dat"), testdata.join("naca0012_xfoil_paneled.dat"));
}
