# 2D Planar Diffuser Separation Control Challenge

## Background & Objective
You are designing a two-dimensional asymmetric expansion diffuser used in a chemical process fluid line. The goal is to maximize static pressure recovery across the diffuser while preventing flow separation and limiting head loss.

You must choose the geometric expansion profile and length parameters to optimize fluid energy efficiency under specified operating bounds.

## Deliverable
Output your final design to `/tmp/output/diffuser_design.json`.

## Required Output Format
```json
{
  "half_angle_deg": 7.5,
  "length_ratio": 4.0,
  "inlet_extension_m": 0.5
}