# ADR-010 Path & Allocation Policy

Status: Proposed
Date: 2025-11-04

## Decision
Adopt a deterministic path allocation policy driven by {domain, capability, layer} and enforced via a Path Planner and CI path linter.

## Why
- Keeps code discoverable and scalable across Dream → Build → Scale.
- Enables generators and automated reviews.

## Alternatives Rejected
- Ad-hoc placement; single flat folders.

## Risks
- Initial friction; requires developer education.

## Rollback Plan
- Keep the planner behind a script; disable linter if needed while refactoring.
