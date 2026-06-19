"""Single-ROP FastAPI microservice.

Self-contained computer-vision pipeline for the automated detection and
classification of Retinopathy of Prematurity (ROP) from retinal fundus images.

The package is intentionally split so the *business logic* (``service.py``) has
no dependency on the web framework, making it unit-testable in isolation and
re-usable from other contexts (CLI, batch jobs, notebooks).
"""
