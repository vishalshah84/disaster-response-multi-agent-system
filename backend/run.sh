#!/bin/bash
# Quick run script
cd "$(dirname "$0")"
python3 -m uvicorn app.main:app --reload --port 8000
