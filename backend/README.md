# Disaster Response Simulation System

**Author:** Vishal Krishna Shah  
**Team:** Devika Shaj Kumar Nair, Vishal Krishna Shah  
**Platform:** MacBook Air M2

## Quick Start

### 1. Install Dependencies
```bash
pip3 install --break-system-packages -r requirements.txt
```

### 2. Test Installation
```bash
python3 test_installation.py
```

### 3. Run Server
```bash
python3 -m uvicorn app.main:app --reload --port 8000
```

### 4. Test API
Open browser: http://localhost:8000

## API Endpoints

- `GET /` - Welcome message
- `GET /health` - Health check
- `GET /api/info` - Project information
- `WS /ws` - WebSocket endpoint

## Project Status

✅ Backend core (40% complete)
⏳ Behavioral modes (TODO)
⏳ Frontend visualization (TODO)
⏳ ML integration (TODO)

## Next Steps

1. Implement three behavioral modes
2. Build React frontend
3. Add machine learning
4. Complete testing and documentation

## Support

Created by: all_in_one_installer.py
Date: November 17, 2025
