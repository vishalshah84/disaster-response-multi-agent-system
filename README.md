# 🚨 Multi-Agent Disaster Response System

A three-phase multi-agent coordination system for disaster response, combining consensus-based victim search with priority-based rescue coordination.

## 🎯 Project Overview

This project implements a complete disaster response pipeline using multi-agent systems:

- **Phase 1:** Distributed consensus-based victim search (Markov random walk + gossip algorithm)
- **Phase 2:** Formation control and navigation (potential fields + K-NN assignment)
- **Phase 3:** Priority-based rescue with dynamic health management

## 👥 Team Members

- **Vishal Krishna Shah** - Phase 3 (Priority-Based Rescue) + System Integration + Visualization
- **Devika** - Phase 1 (Consensus Search) + Phase 2 (Formation Control)

**Course:** MAE 598 - Multi-Robot Systems  
**Institution:** Arizona State University  
**Semester:** Fall 2024

## 🏗️ System Architecture
```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (React)                      │
│  - Real-time visualization                              │
│  - Interactive control panel                            │
│  - Live metrics dashboard                               │
└──────────────────┬──────────────────────────────────────┘
                   │ WebSocket / REST API
┌──────────────────▼──────────────────────────────────────┐
│                  Backend (FastAPI)                       │
│  ┌────────────┐  ┌────────────┐  ┌────────────────┐   │
│  │  Phase 1   │→ │  Phase 2   │→ │    Phase 3     │   │
│  │  Search    │  │ Navigation │  │ Priority Rescue│   │
│  └────────────┘  └────────────┘  └────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## 🔬 Algorithms Implemented

### Phase 1: Consensus-Based Search
- **Markov Random Walk** for exploration
- **Distributed Gossip Consensus** for victim localization
- **Sensor Fusion** with Gaussian noise
- **False Positive Filtering** via signal strength

### Phase 2: Formation Control
- **Potential Field Navigation**
- **K-Nearest Neighbor Assignment**
- **Circular Formation Control** (6 robots per victim)
- **Swarm Coordination**

### Phase 3: Priority-Based Rescue (Novel Contribution)
- **Dynamic Health Model** with degradation
- **Priority Classification** (Critical/High/Medium/Low)
- **Weighted Task Allocation**
- **Real-time Reallocation** based on victim status

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 16+
- pip and npm

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

### Frontend Setup
```bash
cd frontend
npm install
npm start
```

### Access
- **Frontend:** http://localhost:3000
- **API Docs:** http://localhost:8000/docs

## 📊 Features

- ✅ Real-time multi-agent simulation
- ✅ Three behavioral modes: Competitive, Cooperative, Agreement
- ✅ Integrated search & priority-based rescue pipeline
- ✅ Live health monitoring and rescue progress
- ✅ Professional dashboard with metrics visualization
- ✅ WebSocket-based real-time updates

## 📈 Results

- **Consensus Convergence:** < 300 steps
- **Victim Discovery Rate:** 100% with 2 victims, 20 robots
- **Rescue Success Rate:** 100% with priority-based allocation
- **Average Rescue Time:** 45 seconds per scenario

## 📚 Key References

1. Olfati-Saber & Murray (2004) - Consensus in networks
2. Khatib (1986) - Potential field navigation
3. Gerkey & Matarić (2004) - Task allocation taxonomy
4. Balch & Arkin (1998) - Formation control

## 🎨 Tech Stack

**Backend:**
- FastAPI (Python)
- NumPy for simulations
- WebSockets for real-time communication

**Frontend:**
- React.js
- HTML5 Canvas for visualization
- Modern CSS with glassmorphism

## 📝 License

This project is developed for academic purposes as part of MAE 598 coursework.

## 🙏 Acknowledgments

- Prof. Spring Berman - Course Instructor
- Arizona State University - School of Engineering
- Teammates for collaboration and integration

**⭐ Star this repo if you find it useful!**
