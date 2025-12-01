"""
FastAPI application - Main entry point
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional, Any
import asyncio
import json

from app.core.config import settings, BehaviorMode, DisasterType
from app.simulation_engine import SimulationEngine, SimulationConfig, create_simulation
from app.core.utils import Position

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global simulation instance
simulation: Optional[SimulationEngine] = None
websocket_clients: List[WebSocket] = []


# === Simulation Update Loop ===

async def simulation_update_loop():
    """Background task to update simulation"""
    target_fps = 30
    frame_time = 1.0 / target_fps
    
    update_count = 0
    
    while True:
        if simulation and simulation.running:
            simulation.update(frame_time * simulation.config.simulation_speed)
            update_count += 1
            
            # DEBUG: Print every 100 updates
            if update_count % 100 == 0:
                print(f"🔄 Update loop tick {update_count}: Time={simulation.current_time:.1f}s, Agents moved={sum(a.stats.distance_traveled for a in simulation.agents.values()):.1f}")
            
            # Broadcast state every 5 ticks
            if simulation.tick_count % 5 == 0:
                await broadcast_state()
        
        await asyncio.sleep(frame_time)

# async def simulation_update_loop():
#     """Background task to update simulation"""
#     target_fps = 30
#     frame_time = 1.0 / target_fps
    
#     while True:
#         if simulation and simulation.running:
#             simulation.update(frame_time * simulation.config.simulation_speed)
            
#             # Broadcast state every 5 ticks
#             if simulation.tick_count % 5 == 0:
#                 await broadcast_state()
        
#         await asyncio.sleep(frame_time)


@app.on_event("startup")
async def startup_event():
    """Initialize simulation on startup and start background tasks"""
    global simulation
    
    # Initialize simulation
    simulation = create_simulation(
        num_agents=10,
        behavior_mode=BehaviorMode.COOPERATIVE,
        width=50,
        height=50
    )
    
    # Start background update loop
    asyncio.create_task(simulation_update_loop())
    
    print(f"✅ {settings.APP_NAME} started successfully")
    print(f"🚀 Simulation update loop started (30 FPS)")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global simulation
    if simulation:
        simulation.stop()
    print("👋 Simulation shutdown complete")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "author": "Vishal Krishna Shah"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "simulation_running": simulation.running if simulation else False
    }


@app.get("/api/info")
async def get_info():
    """Get project information"""
    return {
        "project": "Multi-Agent Disaster Response Simulation",
        "author": "Vishal Krishna Shah",
        "team": ["Devika Shaj Kumar Nair", "Vishal Krishna Shah"],
        "features": [
            "Real-time simulation",
            "Multi-agent coordination",
            "A* pathfinding",
            "Three behavioral modes",
            "WebSocket updates"
        ],
        "status": "Backend operational"
    }


# === Simulation Control Endpoints ===

@app.post("/simulation/start")
async def start_simulation():
    """Start the simulation"""
    if not simulation:
        raise HTTPException(status_code=500, detail="Simulation not initialized")
    
    simulation.start()
    await broadcast_state()
    return {"status": "started"}


@app.post("/simulation/stop")
async def stop_simulation():
    """Stop the simulation"""
    if not simulation:
        raise HTTPException(status_code=500, detail="Simulation not initialized")
    
    simulation.stop()
    await broadcast_state()
    return {"status": "stopped"}


@app.post("/simulation/pause")
async def pause_simulation():
    """Pause the simulation"""
    if not simulation:
        raise HTTPException(status_code=500, detail="Simulation not initialized")
    
    simulation.pause()
    await broadcast_state()
    return {"status": "paused"}


@app.post("/simulation/resume")
async def resume_simulation():
    """Resume the simulation"""
    if not simulation:
        raise HTTPException(status_code=500, detail="Simulation not initialized")
    
    simulation.resume()
    await broadcast_state()
    return {"status": "resumed"}


@app.post("/simulation/reset")
async def reset_simulation():
    """Reset the simulation"""
    if not simulation:
        raise HTTPException(status_code=500, detail="Simulation not initialized")
    
    simulation.reset()
    await broadcast_state()
    return {"status": "reset"}


@app.post("/simulation/behavior/{mode}")
async def set_behavior_mode(mode: str):
    """Change behavior mode"""
    if not simulation:
        raise HTTPException(status_code=500, detail="Simulation not initialized")
    
    valid_modes = [BehaviorMode.COMPETITIVE, BehaviorMode.COOPERATIVE, BehaviorMode.AGREEMENT]
    if mode not in valid_modes:
        raise HTTPException(status_code=400, detail=f"Invalid mode. Must be one of: {valid_modes}")
    
    simulation.set_behavior_mode(mode)
    await broadcast_state()
    return {"status": "behavior_changed", "mode": mode}


# === State Query Endpoints ===

@app.get("/simulation/state")
async def get_simulation_state():
    """Get current simulation state"""
    if not simulation:
        raise HTTPException(status_code=500, detail="Simulation not initialized")
    
    return simulation.get_state()


@app.get("/simulation/metrics")
async def get_metrics():
    """Get simulation metrics"""
    if not simulation:
        raise HTTPException(status_code=500, detail="Simulation not initialized")
    
    return simulation.get_metrics()


@app.get("/simulation/agents")
async def get_agents():
    """Get all agents"""
    if not simulation:
        raise HTTPException(status_code=500, detail="Simulation not initialized")
    
    return [agent.get_state_dict() for agent in simulation.agents.values()]


@app.get("/simulation/agents/{agent_id}")
async def get_agent(agent_id: str):
    """Get specific agent details"""
    if not simulation:
        raise HTTPException(status_code=500, detail="Simulation not initialized")
    
    if agent_id not in simulation.agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return simulation.agents[agent_id].get_state_dict()


@app.get("/simulation/environment")
async def get_environment():
    """Get environment state"""
    if not simulation:
        raise HTTPException(status_code=500, detail="Simulation not initialized")
    
    return simulation.environment.get_state_dict()


# === Action Endpoints ===

@app.post("/simulation/disaster/spawn")
async def spawn_disaster(disaster_type: str, x: Optional[int] = None, y: Optional[int] = None):
    """Spawn a new disaster"""
    if not simulation:
        raise HTTPException(status_code=500, detail="Simulation not initialized")
    
    valid_types = [DisasterType.FIRE, DisasterType.FLOOD, DisasterType.EARTHQUAKE]
    if disaster_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"Invalid type. Must be one of: {valid_types}")
    
    position = Position(x, y) if x is not None and y is not None else None
    disaster = simulation.spawn_disaster(disaster_type, position)
    
    await broadcast_state()
    
    return {
        "status": "disaster_spawned",
        "disaster": {
            "id": disaster.id,
            "type": disaster.type,
            "position": {"x": disaster.position.x, "y": disaster.position.y}
        }
    }


# === WebSocket Endpoint ===

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    websocket_clients.append(websocket)
    
    try:
        # Send initial state
        if simulation:
            await websocket.send_json(simulation.get_state())
        
        while True:
            data = await websocket.receive_text()
            
            try:
                command = json.loads(data)
                await handle_websocket_command(command, websocket)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
            except Exception as e:
                await websocket.send_json({"error": str(e)})
    
    except WebSocketDisconnect:
        websocket_clients.remove(websocket)
        print("Client disconnected")


async def handle_websocket_command(command: dict, websocket: WebSocket):
    """Handle commands received via WebSocket"""
    cmd_type = command.get("type")
    
    if cmd_type == "start":
        simulation.start()
    elif cmd_type == "stop":
        simulation.stop()
    elif cmd_type == "pause":
        simulation.pause()
    elif cmd_type == "resume":
        simulation.resume()
    elif cmd_type == "reset":
        simulation.reset()
    elif cmd_type == "get_state":
        await websocket.send_json(simulation.get_state())
    else:
        await websocket.send_json({"error": f"Unknown command: {cmd_type}"})


async def broadcast_state():
    """Broadcast current state to all connected WebSocket clients"""
    if not simulation or not websocket_clients:
        return
    
    state = simulation.get_state()
    
    disconnected = []
    for client in websocket_clients:
        try:
            await client.send_json(state)
        except:
            disconnected.append(client)
    
    # Remove disconnected clients
    for client in disconnected:
        websocket_clients.remove(client)

# ============ INTEGRATED SIMULATION ENDPOINTS ============

integrated_sim: Optional[Any] = None

@app.post("/simulation/integrated/start")
async def start_integrated_simulation(
    robots: int = 20,
    victims: int = 2,
    false_positives: int = 10
):
    """Start integrated robot search + agent response simulation"""
    global integrated_sim
    
    from app.integrated_pipeline import IntegratedSimulation
    
    integrated_sim = IntegratedSimulation(
        n_robots=robots,
        n_victims=victims,
        n_false=false_positives
    )
    
    integrated_sim.initialize()
    
    return {
        "status": "initialized",
        "robots": robots,
        "victims": victims,
        "false_positives": false_positives
    }


@app.post("/simulation/integrated/phase1")
async def run_integrated_phase1():
    """Run Phase 1: Robot search"""
    if not integrated_sim:
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    result = integrated_sim.run_combined_phases()
    return result


@app.post("/simulation/integrated/phase2")
async def run_integrated_phase2(behavior_mode: str = "cooperative"):
    """Run Phase 2: Agent response"""
    if not integrated_sim:
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    result = integrated_sim.run_phase2_response(behavior_mode)
    return result


@app.get("/simulation/integrated/state")
async def get_integrated_state():
    """Get current integrated simulation state"""
    if not integrated_sim:
        return {"phase": "idle"}
    
    return integrated_sim.get_state()


@app.get("/simulation/integrated/results")
async def get_integrated_results():
    """Get combined results from both phases"""
    if not integrated_sim:
        raise HTTPException(status_code=400, detail="No simulation run")
    
    results = integrated_sim.get_combined_results()
    return {
        "phase1_complete": results.phase1_complete,
        "phase2_complete": results.phase2_complete,
        "search_time": results.search_time,
        "response_time": results.response_time,
        "victims_discovered": results.victims_discovered,
        "victims_rescued": results.victims_rescued,
        "total_time": results.total_time
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
