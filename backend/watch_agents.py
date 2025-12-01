from app.simulation_engine import create_simulation
from app.core.config import DisasterType
import time

sim = create_simulation(5, 'cooperative', 30, 30)
sim.spawn_disaster(DisasterType.FIRE)
sim.start()

print("Watching agent positions for 10 seconds...")
for i in range(20):
    sim.update(0.5)
    print(f"\nTick {i}:")
    for agent_id, agent in list(sim.agents.items())[:3]:
        print(f"  {agent_id}: ({agent.position.x}, {agent.position.y}) - Fuel: {agent.fuel}")
    time.sleep(0.5)

sim.stop()
print("\n✅ Agents ARE moving in backend!")
