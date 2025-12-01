"""
Main simulation engine
"""
import asyncio
import time
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np

from app.core.config import AgentType, BehaviorMode, DisasterType, settings
from app.core.utils import Position, generate_random_position
from app.environment.grid_world import Environment
from app.agents.base_agent import Agent, Message
from app.agents.behavioral_agent import BehavioralAgent
from app.behaviors.competitive import AuctionManager
from app.behaviors.cooperative import CoalitionManager
from app.behaviors.agreement import ConsensusManager


@dataclass
class SimulationConfig:
    """Configuration for simulation"""
    width: int = settings.GRID_WIDTH
    height: int = settings.GRID_HEIGHT
    num_agents: int = 10
    simulation_speed: float = settings.SIMULATION_SPEED
    disaster_spawn_interval: int = 5
    max_disasters: int = 10
    behavior_mode: str = BehaviorMode.COOPERATIVE


class SimulationEngine:
    """Main simulation engine that coordinates everything"""
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        
        # Core components
        self.environment = Environment(self.config.width, self.config.height)
        self.agents: Dict[str, Agent] = {}
        
        # Behavior managers
        self.auction_manager = AuctionManager()
        self.coalition_manager = CoalitionManager()
        self.consensus_manager = ConsensusManager()
        
        # Simulation state
        self.running = False
        self.paused = False
        self.current_time = 0.0
        self.last_disaster_spawn = 0.0
        self.tick_count = 0
        
        # Message queue
        self.message_queue: List[Message] = []
        
        # Event subscribers
        self.event_callbacks = []
        
        # Performance metrics
        self.fps = 0
        self.update_times: List[float] = []
        
        # Initialize simulation
        self._initialize()
    
    def _initialize(self):
        """Initialize simulation with agents and initial disasters"""
        # Spawn agents
        agent_types = [
            AgentType.FIRE_FIGHTER,
            AgentType.MEDICAL,
            AgentType.SUPPLY,
            AgentType.SEARCH_RESCUE
        ]
        
        for i in range(self.config.num_agents):
            agent_type = agent_types[i % len(agent_types)]
            pos_tuple = generate_random_position(
                self.config.width,
                self.config.height,
                exclude=list(self.environment.obstacles)
            )
            position = Position(pos_tuple[0], pos_tuple[1])
            
            agent = BehavioralAgent(
                agent_id=f"agent_{i:03d}",
                agent_type=agent_type,
                position=position,
                behavior_mode=self.config.behavior_mode
            )
            
            agent.initialize(self.environment)
            agent.set_managers(
                self.auction_manager,
                self.coalition_manager,
                self.consensus_manager
            )
            self.agents[agent.id] = agent
        
        # Spawn initial disasters
        self._spawn_initial_disasters()
    
    def _spawn_initial_disasters(self):
        """Spawn initial disasters"""
        disaster_types = [
            DisasterType.FIRE,
            DisasterType.FLOOD,
            DisasterType.EARTHQUAKE
        ]
        
        for i in range(3):
            disaster_type = disaster_types[i % len(disaster_types)]
            self.environment.spawn_disaster(disaster_type)
    
    def add_agent(self, agent: Agent):
        """Add an agent to the simulation"""
        agent.initialize(self.environment)
        self.agents[agent.id] = agent
    
    def remove_agent(self, agent_id: str):
        """Remove an agent from the simulation"""
        if agent_id in self.agents:
            del self.agents[agent_id]
    
    def spawn_disaster(self, disaster_type: str, 
                      position: Optional[Position] = None):
        """Spawn a new disaster"""
        return self.environment.spawn_disaster(disaster_type, position)
    
    def update(self, delta_time: float):
        """Main update loop"""
        if not self.running or self.paused:
            return
        
        start_time = time.time()
        
        # Update environment
        self.environment.update(delta_time)
        
        # Spawn disasters periodically
        if (self.current_time - self.last_disaster_spawn) >= self.config.disaster_spawn_interval:
            active_disasters = len([d for d in self.environment.disasters.values() if d.active])
            if active_disasters < self.config.max_disasters:
                disaster_types = [DisasterType.FIRE, DisasterType.FLOOD, DisasterType.EARTHQUAKE]
                disaster_type = np.random.choice(disaster_types)
                self.spawn_disaster(disaster_type)
            self.last_disaster_spawn = self.current_time
        
        # Process message queue
        self._process_messages()
        
        # Update behavior managers
        if self.config.behavior_mode == BehaviorMode.COMPETITIVE:
            # Resolve auctions periodically
            if self.tick_count % 10 == 0:  # Every 10 ticks
                for target_id in list(self.auction_manager.active_auctions.keys()):
                    winner = self.auction_manager.resolve_auction(target_id)
                    if winner and winner in self.agents:
                        self.agents[winner].competitive_behavior.my_assignments.append(target_id)
        
        elif self.config.behavior_mode == BehaviorMode.AGREEMENT:
            # Update consensus
            self.consensus_manager.update(delta_time)
        
        # Update all agents
        for agent in self.agents.values():
            agent.update(delta_time, self.environment, self.agents)
            
            # Collect outgoing messages
            while agent.outbox:
                message = agent.outbox.pop(0)
                self.message_queue.append(message)
        
        # Update time and tick count
        self.current_time += delta_time
        self.tick_count += 1
        
        # Track performance
        update_time = time.time() - start_time
        self.update_times.append(update_time)
        if len(self.update_times) > 100:
            self.update_times.pop(0)
        
        avg_update_time = sum(self.update_times) / len(self.update_times)
        self.fps = 1.0 / avg_update_time if avg_update_time > 0 else 0
        
        # Emit events
        self._emit_event("update", {
            "time": self.current_time,
            "tick": self.tick_count
        })
    
    def _process_messages(self):
        """Process inter-agent messages"""
        while self.message_queue:
            message = self.message_queue.pop(0)
            
            if message.receiver_id is None:
                # Broadcast to all agents
                for agent in self.agents.values():
                    if agent.id != message.sender_id:
                        agent.receive_message(message)
            else:
                # Send to specific agent
                if message.receiver_id in self.agents:
                    self.agents[message.receiver_id].receive_message(message)
    
    def start(self):
        """Start the simulation"""
        self.running = True
        self.paused = False
        self._emit_event("started", {})
    
    def stop(self):
        """Stop the simulation"""
        self.running = False
        self._emit_event("stopped", {})
    
    def pause(self):
        """Pause the simulation"""
        self.paused = True
        self._emit_event("paused", {})
    
    def resume(self):
        """Resume the simulation"""
        self.paused = False
        self._emit_event("resumed", {})
    
    def reset(self):
        """Reset the simulation to initial state"""
        self.running = False
        self.paused = False
        self.current_time = 0.0
        self.last_disaster_spawn = 0.0
        self.tick_count = 0
        
        self.environment.reset()
        self.agents.clear()
        self.message_queue.clear()
        
        self._initialize()
        self._emit_event("reset", {})
    
    def set_behavior_mode(self, mode: str):
        """Change behavior mode for all agents"""
        self.config.behavior_mode = mode
        for agent in self.agents.values():
            agent.behavior_mode = mode
        self._emit_event("behavior_changed", {"mode": mode})
    
    def subscribe_to_events(self, callback):
        """Subscribe to simulation events"""
        self.event_callbacks.append(callback)
    
    def _emit_event(self, event_type: str, data: Dict):
        """Emit an event to all subscribers"""
        for callback in self.event_callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                print(f"Error in event callback: {e}")
    
    def get_state(self) -> Dict:
        """Get complete simulation state"""
        return {
            "time": self.current_time,
            "tick": self.tick_count,
            "running": self.running,
            "paused": self.paused,
            "fps": self.fps,
            "config": {
                "width": self.config.width,
                "height": self.config.height,
                "num_agents": len(self.agents),
                "behavior_mode": self.config.behavior_mode,
                "simulation_speed": self.config.simulation_speed
            },
            "environment": self.environment.get_state_dict(),
            "agents": [agent.get_state_dict() for agent in self.agents.values()],
            "metrics": self.get_metrics()
        }
    
    def get_metrics(self) -> Dict:
        """Calculate and return simulation metrics"""
        total_disasters_resolved = sum(a.stats.disasters_resolved for a in self.agents.values())
        total_victims_rescued = self.environment.victims_saved  # Use environment count instead
        total_distance_traveled = sum(a.stats.distance_traveled for a in self.agents.values())
        total_messages = sum(a.stats.messages_sent for a in self.agents.values())
        total_rewards = sum(a.stats.rewards_earned for a in self.agents.values())
        
        avg_agent_fuel = np.mean([a.fuel for a in self.agents.values()]) if self.agents else 0
        avg_agent_supplies = np.mean([a.supplies for a in self.agents.values()]) if self.agents else 0
        
        active_disasters = len([d for d in self.environment.disasters.values() if d.active])
        
        avg_response_time = self.current_time / max(total_disasters_resolved, 1)
        
        efficiency = total_victims_rescued / max(self.environment.victims_saved + self.environment.victims_lost, 1)
        
        return {
            "disasters_resolved": total_disasters_resolved,
            "victims_rescued": total_victims_rescued,
            "victims_lost": self.environment.victims_lost,
            "active_disasters": active_disasters,
            "total_distance_traveled": total_distance_traveled,
            "total_messages": total_messages,
            "total_rewards": total_rewards,
            "avg_agent_fuel": avg_agent_fuel,
            "avg_agent_supplies": avg_agent_supplies,
            "avg_response_time": avg_response_time,
            "rescue_efficiency": efficiency * 100,
            "resource_utilization": (1 - avg_agent_fuel / settings.AGENT_MAX_FUEL) * 100
        }
    
    def run_for_ticks(self, num_ticks: int, delta_time: float = 0.1):
        """Run simulation for a specific number of ticks"""
        self.start()
        
        for _ in range(num_ticks):
            self.update(delta_time)
        
        self.stop()


def create_simulation(num_agents: int = 10, 
                     behavior_mode: str = BehaviorMode.COOPERATIVE,
                     width: int = 50,
                     height: int = 50) -> SimulationEngine:
    """Create a new simulation with given parameters"""
    config = SimulationConfig(
        width=width,
        height=height,
        num_agents=num_agents,
        behavior_mode=behavior_mode
    )
    
    return SimulationEngine(config)
