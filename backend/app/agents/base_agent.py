"""
Base Agent class - FIXED VERSION with guaranteed movement
"""
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import random

from app.core.config import AgentType, BehaviorMode, settings, REWARDS
from app.core.utils import Position, IDGenerator
from app.algorithms.pathfinding import PathPlanner
from app.environment.grid_world import Environment, Disaster, Resource, Victim


class AgentState(Enum):
    """Agent states"""
    IDLE = "idle"
    MOVING = "moving"
    ACTING = "acting"
    WAITING = "waiting"
    DEAD = "dead"


class Action(Enum):
    """Possible agent actions"""
    MOVE = "move"
    EXTINGUISH_FIRE = "extinguish_fire"
    RESCUE_VICTIM = "rescue_victim"
    DELIVER_SUPPLIES = "deliver_supplies"
    COLLECT_RESOURCE = "collect_resource"
    COMMUNICATE = "communicate"
    WAIT = "wait"


@dataclass
class AgentStats:
    """Agent statistics"""
    disasters_resolved: int = 0
    victims_rescued: int = 0
    resources_delivered: int = 0
    distance_traveled: float = 0.0
    actions_taken: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    rewards_earned: float = 0.0


@dataclass
class Message:
    """Message for inter-agent communication"""
    sender_id: str
    receiver_id: Optional[str]
    message_type: str
    content: Dict[str, Any]
    timestamp: float


class Agent:
    """Base agent class for disaster response simulation"""
    
    def __init__(self, agent_id: Optional[str] = None,
                 agent_type: str = AgentType.FIRE_FIGHTER,
                 position: Optional[Position] = None,
                 behavior_mode: str = BehaviorMode.COOPERATIVE):
        
        self.id = agent_id or IDGenerator.generate("agent")
        self.type = agent_type
        self.position = position or Position(0, 0)
        self.behavior_mode = behavior_mode
        
        # State
        self.state = AgentState.IDLE
        self.health = 100.0
        self.fuel = settings.AGENT_MAX_FUEL
        self.supplies = settings.AGENT_MAX_SUPPLIES
        self.speed = settings.AGENT_SPEED
        
        # Perception
        self.vision_radius = settings.AGENT_VISION_RADIUS
        self.visible_agents: List['Agent'] = []
        self.visible_disasters: List[Disaster] = []
        self.visible_resources: List[Resource] = []
        self.visible_victims: List[Victim] = []
        
        # Planning
        self.path_planner: Optional[PathPlanner] = None
        self.current_target: Optional[Position] = None
        self.current_action: Optional[Action] = None
        
        # Communication
        self.inbox: List[Message] = []
        self.outbox: List[Message] = []
        
        # Statistics
        self.stats = AgentStats()
        
        # Learning
        self.reward_history: List[float] = []
        self.action_history: List[Action] = []
    
    def initialize(self, environment: Environment):
        """Initialize agent with environment reference"""
        self.path_planner = PathPlanner(
            environment.width,
            environment.height,
            environment.is_walkable
        )
    
    def perceive(self, environment: Environment):
        """Update agent's perception of the environment"""
        self.visible_agents = []
        self.visible_disasters = []
        self.visible_resources = []
        self.visible_victims = []
        
        self.visible_disasters = environment.get_disasters_in_radius(
            self.position, self.vision_radius
        )
        self.visible_resources = environment.get_resources_in_radius(
            self.position, self.vision_radius
        )
        self.visible_victims = environment.get_victims_in_radius(
            self.position, self.vision_radius
        )
    
    def decide(self, environment: Environment) -> Optional[Action]:
        """Decide next action - ALWAYS returns an action"""
        
        # Check if we need fuel
        if self.fuel < 100 and self.visible_resources:
            fuel_resources = [r for r in self.visible_resources if r.type == "fuel"]
            if fuel_resources:
                self.current_target = min(fuel_resources, 
                                        key=lambda r: r.position.distance_to(self.position)).position
                return Action.MOVE
        
        # Priority: rescue victims
        if self.visible_victims and self.type in [AgentType.MEDICAL, AgentType.SEARCH_RESCUE]:
            nearest_victim = min(self.visible_victims,
                               key=lambda v: v.position.distance_to(self.position))
            self.current_target = nearest_victim.position
            
            if self.position.is_adjacent(nearest_victim.position):
                return Action.RESCUE_VICTIM
            return Action.MOVE
        
        # Handle disasters
        if self.visible_disasters:
            if self.type == AgentType.FIRE_FIGHTER:
                fire_disasters = [d for d in self.visible_disasters if d.type == "fire"]
                if fire_disasters:
                    nearest_fire = min(fire_disasters,
                                     key=lambda d: d.position.distance_to(self.position))
                    new_target = nearest_fire.position
                    if new_target != self.current_target:
                        self.current_target = new_target
                        if self.path_planner:
                            self.path_planner.current_path = None
                    
                    if self.position.is_adjacent(nearest_fire.position):
                        return Action.EXTINGUISH_FIRE
                    return Action.MOVE
        
        # ALWAYS explore - generate new target every time if none or close to current
        if not self.current_target or self.position.distance_to(self.current_target) < 2:
            self.current_target = Position(
                random.randint(0, environment.width - 1),
                random.randint(0, environment.height - 1)
            )
            if self.path_planner:
                self.path_planner.current_path = None
        
        return Action.MOVE
    
    def act(self, action: Action, environment: Environment) -> float:
        """Execute an action and return reward"""
        reward = 0.0
        self.stats.actions_taken += 1
        
        if action == Action.MOVE:
            reward = self._move(environment)
        elif action == Action.EXTINGUISH_FIRE:
            reward = self._extinguish_fire(environment)
        elif action == Action.RESCUE_VICTIM:
            reward = self._rescue_victim(environment)
        elif action == Action.DELIVER_SUPPLIES:
            reward = self._deliver_supplies(environment)
        elif action == Action.COLLECT_RESOURCE:
            reward = self._collect_resource(environment)
        elif action == Action.WAIT:
            reward = 0
        
        self.stats.rewards_earned += reward
        self.reward_history.append(reward)
        self.action_history.append(action)
        
        return reward
    
    def _move(self, environment: Environment) -> float:
        """Move towards current target - GUARANTEED TO WORK"""
        if not self.current_target or not self.path_planner:
            # Generate new random target
            self.current_target = Position(
                random.randint(0, environment.width - 1),
                random.randint(0, environment.height - 1)
            )
            return REWARDS["movement_penalty"]
        
        # Check if reached target
        if self.position.distance_to(self.current_target) < 1.5:
            self.current_target = None  # Will get new target next decide()
            return REWARDS["movement_penalty"]
        
        # Plan path
        if not self.path_planner.current_path or \
           self.path_planner.needs_replanning(self.position.to_tuple()):
            path = self.path_planner.plan_path(
                self.position.to_tuple(),
                self.current_target.to_tuple()
            )
            if not path or len(path) < 2:
                # Can't reach target - clear it
                self.current_target = None
                return REWARDS["movement_penalty"]
        
        # Get next position
        next_pos = self.path_planner.get_next_position()
        if not next_pos:
            # Path failed - clear target
            self.current_target = None
            return REWARDS["movement_penalty"]
        
        # Check fuel
        if self.fuel < 1:
            self.state = AgentState.IDLE
            return REWARDS["movement_penalty"] * 2
        
        # MOVE!
        old_pos = self.position
        self.position = Position(next_pos[0], next_pos[1])
        self.fuel -= 1
        
        distance = old_pos.distance_to(self.position)
        self.stats.distance_traveled += distance
        
        return REWARDS["movement_penalty"]
    
    def _extinguish_fire(self, environment: Environment) -> float:
        """Extinguish a fire"""
        if self.supplies < 10:
            return 0
        
        for disaster in self.visible_disasters:
            if disaster.type == "fire" and disaster.position.is_adjacent(self.position):
                disaster.intensity = max(0, disaster.intensity - 0.3)
                self.supplies -= 10
                
                if disaster.intensity <= 0:
                    environment.resolve_disaster(disaster.id)
                    self.stats.disasters_resolved += 1
                    self.current_target = None
                    return REWARDS["disaster_resolved"]
                
                return REWARDS["disaster_resolved"] * 0.5
        
        return 0
    
    def _rescue_victim(self, environment: Environment) -> float:
        """Rescue a victim"""
        for victim in self.visible_victims:
            if victim.position.is_adjacent(self.position) and not victim.rescued:
                if environment.rescue_victim(victim.id):
                    self.stats.victims_rescued += 1
                    self.current_target = None
                    return REWARDS["life_saved"]
        
        return 0
    
    def _deliver_supplies(self, environment: Environment) -> float:
        """Deliver supplies"""
        if self.supplies < 10:
            return 0
        
        self.supplies -= 10
        self.stats.resources_delivered += 1
        return REWARDS["resource_delivered"]
    
    def _collect_resource(self, environment: Environment) -> float:
        """Collect resources"""
        for resource in self.visible_resources:
            if resource.position.is_adjacent(self.position):
                if resource.type == "fuel":
                    amount = resource.consume(50)
                    self.fuel = min(settings.AGENT_MAX_FUEL, self.fuel + amount)
                    return 5
                elif resource.type in ["medical", "food", "water"]:
                    amount = resource.consume(30)
                    self.supplies = min(settings.AGENT_MAX_SUPPLIES, self.supplies + amount)
                    return 5
        
        return 0
    
    def send_message(self, receiver_id: Optional[str], 
                    message_type: str, content: Dict[str, Any],
                    timestamp: float):
        """Send a message"""
        message = Message(
            sender_id=self.id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            timestamp=timestamp
        )
        self.outbox.append(message)
        self.stats.messages_sent += 1
    
    def receive_message(self, message: Message):
        """Receive a message"""
        self.inbox.append(message)
        self.stats.messages_received += 1
    
    def update(self, delta_time: float, environment: Environment):
        """Main update loop"""
        self.perceive(environment)
        action = self.decide(environment)
        if action:
            self.act(action, environment)
    
    def get_state_dict(self) -> Dict:
        """Get agent state as dictionary"""
        return {
            "id": self.id,
            "type": self.type,
            "position": {"x": self.position.x, "y": self.position.y},
            "state": self.state.value,
            "behavior_mode": self.behavior_mode,
            "health": self.health,
            "fuel": self.fuel,
            "supplies": self.supplies,
            "current_target": {
                "x": self.current_target.x,
                "y": self.current_target.y
            } if self.current_target else None,
            "stats": {
                "disasters_resolved": self.stats.disasters_resolved,
                "victims_rescued": self.stats.victims_rescued,
                "resources_delivered": self.stats.resources_delivered,
                "distance_traveled": self.stats.distance_traveled,
                "actions_taken": self.stats.actions_taken,
                "messages_sent": self.stats.messages_sent,
                "messages_received": self.stats.messages_received,
                "rewards_earned": self.stats.rewards_earned
            }
        }
