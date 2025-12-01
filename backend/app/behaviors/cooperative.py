"""
Cooperative Behavior - Coalition formation and resource sharing
"""
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

from app.core.config import REWARDS
from app.core.utils import Position


@dataclass
class Coalition:
    """Represents a coalition of cooperating agents"""
    id: str
    leader_id: str
    member_ids: Set[str] = field(default_factory=set)
    shared_resources: Dict[str, float] = field(default_factory=dict)  # resource_type -> amount
    target_id: Optional[str] = None  # Current shared objective
    formation_time: float = 0.0
    success_count: int = 0
    
    def add_member(self, agent_id: str):
        """Add member to coalition"""
        self.member_ids.add(agent_id)
    
    def remove_member(self, agent_id: str):
        """Remove member from coalition"""
        if agent_id in self.member_ids:
            self.member_ids.remove(agent_id)
    
    def is_member(self, agent_id: str) -> bool:
        """Check if agent is member"""
        return agent_id in self.member_ids or agent_id == self.leader_id
    
    def get_all_members(self) -> List[str]:
        """Get all member IDs including leader"""
        return [self.leader_id] + list(self.member_ids)
    
    def share_resource(self, resource_type: str, amount: float):
        """Add resources to shared pool"""
        if resource_type not in self.shared_resources:
            self.shared_resources[resource_type] = 0
        self.shared_resources[resource_type] += amount
    
    def take_resource(self, resource_type: str, amount: float) -> float:
        """Take resources from shared pool"""
        if resource_type not in self.shared_resources:
            return 0
        
        actual_amount = min(amount, self.shared_resources[resource_type])
        self.shared_resources[resource_type] -= actual_amount
        return actual_amount


class CoalitionManager:
    """Manages coalition formation and coordination"""
    
    def __init__(self):
        self.coalitions: Dict[str, Coalition] = {}  # coalition_id -> Coalition
        self.agent_to_coalition: Dict[str, str] = {}  # agent_id -> coalition_id
        self.coalition_counter = 0
    
    def create_coalition(self, leader_id: str, target_id: Optional[str] = None) -> Coalition:
        """Create a new coalition"""
        coalition_id = f"coalition_{self.coalition_counter}"
        self.coalition_counter += 1
        
        coalition = Coalition(
            id=coalition_id,
            leader_id=leader_id,
            target_id=target_id
        )
        
        self.coalitions[coalition_id] = coalition
        self.agent_to_coalition[leader_id] = coalition_id
        
        return coalition
    
    def join_coalition(self, agent_id: str, coalition_id: str) -> bool:
        """Agent joins a coalition"""
        if coalition_id not in self.coalitions:
            return False
        
        # Leave current coalition if any
        self.leave_coalition(agent_id)
        
        self.coalitions[coalition_id].add_member(agent_id)
        self.agent_to_coalition[agent_id] = coalition_id
        return True
    
    def leave_coalition(self, agent_id: str):
        """Agent leaves their coalition"""
        if agent_id not in self.agent_to_coalition:
            return
        
        coalition_id = self.agent_to_coalition[agent_id]
        coalition = self.coalitions[coalition_id]
        
        coalition.remove_member(agent_id)
        del self.agent_to_coalition[agent_id]
        
        # Dissolve coalition if empty
        if len(coalition.member_ids) == 0 and coalition.leader_id not in self.agent_to_coalition:
            del self.coalitions[coalition_id]
    
    def get_coalition(self, agent_id: str) -> Optional[Coalition]:
        """Get agent's coalition"""
        if agent_id not in self.agent_to_coalition:
            return None
        return self.coalitions.get(self.agent_to_coalition[agent_id])
    
    def find_nearby_coalitions(self, position: Position, radius: float, 
                              agents: Dict) -> List[Coalition]:
        """Find coalitions with members near a position"""
        nearby = []
        
        for coalition in self.coalitions.values():
            for member_id in coalition.get_all_members():
                if member_id in agents:
                    agent = agents[member_id]
                    if agent.position.distance_to(position) <= radius:
                        nearby.append(coalition)
                        break
        
        return nearby


class CooperativeBehavior:
    """Cooperative behavior implementation for agents"""
    
    def __init__(self, agent):
        self.agent = agent
        self.coalition_manager: Optional[CoalitionManager] = None
        self.resource_contributions = {"fuel": 0, "supplies": 0}
        self.help_requests_sent = 0
        self.help_requests_answered = 0
    
    def set_coalition_manager(self, manager: CoalitionManager):
        """Set reference to global coalition manager"""
        self.coalition_manager = manager
    
    def decide_cooperative(self, environment, all_agents: Dict) -> Optional[str]:
        """
        Cooperative decision making:
        1. Check if in coalition
        2. If not, consider forming/joining one
        3. Coordinate with coalition members
        4. Share resources when needed
        """
        if not self.coalition_manager:
            return None
        
        my_coalition = self.coalition_manager.get_coalition(self.agent.id)
        
        if not my_coalition:
            # Not in coalition, consider forming or joining one
            self._consider_coalition(environment, all_agents)
            my_coalition = self.coalition_manager.get_coalition(self.agent.id)
        
        if my_coalition:
            # Work with coalition
            return self._coordinate_with_coalition(my_coalition, environment, all_agents)
        else:
            # Work independently but help others
            return self._independent_cooperative_action(environment, all_agents)
    
    def _consider_coalition(self, environment, all_agents: Dict):
        """Decide whether to form or join a coalition"""
        # Check for nearby disasters that benefit from cooperation
        high_priority_disasters = [
            d for d in self.agent.visible_disasters 
            if d.intensity > 0.5 or d.victims_count > 2
        ]
        
        if not high_priority_disasters:
            return
        
        target_disaster = high_priority_disasters[0]
        
        # Look for nearby coalitions working on this
        nearby_coalitions = self.coalition_manager.find_nearby_coalitions(
            target_disaster.position, 10.0, all_agents
        )
        
        if nearby_coalitions:
            # Join existing coalition
            best_coalition = nearby_coalitions[0]
            self.coalition_manager.join_coalition(self.agent.id, best_coalition.id)
        else:
            # Form new coalition
            coalition = self.coalition_manager.create_coalition(
                self.agent.id, 
                target_disaster.id
            )
            
            # Invite nearby agents
            self._invite_nearby_agents(coalition, all_agents, target_disaster.position)
    
    def _invite_nearby_agents(self, coalition: Coalition, all_agents: Dict, 
                             target_position: Position):
        """Invite nearby agents to join coalition"""
        for agent_id, agent in all_agents.items():
            if agent_id == self.agent.id:
                continue
            
            # Check if agent is nearby and not in coalition
            if agent.position.distance_to(target_position) <= 8.0:
                existing_coalition = self.coalition_manager.get_coalition(agent_id)
                if not existing_coalition:
                    # Send invitation (simplified - direct join)
                    self.coalition_manager.join_coalition(agent_id, coalition.id)
    
    def _coordinate_with_coalition(self, coalition: Coalition, 
                                   environment, all_agents: Dict) -> Optional[str]:
        """Coordinate actions with coalition members"""
        # If coalition has a target, work on it
        if coalition.target_id:
            target = self._find_target(coalition.target_id, environment)
            
            if target:
                # Set as current target
                self.agent.current_target = target.position
                
                # Share resources if needed
                self._share_resources_with_coalition(coalition, all_agents)
                
                # Determine role in coalition
                if coalition.leader_id == self.agent.id:
                    # Leader coordinates
                    return self._lead_coalition(coalition, target, environment, all_agents)
                else:
                    # Member follows leader's plan
                    return self._follow_coalition_plan(coalition, target, environment, all_agents)
            else:
                # Target completed, dissolve coalition
                coalition.success_count += 1
                coalition.target_id = None
        
        return None
    
    def _lead_coalition(self, coalition: Coalition, target, 
                       environment, all_agents: Dict) -> Optional[str]:
        """Leader's decision making for coalition"""
        # Assess coalition strength
        total_fuel = sum(
            all_agents[aid].fuel for aid in coalition.get_all_members() 
            if aid in all_agents
        )
        total_supplies = sum(
            all_agents[aid].supplies for aid in coalition.get_all_members() 
            if aid in all_agents
        )
        
        # If coalition is well-resourced, attack target aggressively
        if total_supplies > 100:
            if self.agent.position.is_adjacent(target.position):
                return "extinguish_fire" if hasattr(target, 'intensity') else "rescue_victim"
            return "move"
        else:
            # Need to gather resources first
            if self.agent.fuel < 30:
                # Look for fuel
                fuel_resources = [r for r in self.agent.visible_resources if r.type == "fuel"]
                if fuel_resources:
                    self.agent.current_target = fuel_resources[0].position
                    return "collect_resource"
        
        return "move"
    
    def _follow_coalition_plan(self, coalition: Coalition, target, 
                              environment, all_agents: Dict) -> Optional[str]:
        """Member's actions following coalition plan"""
        # Support leader's efforts
        leader = all_agents.get(coalition.leader_id)
        
        if leader:
            # Stay close to leader
            if self.agent.position.distance_to(leader.position) > 5.0:
                self.agent.current_target = leader.position
                return "move"
        
        # Work on target if adjacent
        if self.agent.position.is_adjacent(target.position):
            return "extinguish_fire" if hasattr(target, 'intensity') else "rescue_victim"
        
        # Move towards target
        self.agent.current_target = target.position
        return "move"
    
    def _share_resources_with_coalition(self, coalition: Coalition, all_agents: Dict):
        """Share resources with coalition members who need them"""
        # If we have excess, contribute to shared pool
        if self.agent.fuel > 80:
            contribution = 20
            self.agent.fuel -= contribution
            coalition.share_resource("fuel", contribution)
            self.resource_contributions["fuel"] += contribution
        
        if self.agent.supplies > 40:
            contribution = 10
            self.agent.supplies -= contribution
            coalition.share_resource("supplies", contribution)
            self.resource_contributions["supplies"] += contribution
        
        # If we're low, take from shared pool
        if self.agent.fuel < 20:
            taken = coalition.take_resource("fuel", 20)
            self.agent.fuel += taken
        
        if self.agent.supplies < 10:
            taken = coalition.take_resource("supplies", 10)
            self.agent.supplies += taken
    
    def _independent_cooperative_action(self, environment, all_agents: Dict) -> Optional[str]:
        """Act independently but help others"""
        # Look for agents in trouble
        for agent in all_agents.values():
            if agent.id == self.agent.id:
                continue
            
            # Check if agent is low on resources and nearby
            if agent.position.distance_to(self.agent.position) <= 5.0:
                if agent.fuel < 20 and self.agent.fuel > 50:
                    # Help by sharing location of fuel resources
                    fuel_resources = [r for r in self.agent.visible_resources if r.type == "fuel"]
                    if fuel_resources:
                        # In full implementation, send message to agent
                        self.help_requests_answered += 1
        
        # Continue with normal actions
        return None
    
    def _find_target(self, target_id: str, environment):
        """Find target by ID"""
        if target_id in environment.disasters:
            return environment.disasters[target_id]
        elif target_id in environment.victims:
            return environment.victims[target_id]
        return None


def calculate_pareto_optimal_allocation(agents: List, tasks: List) -> Dict[str, str]:
    """
    Calculate Pareto optimal task allocation
    No agent can be made better off without making another worse off
    """
    # Build cost matrix (agent x task)
    n_agents = len(agents)
    n_tasks = len(tasks)
    
    cost_matrix = np.zeros((n_agents, n_tasks))
    
    for i, agent in enumerate(agents):
        for j, task in enumerate(tasks):
            # Cost = distance + resource cost
            distance = agent.position.distance_to(task.position)
            resource_cost = 10 if agent.fuel < 50 else 0
            cost_matrix[i, j] = distance + resource_cost
    
    # Simple greedy allocation (for demonstration)
    allocation = {}
    used_agents = set()
    used_tasks = set()
    
    # Sort all agent-task pairs by cost
    pairs = []
    for i in range(n_agents):
        for j in range(n_tasks):
            pairs.append((cost_matrix[i, j], i, j))
    
    pairs.sort()
    
    # Assign greedily
    for cost, agent_idx, task_idx in pairs:
        if agent_idx not in used_agents and task_idx not in used_tasks:
            allocation[tasks[task_idx].id] = agents[agent_idx].id
            used_agents.add(agent_idx)
            used_tasks.add(task_idx)
    
    return allocation


def calculate_social_welfare(agents: List, allocation: Dict) -> float:
    """
    Calculate total social welfare (sum of utilities)
    Higher is better for the group
    """
    total_welfare = 0.0
    
    for agent in agents:
        # Utility = resources + success rate - effort
        utility = agent.fuel * 0.5 + agent.supplies * 1.0
        utility += agent.stats.disasters_resolved * 50
        utility += agent.stats.victims_rescued * 100
        utility -= agent.stats.distance_traveled * 0.1
        
        total_welfare += utility
    
    return total_welfare
