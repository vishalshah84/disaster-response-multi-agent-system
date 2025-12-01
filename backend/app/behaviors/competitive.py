"""
Competitive Behavior - Auction-based resource allocation with game theory
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from app.core.config import REWARDS
from app.core.utils import Position


@dataclass
class Bid:
    """Represents a bid for a resource or disaster"""
    agent_id: str
    target_id: str  # disaster_id or resource_id
    bid_amount: float
    urgency: float  # 0.0 to 1.0
    capability: float  # 0.0 to 1.0 - how capable is agent for this task
    
    def calculate_score(self) -> float:
        """Calculate bid score for comparison"""
        return self.bid_amount * (0.5 + 0.3 * self.urgency + 0.2 * self.capability)


class AuctionManager:
    """Manages auctions for disaster assignments and resource allocation"""
    
    def __init__(self):
        self.active_auctions: Dict[str, List[Bid]] = {}  # target_id -> list of bids
        self.assignments: Dict[str, str] = {}  # target_id -> agent_id
        self.auction_history: List[Dict] = []
    
    def create_auction(self, target_id: str, target_type: str):
        """Create a new auction for a disaster or resource"""
        if target_id not in self.active_auctions:
            self.active_auctions[target_id] = []
    
    def place_bid(self, agent_id: str, target_id: str, bid_amount: float, 
                  urgency: float, capability: float):
        """Agent places a bid"""
        if target_id not in self.active_auctions:
            self.create_auction(target_id, "unknown")
        
        bid = Bid(
            agent_id=agent_id,
            target_id=target_id,
            bid_amount=bid_amount,
            urgency=urgency,
            capability=capability
        )
        
        self.active_auctions[target_id].append(bid)
    
    def resolve_auction(self, target_id: str) -> Optional[str]:
        """
        Resolve auction using Vickrey (second-price) auction
        Returns winning agent_id
        """
        if target_id not in self.active_auctions or not self.active_auctions[target_id]:
            return None
        
        bids = self.active_auctions[target_id]
        
        # Sort bids by score (descending)
        sorted_bids = sorted(bids, key=lambda b: b.calculate_score(), reverse=True)
        
        if not sorted_bids:
            return None
        
        # Winner is highest bidder
        winner = sorted_bids[0]
        
        # In Vickrey auction, winner pays second-highest price
        second_price = sorted_bids[1].bid_amount if len(sorted_bids) > 1 else winner.bid_amount
        
        # Record assignment
        self.assignments[target_id] = winner.agent_id
        
        # Record history
        self.auction_history.append({
            "target_id": target_id,
            "winner": winner.agent_id,
            "winning_bid": winner.bid_amount,
            "price_paid": second_price,
            "num_bidders": len(bids)
        })
        
        # Clear auction
        del self.active_auctions[target_id]
        
        return winner.agent_id
    
    def get_assignment(self, target_id: str) -> Optional[str]:
        """Get current assignment for a target"""
        return self.assignments.get(target_id)
    
    def clear_assignment(self, target_id: str):
        """Clear assignment when task is completed"""
        if target_id in self.assignments:
            del self.assignments[target_id]


class CompetitiveBehavior:
    """Competitive behavior implementation for agents"""
    
    def __init__(self, agent):
        self.agent = agent
        self.auction_manager: Optional[AuctionManager] = None
        self.my_assignments: List[str] = []
        self.points_earned = 0
    
    def set_auction_manager(self, manager: AuctionManager):
        """Set reference to global auction manager"""
        self.auction_manager = manager
    
    def calculate_bid_amount(self, target, distance: float) -> float:
        """
        Calculate bid amount based on:
        - Distance to target (closer = higher bid)
        - Agent's current resources (more resources = higher bid)
        - Competition level
        """
        # Base bid on agent's available resources
        resource_factor = (self.agent.fuel / 100.0) * 0.5 + (self.agent.supplies / 50.0) * 0.5
        
        # Distance factor (closer is better)
        max_distance = 50.0
        distance_factor = max(0, 1.0 - (distance / max_distance))
        
        # Calculate bid
        base_bid = 10.0
        bid = base_bid * resource_factor * (0.3 + 0.7 * distance_factor)
        
        # Add some randomness for variety
        bid *= np.random.uniform(0.9, 1.1)
        
        return max(1.0, bid)
    
    def calculate_urgency(self, target) -> float:
        """Calculate urgency of handling this target"""
        if hasattr(target, 'intensity'):  # Disaster
            return target.intensity
        elif hasattr(target, 'health'):  # Victim
            return 1.0 - target.health
        else:
            return 0.5
    
    def calculate_capability(self, target) -> float:
        """Calculate agent's capability for this target"""
        # Agent type matching
        if hasattr(target, 'type'):  # Disaster
            if target.type == "fire" and self.agent.type == "fire_fighter":
                return 1.0
            elif self.agent.type == "medical":
                return 0.7
            else:
                return 0.5
        
        return 0.5
    
    def decide_competitive(self, environment) -> Optional[str]:
        """
        Competitive decision making:
        1. Identify available disasters/victims
        2. Calculate bids for each
        3. Place bids in auctions
        4. Act on won assignments
        """
        if not self.auction_manager:
            return None
        
        # Check if we have any existing assignments
        for target_id in self.my_assignments[:]:
            # Check if assignment is still valid
            assigned_agent = self.auction_manager.get_assignment(target_id)
            if assigned_agent != self.agent.id:
                self.my_assignments.remove(target_id)
                continue
            
            # Work on this assignment
            return self._work_on_assignment(target_id, environment)
        
        # No assignments, participate in auctions
        self._participate_in_auctions(environment)
        
        return None
    
    def _participate_in_auctions(self, environment):
        """Scan for opportunities and place bids"""
        # Bid on visible disasters
        for disaster in self.agent.visible_disasters:
            if disaster.id in self.auction_manager.assignments:
                continue  # Already assigned
            
            distance = self.agent.position.distance_to(disaster.position)
            bid_amount = self.calculate_bid_amount(disaster, distance)
            urgency = self.calculate_urgency(disaster)
            capability = self.calculate_capability(disaster)
            
            self.auction_manager.place_bid(
                self.agent.id,
                disaster.id,
                bid_amount,
                urgency,
                capability
            )
        
        # Bid on victims
        for victim in self.agent.visible_victims:
            if victim.id in self.auction_manager.assignments:
                continue
            
            distance = self.agent.position.distance_to(victim.position)
            bid_amount = self.calculate_bid_amount(victim, distance)
            urgency = self.calculate_urgency(victim)
            capability = 0.8 if self.agent.type in ["medical", "search_rescue"] else 0.5
            
            self.auction_manager.place_bid(
                self.agent.id,
                victim.id,
                bid_amount,
                urgency,
                capability
            )
    
    def _work_on_assignment(self, target_id: str, environment):
        """Work on assigned target"""
        # Find the target
        target = None
        
        # Check disasters
        if target_id in environment.disasters:
            target = environment.disasters[target_id]
            target_type = "disaster"
        
        # Check victims
        elif target_id in environment.victims:
            target = environment.victims[target_id]
            target_type = "victim"
        
        if not target:
            # Target no longer exists, clear assignment
            self.auction_manager.clear_assignment(target_id)
            self.my_assignments.remove(target_id)
            return None
        
        # Move towards target
        self.agent.current_target = target.position
        
        # If adjacent, perform action
        if self.agent.position.is_adjacent(target.position):
            if target_type == "disaster":
                return "extinguish_fire"
            elif target_type == "victim":
                return "rescue_victim"
        
        return "move"
    
    def on_task_completed(self, target_id: str):
        """Called when agent completes a task"""
        # Award points
        self.points_earned += 10
        
        # Clear assignment
        if target_id in self.my_assignments:
            self.my_assignments.remove(target_id)
        
        self.auction_manager.clear_assignment(target_id)


def calculate_nash_equilibrium(agents: List, resources: List) -> Dict[str, List]:
    """
    Calculate Nash Equilibrium for agent-resource allocation
    Simplified version using best response dynamics
    """
    # Create utility matrix
    n_agents = len(agents)
    n_resources = len(resources)
    
    # Initialize strategy (each agent chooses a resource)
    strategy = {agent.id: 0 for agent in agents}  # agent_id -> resource_index
    
    # Iterate until convergence (simplified)
    max_iterations = 100
    for iteration in range(max_iterations):
        changed = False
        
        for i, agent in enumerate(agents):
            # Find best response for this agent
            best_resource = 0
            best_utility = -float('inf')
            
            for r_idx in range(n_resources):
                # Calculate utility if agent chooses this resource
                # Utility decreases with competition
                competitors = sum(1 for aid, ridx in strategy.items() if ridx == r_idx and aid != agent.id)
                distance = agent.position.distance_to(resources[r_idx].position) if r_idx < len(resources) else 100
                
                utility = 100 / (1 + competitors) - distance * 0.5
                
                if utility > best_utility:
                    best_utility = utility
                    best_resource = r_idx
            
            if strategy[agent.id] != best_resource:
                strategy[agent.id] = best_resource
                changed = True
        
        if not changed:
            break  # Reached equilibrium
    
    # Convert to assignment format
    assignments = {}
    for agent_id, resource_idx in strategy.items():
        if resource_idx < len(resources):
            resource_id = resources[resource_idx].id
            if resource_id not in assignments:
                assignments[resource_id] = []
            assignments[resource_id].append(agent_id)
    
    return assignments
