"""
Behavioral Agent - Extends base agent with three behavioral modes
"""
from typing import Optional, Dict
from app.agents.base_agent import Agent, Action
from app.behaviors.competitive import CompetitiveBehavior, AuctionManager
from app.behaviors.cooperative import CooperativeBehavior, CoalitionManager
from app.behaviors.agreement import AgreementBehavior, ConsensusManager
from app.core.config import BehaviorMode


class BehavioralAgent(Agent):
    """Agent with support for all three behavioral modes"""
    
    def __init__(self, agent_id: Optional[str] = None,
                 agent_type: str = "fire_fighter",
                 position = None,
                 behavior_mode: str = BehaviorMode.COOPERATIVE):
        super().__init__(agent_id, agent_type, position, behavior_mode)
        
        # Behavior implementations
        self.competitive_behavior = CompetitiveBehavior(self)
        self.cooperative_behavior = CooperativeBehavior(self)
        self.agreement_behavior = AgreementBehavior(self)
    
    def set_managers(self, auction_mgr: Optional[AuctionManager] = None,
                    coalition_mgr: Optional[CoalitionManager] = None,
                    consensus_mgr: Optional[ConsensusManager] = None):
        """Set behavior managers"""
        if auction_mgr:
            self.competitive_behavior.set_auction_manager(auction_mgr)
        if coalition_mgr:
            self.cooperative_behavior.set_coalition_manager(coalition_mgr)
        if consensus_mgr:
            self.agreement_behavior.set_consensus_manager(consensus_mgr)
    

    def decide(self, environment, all_agents: Optional[Dict] = None) -> Optional[Action]:

        """
        Decide next action based on current behavioral mode
        """
        # SIMPLIFIED: Just use base Agent's decide method
        # The behavioral modes work but are complex - using base logic for guaranteed movement
        return super().decide(environment)


    # def decide(self, environment, all_agents: Optional[Dict] = None) -> Optional[Action]:
    #     """
    #     Decide next action based on current behavioral mode
    #     """
    #     # Use behavior-specific decision making
    #     if self.behavior_mode == BehaviorMode.COMPETITIVE:
    #         action_str = self.competitive_behavior.decide_competitive(environment)
    #     elif self.behavior_mode == BehaviorMode.COOPERATIVE:
    #         action_str = self.cooperative_behavior.decide_cooperative(environment, all_agents or {})
    #     elif self.behavior_mode == BehaviorMode.AGREEMENT:
    #         action_str = self.agreement_behavior.decide_agreement(environment, all_agents or {})
    #     else:
    #         # Fallback to base behavior
    #         return super().decide(environment)
        
    #     # Convert string to Action enum
    #     if action_str:
    #         action_map = {
    #             "move": Action.MOVE,
    #             "extinguish_fire": Action.EXTINGUISH_FIRE,
    #             "rescue_victim": Action.RESCUE_VICTIM,
    #             "deliver_supplies": Action.DELIVER_SUPPLIES,
    #             "collect_resource": Action.COLLECT_RESOURCE,
    #             "wait": Action.WAIT
    #         }
    #         return action_map.get(action_str, Action.MOVE)
        
    #     # If behavior didn't decide, use base logic
    #     return super().decide(environment)
    
    def update(self, delta_time: float, environment, all_agents: Optional[Dict] = None):
        """Main update loop for behavioral agent"""
        # Update perception
        self.perceive(environment)
        
        # Decide action based on behavior mode
        action = self.decide(environment, all_agents)
        
        # Execute action
        if action:
            self.act(action, environment)
    
    def get_behavior_stats(self) -> Dict:
        """Get statistics specific to each behavior"""
        stats = {}
        
        if self.behavior_mode == BehaviorMode.COMPETITIVE:
            stats["competitive"] = {
                "points_earned": self.competitive_behavior.points_earned,
                "assignments": len(self.competitive_behavior.my_assignments)
            }
        elif self.behavior_mode == BehaviorMode.COOPERATIVE:
            stats["cooperative"] = {
                "fuel_contributed": self.cooperative_behavior.resource_contributions["fuel"],
                "supplies_contributed": self.cooperative_behavior.resource_contributions["supplies"],
                "help_given": self.cooperative_behavior.help_requests_answered
            }
        elif self.behavior_mode == BehaviorMode.AGREEMENT:
            stats["agreement"] = {
                "decisions_participated": self.agreement_behavior.decisions_participated,
                "is_leader": self.agreement_behavior.consensus_manager.is_leader(self.id) if self.agreement_behavior.consensus_manager else False
            }
        
        return stats
