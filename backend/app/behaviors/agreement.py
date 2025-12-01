"""
Agreement Behavior - Consensus-based decision making with Raft protocol
"""
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import time
import random


class NodeState(Enum):
    """Raft node states"""
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


@dataclass
class LogEntry:
    """Entry in the replicated log"""
    term: int
    command: str
    data: Dict
    index: int


@dataclass
class VoteRequest:
    """Request for votes during election"""
    candidate_id: str
    term: int
    last_log_index: int
    last_log_term: int


@dataclass
class VoteResponse:
    """Response to vote request"""
    voter_id: str
    term: int
    vote_granted: bool


@dataclass
class Proposal:
    """Proposal for group decision"""
    id: str
    proposer_id: str
    proposal_type: str  # "target_assignment", "resource_distribution", "priority_change"
    data: Dict
    votes_for: Set[str] = field(default_factory=set)
    votes_against: Set[str] = field(default_factory=set)
    status: str = "pending"  # "pending", "accepted", "rejected"
    timestamp: float = 0.0


class ConsensusNode:
    """Raft consensus node for an agent"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.state = NodeState.FOLLOWER
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.log: List[LogEntry] = []
        self.commit_index = 0
        self.last_applied = 0
        
        # Leader-specific
        self.next_index: Dict[str, int] = {}
        self.match_index: Dict[str, int] = {}
        
        # Timing
        self.last_heartbeat = time.time()
        self.election_timeout = random.uniform(5.0, 10.0)  # seconds
    
    def reset_election_timeout(self):
        """Reset election timeout with randomization"""
        self.election_timeout = random.uniform(5.0, 10.0)
        self.last_heartbeat = time.time()
    
    def should_start_election(self) -> bool:
        """Check if should start election"""
        if self.state == NodeState.LEADER:
            return False
        
        time_since_heartbeat = time.time() - self.last_heartbeat
        return time_since_heartbeat > self.election_timeout
    
    def start_election(self, all_nodes: List[str]):
        """Start election for leadership"""
        self.state = NodeState.CANDIDATE
        self.current_term += 1
        self.voted_for = self.agent_id
        self.reset_election_timeout()
        
        # Vote for self
        votes = 1
        
        return votes
    
    def receive_vote_request(self, request: VoteRequest) -> VoteResponse:
        """Handle vote request from candidate"""
        # Update term if necessary
        if request.term > self.current_term:
            self.current_term = request.term
            self.state = NodeState.FOLLOWER
            self.voted_for = None
        
        # Grant vote if haven't voted and candidate's log is up-to-date
        vote_granted = False
        
        if request.term == self.current_term:
            if self.voted_for is None or self.voted_for == request.candidate_id:
                # Check if candidate's log is at least as up-to-date as ours
                last_log_index = len(self.log) - 1
                last_log_term = self.log[-1].term if self.log else 0
                
                if (request.last_log_term > last_log_term or 
                    (request.last_log_term == last_log_term and 
                     request.last_log_index >= last_log_index)):
                    vote_granted = True
                    self.voted_for = request.candidate_id
                    self.reset_election_timeout()
        
        return VoteResponse(
            voter_id=self.agent_id,
            term=self.current_term,
            vote_granted=vote_granted
        )
    
    def become_leader(self):
        """Transition to leader state"""
        self.state = NodeState.LEADER
        print(f"Agent {self.agent_id} became leader for term {self.current_term}")
    
    def append_entry(self, command: str, data: Dict):
        """Leader appends entry to log"""
        if self.state != NodeState.LEADER:
            return False
        
        entry = LogEntry(
            term=self.current_term,
            command=command,
            data=data,
            index=len(self.log)
        )
        self.log.append(entry)
        return True
    
    def receive_heartbeat(self, leader_id: str, term: int):
        """Receive heartbeat from leader"""
        if term >= self.current_term:
            self.current_term = term
            self.state = NodeState.FOLLOWER
            self.reset_election_timeout()


class ConsensusManager:
    """Manages consensus protocol for agent group"""
    
    def __init__(self):
        self.nodes: Dict[str, ConsensusNode] = {}
        self.current_leader: Optional[str] = None
        self.proposals: Dict[str, Proposal] = {}
        self.proposal_counter = 0
        self.decisions: List[Dict] = []
    
    def add_node(self, agent_id: str):
        """Add agent to consensus group"""
        if agent_id not in self.nodes:
            self.nodes[agent_id] = ConsensusNode(agent_id)
    
    def remove_node(self, agent_id: str):
        """Remove agent from consensus group"""
        if agent_id in self.nodes:
            del self.nodes[agent_id]
            if self.current_leader == agent_id:
                self.current_leader = None
    
    def update(self, delta_time: float):
        """Update consensus state"""
        # Check for elections
        for agent_id, node in self.nodes.items():
            if node.should_start_election():
                self._conduct_election(agent_id)
        
        # Leader sends heartbeats
        if self.current_leader and self.current_leader in self.nodes:
            self._send_heartbeats()
    
    def _conduct_election(self, candidate_id: str):
        """Conduct election"""
        candidate = self.nodes[candidate_id]
        
        # Start election
        votes = candidate.start_election(list(self.nodes.keys()))
        
        # Request votes from other nodes
        last_log_index = len(candidate.log) - 1
        last_log_term = candidate.log[-1].term if candidate.log else 0
        
        request = VoteRequest(
            candidate_id=candidate_id,
            term=candidate.current_term,
            last_log_index=last_log_index,
            last_log_term=last_log_term
        )
        
        # Collect votes
        for other_id, other_node in self.nodes.items():
            if other_id == candidate_id:
                continue
            
            response = other_node.receive_vote_request(request)
            if response.vote_granted:
                votes += 1
        
        # Check if won election (majority)
        total_nodes = len(self.nodes)
        if votes > total_nodes / 2:
            candidate.become_leader()
            self.current_leader = candidate_id
    
    def _send_heartbeats(self):
        """Leader sends heartbeats to followers"""
        if not self.current_leader or self.current_leader not in self.nodes:
            return
        
        leader = self.nodes[self.current_leader]
        
        for other_id, other_node in self.nodes.items():
            if other_id != self.current_leader:
                other_node.receive_heartbeat(self.current_leader, leader.current_term)
    
    def create_proposal(self, proposer_id: str, proposal_type: str, 
                       data: Dict) -> Proposal:
        """Create a new proposal for group decision"""
        proposal_id = f"proposal_{self.proposal_counter}"
        self.proposal_counter += 1
        
        proposal = Proposal(
            id=proposal_id,
            proposer_id=proposer_id,
            proposal_type=proposal_type,
            data=data,
            timestamp=time.time()
        )
        
        self.proposals[proposal_id] = proposal
        return proposal
    
    def vote_on_proposal(self, agent_id: str, proposal_id: str, vote: bool):
        """Agent votes on a proposal"""
        if proposal_id not in self.proposals:
            return
        
        proposal = self.proposals[proposal_id]
        
        if vote:
            proposal.votes_for.add(agent_id)
            proposal.votes_against.discard(agent_id)
        else:
            proposal.votes_against.add(agent_id)
            proposal.votes_for.discard(agent_id)
        
        # Check if decision reached
        total_votes = len(proposal.votes_for) + len(proposal.votes_against)
        if total_votes >= len(self.nodes):
            self._finalize_proposal(proposal_id)
    
    def _finalize_proposal(self, proposal_id: str):
        """Finalize proposal after all votes collected"""
        proposal = self.proposals[proposal_id]
        
        # Majority wins
        if len(proposal.votes_for) > len(proposal.votes_against):
            proposal.status = "accepted"
            self._execute_proposal(proposal)
        else:
            proposal.status = "rejected"
        
        # Record decision
        self.decisions.append({
            "proposal_id": proposal_id,
            "type": proposal.proposal_type,
            "status": proposal.status,
            "votes_for": len(proposal.votes_for),
            "votes_against": len(proposal.votes_against)
        })
    
    def _execute_proposal(self, proposal: Proposal):
        """Execute accepted proposal"""
        # Log to leader's replicated log
        if self.current_leader and self.current_leader in self.nodes:
            leader = self.nodes[self.current_leader]
            leader.append_entry(
                command=proposal.proposal_type,
                data=proposal.data
            )
    
    def get_leader(self) -> Optional[str]:
        """Get current leader"""
        return self.current_leader
    
    def is_leader(self, agent_id: str) -> bool:
        """Check if agent is current leader"""
        return self.current_leader == agent_id


class AgreementBehavior:
    """Agreement-based behavior implementation for agents"""
    
    def __init__(self, agent):
        self.agent = agent
        self.consensus_manager: Optional[ConsensusManager] = None
        self.pending_proposals: List[str] = []
        self.decisions_participated = 0
    
    def set_consensus_manager(self, manager: ConsensusManager):
        """Set reference to global consensus manager"""
        self.consensus_manager = manager
        self.consensus_manager.add_node(self.agent.id)
    
    def decide_agreement(self, environment, all_agents: Dict) -> Optional[str]:
        """
        Agreement-based decision making:
        1. Participate in leader election
        2. If leader, propose actions
        3. If follower, vote on proposals
        4. Execute agreed-upon decisions
        """
        if not self.consensus_manager:
            return None
        
        # Check if we're the leader
        is_leader = self.consensus_manager.is_leader(self.agent.id)
        
        if is_leader:
            return self._lead_with_consensus(environment, all_agents)
        else:
            return self._follow_with_consensus(environment, all_agents)
    
    def _lead_with_consensus(self, environment, all_agents: Dict) -> Optional[str]:
        """Leader proposes actions for the group"""
        # Identify highest priority disaster
        high_priority = self._identify_priority_targets(environment)
        
        if high_priority:
            # Create proposal for group to tackle this disaster
            proposal = self.consensus_manager.create_proposal(
                proposer_id=self.agent.id,
                proposal_type="target_assignment",
                data={
                    "target_id": high_priority.id,
                    "target_type": "disaster",
                    "priority": high_priority.intensity,
                    "recommended_agents": 3
                }
            )
            
            # Vote on own proposal
            self.consensus_manager.vote_on_proposal(
                self.agent.id, proposal.id, True
            )
            
            self.pending_proposals.append(proposal.id)
        
        # Execute current consensus
        return self._execute_consensus_action(environment)
    
    def _follow_with_consensus(self, environment, all_agents: Dict) -> Optional[str]:
        """Follower votes on proposals and executes decisions"""
        # Vote on pending proposals
        for proposal_id in list(self.pending_proposals):
            if proposal_id in self.consensus_manager.proposals:
                proposal = self.consensus_manager.proposals[proposal_id]
                
                if proposal.status == "pending":
                    # Decide whether to vote for or against
                    vote = self._evaluate_proposal(proposal, environment)
                    self.consensus_manager.vote_on_proposal(
                        self.agent.id, proposal_id, vote
                    )
                    self.decisions_participated += 1
                else:
                    self.pending_proposals.remove(proposal_id)
        
        # Execute current consensus
        return self._execute_consensus_action(environment)
    
    def _evaluate_proposal(self, proposal: Proposal, environment) -> bool:
        """Evaluate whether to vote for a proposal"""
        if proposal.proposal_type == "target_assignment":
            target_id = proposal.data.get("target_id")
            
            # Vote yes if we can reach it and have resources
            if target_id in environment.disasters:
                disaster = environment.disasters[target_id]
                distance = self.agent.position.distance_to(disaster.position)
                
                # Vote yes if close and have resources
                if distance < 20 and self.agent.fuel > 30 and self.agent.supplies > 20:
                    return True
        
        # Default: vote based on priority
        priority = proposal.data.get("priority", 0.5)
        return priority > 0.6
    
    def _execute_consensus_action(self, environment) -> Optional[str]:
        """Execute action based on current consensus"""
        # Find accepted proposals
        for proposal in self.consensus_manager.proposals.values():
            if proposal.status == "accepted":
                if proposal.proposal_type == "target_assignment":
                    target_id = proposal.data["target_id"]
                    
                    # Find target
                    if target_id in environment.disasters:
                        disaster = environment.disasters[target_id]
                        self.agent.current_target = disaster.position
                        
                        # If adjacent, act
                        if self.agent.position.is_adjacent(disaster.position):
                            return "extinguish_fire"
                        return "move"
        
        return None
    
    def _identify_priority_targets(self, environment) -> Optional:
        """Identify highest priority target"""
        if not self.agent.visible_disasters:
            return None
        
        # Sort by intensity
        sorted_disasters = sorted(
            self.agent.visible_disasters,
            key=lambda d: d.intensity,
            reverse=True
        )
        
        return sorted_disasters[0] if sorted_disasters else None


def calculate_byzantine_fault_tolerance(total_nodes: int, faulty_nodes: int) -> bool:
    """
    Check if system can tolerate faulty nodes
    BFT requires: total_nodes >= 3 * faulty_nodes + 1
    """
    return total_nodes >= 3 * faulty_nodes + 1


def calculate_quorum_size(total_nodes: int) -> int:
    """Calculate quorum size (majority)"""
    return (total_nodes // 2) + 1


def simulate_consensus_rounds(num_nodes: int, num_proposals: int) -> Dict:
    """Simulate consensus rounds and calculate statistics"""
    results = {
        "total_proposals": num_proposals,
        "accepted": 0,
        "rejected": 0,
        "avg_rounds": 0,
        "avg_messages": 0
    }
    
    total_rounds = 0
    total_messages = 0
    
    for _ in range(num_proposals):
        # Simulate voting rounds
        rounds = random.randint(1, 3)
        messages = num_nodes * rounds * 2  # Request + response
        
        # Random outcome
        if random.random() > 0.3:
            results["accepted"] += 1
        else:
            results["rejected"] += 1
        
        total_rounds += rounds
        total_messages += messages
    
    results["avg_rounds"] = total_rounds / num_proposals
    results["avg_messages"] = total_messages / num_proposals
    
    return results
