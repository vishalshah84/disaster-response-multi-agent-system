"""
Integrated Pipeline - Connects Devika's search with my response
"""
from typing import List, Tuple, Optional
import numpy as np
from dataclasses import dataclass

from app.robot_search.environment import WorldConfig, build_environment
from app.robot_search.robot import RobotConfig, init_robots
from app.robot_search.phase1_markov import Phase1Config, run_phase1
from app.robot_search.phase2_navigation import Phase2Config, run_phase2
from app.simulation_engine import SimulationEngine, create_simulation
from app.core.config import DisasterType
from app.core.utils import Position


@dataclass
class IntegratedResult:
    """Results from integrated simulation"""
    phase1_complete: bool
    phase2_complete: bool
    search_time: float
    response_time: float
    victims_discovered: int
    victims_rescued: int
    total_time: float


class IntegratedSimulation:
    """
    Runs Devika's robot search (Phase 1), then Devika's navigation (Phase 2)
    """
    
    def __init__(self, n_robots: int = 20, n_victims: int = 2, n_false: int = 10):
        self.phase = "idle"
        self.n_robots = n_robots
        self.n_victims = n_victims
        self.n_false = n_false
        
        # Phase 1 (Devika's)
        self.world_cfg = None
        self.robots = []
        self.victims = []
        self.false_sites = []
        self.phase1_result = None
        self.phase2_result = None
        self.discovered_positions = []
        self.robot_history = []
        
        # Phase 2 (My response system)
        self.your_sim = None
        self.response_metrics = None
        
        # Timing
        self.search_start_time = 0
        self.search_end_time = 0
        self.response_start_time = 0
        self.response_end_time = 0
    
    def initialize(self, seed: Optional[int] = None):
        """Initialize the integrated simulation"""
        self.world_cfg = WorldConfig(
            n_victims=self.n_victims,
            n_false_positives=self.n_false,
            rng_seed=seed
        )
        self.victims, self.false_sites = build_environment(self.world_cfg)
        
        r_cfg = RobotConfig()
        self.robots = init_robots(self.n_robots, self.world_cfg, r_cfg, rng_seed=seed)
        
        print(f"✅ Initialized: {self.n_robots} robots, {self.n_victims} victims, {self.n_false} false positives")
    
    def run_phase1_search(self) -> dict:
        """Run Phase 1: Robot search with consensus (Devika's code)"""
        import time
        
        print("\n🔍 PHASE 1: Starting robot search...")
        self.phase = "searching"
        self.search_start_time = time.time()
        
        p1_cfg = Phase1Config()
        p1_cfg.n_steps = 300
        
        self.phase1_result = run_phase1(
            world_cfg=self.world_cfg,
            robot_cfg=RobotConfig(),
            phase_cfg=p1_cfg,
            victims=self.victims,
            false_sites=self.false_sites,
            robots=self.robots,
            rng_seed=None
        )
        
        self.search_end_time = time.time()
        self.discovered_positions = self.phase1_result.consensus_positions
        
        self.robot_history = []
        max_history_len = max(len(r.pos_history_phase1) for r in self.robots) if self.robots else 0
        
        for i in range(0, max_history_len, 1):
            snapshot = []
            for robot in self.robots:
                if i < len(robot.pos_history_phase1):
                    pos = robot.pos_history_phase1[i]
                    snapshot.append({
                        'id': robot.id,
                        'x': float(pos[0]),
                        'y': float(pos[1]),
                        'theta': robot.theta,
                        'status': 'searching'
                    })
            if snapshot:
                self.robot_history.append({
                    'robots': snapshot,
                    'victims': [{'x': float(v.x), 'y': float(v.y), 'health': 100.0, 'priority': 'medium', 'rescue_progress': 0.0, 'num_robots': 0} for v in self.victims],
                    'phase': 'search'
                })
        
        print(f"✅ Phase 1 Complete!")
        print(f"   Consensus reached: {self.phase1_result.consensus_reached}")
        print(f"   Steps run: {self.phase1_result.steps_run}")
        print(f"   Victims discovered: {len(self.discovered_positions)}")
        print(f"   Robot history frames: {len(self.robot_history)}")
        
        self.phase = "search_complete"
        
        return {
            "consensus_reached": self.phase1_result.consensus_reached,
            "steps": self.phase1_result.steps_run,
            "discovered": len(self.discovered_positions),
            "positions": self.discovered_positions.tolist(),
            "robot_history": self.robot_history,
            "victims": [{'x': float(v.x), 'y': float(v.y)} for v in self.victims],
            "false_sites": [{'x': float(f.x), 'y': float(f.y)} for f in self.false_sites],
            "world_size": float(self.world_cfg.width)
        }
    
    def run_combined_phases(self) -> dict:
        """Run Phase 1 (search) + Phase 2 (navigation) + Phase 3 (priority rescue)"""
        import time
        from app.priority_rescue import (
            degrade_victim_health,
            assign_robots_by_priority,
            update_robot_targets,
            update_rescue_progress,
            calculate_victim_priority
        )
        
        print("\n🔍 PHASE 1: Search + PHASE 2: Navigation + PHASE 3: Priority Rescue")
        self.phase = "searching"
        self.search_start_time = time.time()
        
        # Initialize victim health for Phase 3
        for victim in self.victims:
            victim.health = np.random.uniform(30, 80)  # Injured victims
            victim.priority = calculate_victim_priority(victim)
            victim.rescue_progress = 0.0
            victim.num_robots_working = 0
        
        # === PHASE 1: Search (Devika's) ===
        print("\n🔍 PHASE 1: Consensus-based search...")
        p1_cfg = Phase1Config()
        p1_cfg.n_steps = 300
        
        self.phase1_result = run_phase1(
            world_cfg=self.world_cfg,
            robot_cfg=RobotConfig(),
            phase_cfg=p1_cfg,
            victims=self.victims,
            false_sites=self.false_sites,
            robots=self.robots,
            rng_seed=None
        )
        
        self.search_end_time = time.time()
        self.discovered_positions = self.phase1_result.consensus_positions
        
        # Collect Phase 1 history
        self.robot_history = []
        max_history_len_p1 = max(len(r.pos_history_phase1) for r in self.robots) if self.robots else 0
        
        for i in range(0, max_history_len_p1, 3):
            snapshot = []
            victim_states = []
            
            for robot in self.robots:
                if i < len(robot.pos_history_phase1):
                    pos = robot.pos_history_phase1[i]
                    snapshot.append({
                        'id': robot.id,
                        'x': float(pos[0]),
                        'y': float(pos[1]),
                        'theta': robot.theta,
                        'status': 'searching'
                    })
            
            for victim in self.victims:
                victim_states.append({
                    'x': float(victim.x),
                    'y': float(victim.y),
                    'health': float(victim.health),
                    'priority': victim.priority,
                    'rescue_progress': 0.0,
                    'num_robots': 0
                })
            
            if snapshot:
                self.robot_history.append({
                    'robots': snapshot,
                    'victims': victim_states,
                    'phase': 'search'
                })
        
        print(f"✅ Phase 1 Complete! Found {len(self.discovered_positions)} victims")
        
        # === PHASE 2: Navigation (Devika's) ===
        print("\n🚀 PHASE 2: Navigating to victims...")
        self.phase = "responding"
        
        p2_cfg = Phase2Config()
        p2_cfg.n_steps = 400  # Enough for robots to reach victims
        
        self.phase2_result = run_phase2(
            world_cfg=self.world_cfg,
            robot_cfg=RobotConfig(),
            phase_cfg=p2_cfg,
            victims=self.victims,
            false_sites=self.false_sites,
            robots=self.robots,
            initial_estimates=self.discovered_positions,
            rng_seed=None
        )
        
        # Collect Phase 2 history
        max_history_len_p2 = max(len(r.pos_history_phase2) for r in self.robots) if self.robots else 0
        
        for i in range(0, max_history_len_p2, 2):  # Every 2nd frame to reduce data
            snapshot = []
            victim_states = []
            
            for robot in self.robots:
                if i < len(robot.pos_history_phase2):
                    pos = robot.pos_history_phase2[i]
                    
                    # Check if near victim
                    is_working = False
                    for victim in self.victims:
                        dist = np.sqrt((pos[0] - victim.x)**2 + (pos[1] - victim.y)**2)
                        if dist < 1.5:
                            is_working = True
                            break
                    
                    snapshot.append({
                        'id': robot.id,
                        'x': float(pos[0]),
                        'y': float(pos[1]),
                        'theta': robot.theta,
                        'status': 'working' if is_working else 'navigating'
                    })
            
            # Count robots near each victim
            for victim in self.victims:
                robots_near = 0
                for robot in self.robots:
                    if i < len(robot.pos_history_phase2):
                        pos = robot.pos_history_phase2[i]
                        dist = np.sqrt((pos[0] - victim.x)**2 + (pos[1] - victim.y)**2)
                        if dist < 1.5:
                            robots_near += 1
                
                victim_states.append({
                    'x': float(victim.x),
                    'y': float(victim.y),
                    'health': float(victim.health),
                    'priority': victim.priority,
                    'rescue_progress': 0.0,
                    'num_robots': robots_near
                })
            
            if snapshot:
                self.robot_history.append({
                    'robots': snapshot,
                    'victims': victim_states,
                    'phase': 'navigation'
                })
        
        print(f"✅ Phase 2 Complete! Robots reached victims")
        
        # === PHASE 3: Priority-Based Rescue (YOUR CONTRIBUTION) ===
        print("\n💡 PHASE 3: Priority-based rescue with health dynamics...")
        self.response_start_time = time.time()
        
        # Initialize robot attributes
        for robot in self.robots:
            robot.target_victim_idx = -1
            robot.target_x = 0.0
            robot.target_y = 0.0
            robot.formation_x = robot.x
            robot.formation_y = robot.y
        
        # Run Phase 3 simulation
        n_steps_p3 = 600
        
        for step in range(n_steps_p3):
            # 1. Degrade victim health over time
            for victim in self.victims:
                degrade_victim_health(victim, dt=0.1)
            
            # 2. Reassign robots based on priorities (every 30 steps)
            if step % 30 == 0:
                assignments = assign_robots_by_priority(self.robots, self.victims)
                update_robot_targets(self.robots, self.victims, assignments)
            
            # 3. Move robots toward their assigned formation positions
            for robot in self.robots:
                if hasattr(robot, 'formation_x'):
                    dx = robot.formation_x - robot.x
                    dy = robot.formation_y - robot.y
                    dist = np.sqrt(dx**2 + dy**2)
                    
                    if dist > 0.1:
                        speed = 0.08  # Faster movement
                        robot.x += speed * dx / dist
                        robot.y += speed * dy / dist
                        robot.theta = np.arctan2(dy, dx)
            
            # 4. Update rescue progress
            update_rescue_progress(self.robots, self.victims)
            
            # 5. Save frame (every 3rd frame)
            if step % 2 == 0:
                snapshot = []
                victim_states = []
                
                for robot in self.robots:
                    status = 'navigating'
                    if hasattr(robot, 'target_victim_idx') and robot.target_victim_idx >= 0:
                        victim = self.victims[robot.target_victim_idx]
                        dist = np.sqrt((robot.x - victim.x)**2 + (robot.y - victim.y)**2)
                        if dist < 2.0:
                            status = 'working'
                    
                    snapshot.append({
                        'id': robot.id,
                        'x': float(robot.x),
                        'y': float(robot.y),
                        'theta': float(robot.theta),
                        'status': status
                    })
                
                for victim in self.victims:
                    victim_states.append({
                        'x': float(victim.x),
                        'y': float(victim.y),
                        'health': float(victim.health),
                        'priority': victim.priority,
                        'rescue_progress': float(victim.rescue_progress),
                        'num_robots': victim.num_robots_working
                    })
                
                self.robot_history.append({
                    'robots': snapshot,
                    'victims': victim_states,
                    'phase': 'rescue'
                })
        
        self.response_end_time = time.time()
        
        print(f"✅ Phase 3 Complete!")
        print(f"   Total animation frames: {len(self.robot_history)}")
        for i, victim in enumerate(self.victims):
            print(f"   Victim {i+1}: Health={victim.health:.1f}%, Progress={victim.rescue_progress:.1f}%, Priority={victim.priority}")
        
        self.phase = "complete"
        
        return {
            "consensus_reached": self.phase1_result.consensus_reached,
            "steps": self.phase1_result.steps_run,
            "discovered": len(self.discovered_positions),
            "positions": self.discovered_positions.tolist(),
            "robot_history": self.robot_history,
            "victims": [{'x': float(v.x), 'y': float(v.y)} for v in self.victims],
            "false_sites": [{'x': float(f.x), 'y': float(f.y)} for f in self.false_sites],
            "world_size": float(self.world_cfg.width),
            "phase2_consensus": self.phase2_result.consensus_reached
        }
    
    def run_phase2_response(self, behavior_mode: str = "cooperative") -> dict:
        """Run Phase 2: My agent response"""
        import time
        
        print("\n🚀 PHASE 2: Starting coordinated response...")
        self.phase = "responding"
        self.response_start_time = time.time()
        
        self.your_sim = create_simulation(
            num_agents=10,
            behavior_mode=behavior_mode,
            width=int(self.world_cfg.width),
            height=int(self.world_cfg.height)
        )
        
        for pos in self.discovered_positions:
            victim_pos = Position(int(pos[0]), int(pos[1]))
            self.your_sim.spawn_disaster(DisasterType.FIRE, position=victim_pos)
        
        self.your_sim.start()
        for _ in range(300):
            self.your_sim.update(0.1)
        self.your_sim.stop()
        
        self.response_end_time = time.time()
        self.response_metrics = self.your_sim.get_metrics()
        
        self.phase = "complete"
        
        return self.response_metrics
    
    def get_combined_results(self) -> IntegratedResult:
        """Get final combined results"""
        return IntegratedResult(
            phase1_complete=self.phase == "complete",
            phase2_complete=self.phase == "complete",
            search_time=self.search_end_time - self.search_start_time,
            response_time=self.response_end_time - self.response_start_time,
            victims_discovered=len(self.discovered_positions),
            victims_rescued=self.response_metrics.get('victims_rescued', 0) if self.response_metrics else 0,
            total_time=(self.response_end_time - self.search_start_time)
        )
    
    def get_state(self) -> dict:
        """Get current state for API"""
        state = {
            "phase": self.phase,
            "n_robots": self.n_robots,
            "n_victims": self.n_victims,
        }
        
        if self.phase1_result:
            state["phase1"] = {
                "consensus_reached": self.phase1_result.consensus_reached,
                "steps": self.phase1_result.steps_run,
                "discovered": len(self.discovered_positions)
            }
        
        if self.response_metrics:
            state["phase2"] = self.response_metrics
        
        return state