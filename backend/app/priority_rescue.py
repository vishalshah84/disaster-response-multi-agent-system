"""
Priority-Based Rescue System
Victims have degrading health, robots prioritize critical victims
"""
import numpy as np
from typing import List
from app.robot_search.environment import Site
from app.robot_search.robot import Robot


def calculate_victim_priority(victim: Site) -> str:
    """Determine priority based on health"""
    if victim.health <= 25:
        return "critical"
    elif victim.health <= 50:
        return "high"
    elif victim.health <= 75:
        return "medium"
    else:
        return "low"


def degrade_victim_health(victim: Site, dt: float = 0.1):
    """
    Victims lose health over time if not being rescued
    Degradation rate depends on number of robots helping
    """
    if victim.num_robots_working == 0:
        # No help - health degrades fast
        victim.health -= 0.5 * dt
    elif victim.num_robots_working < 3:
        # Some help - health degrades slowly
        victim.health -= 0.1 * dt
    else:
        # Sufficient help - health stabilizes or improves
        victim.health += 0.2 * dt * victim.num_robots_working
    
    # Clamp health between 0 and 100
    victim.health = max(0.0, min(100.0, victim.health))
    
    # Update priority
    victim.priority = calculate_victim_priority(victim)


def assign_robots_by_priority(robots: List[Robot], victims: List[Site]) -> dict:
    """
    Assign robots to victims based on priority
    Critical victims get more robots
    """
    assignments = {i: [] for i in range(len(victims))}
    
    # Calculate how many robots each victim should get
    total_robots = len(robots)
    priority_weights = {
        "critical": 5.0,
        "high": 3.0,
        "medium": 2.0,
        "low": 1.0
    }
    
    # Calculate weighted distribution
    victim_weights = [priority_weights.get(v.priority, 1.0) for v in victims]
    total_weight = sum(victim_weights)
    
    target_robots = []
    for weight in victim_weights:
        target = int((weight / total_weight) * total_robots)
        target = max(1, target)  # At least 1 robot per victim
        target_robots.append(target)
    
    # Adjust to match total robots
    while sum(target_robots) < total_robots:
        # Add to most critical victim
        critical_idx = max(range(len(victims)), key=lambda i: priority_weights.get(victims[i].priority, 0))
        target_robots[critical_idx] += 1
    
    while sum(target_robots) > total_robots:
        # Remove from least critical victim
        low_priority_idx = min(range(len(victims)), key=lambda i: priority_weights.get(victims[i].priority, 0))
        if target_robots[low_priority_idx] > 1:
            target_robots[low_priority_idx] -= 1
    
    # Assign robots to closest available victim slot
    available_robots = list(range(len(robots)))
    
    for victim_idx, num_needed in enumerate(target_robots):
        victim = victims[victim_idx]
        assigned = 0
        
        # Find closest robots to this victim
        distances = []
        for robot_idx in available_robots:
            robot = robots[robot_idx]
            dist = np.sqrt((robot.x - victim.x)**2 + (robot.y - victim.y)**2)
            distances.append((dist, robot_idx))
        
        distances.sort()
        
        # Assign closest robots
        for _, robot_idx in distances[:num_needed]:
            assignments[victim_idx].append(robot_idx)
            available_robots.remove(robot_idx)
            assigned += 1
            if assigned >= num_needed:
                break
    
    return assignments


def update_robot_targets(robots: List[Robot], victims: List[Site], assignments: dict):
    """Update each robot's target based on priority assignments"""
    for victim_idx, robot_indices in assignments.items():
        victim = victims[victim_idx]
        victim.num_robots_working = len(robot_indices)
        
        for robot_idx in robot_indices:
            robot = robots[robot_idx]
            robot.target_victim_idx = victim_idx
            robot.target_x = victim.x
            robot.target_y = victim.y
            
            # Calculate formation position around victim
            num_robots = len(robot_indices)
            slot_idx = robot_indices.index(robot_idx)
            angle = (2 * np.pi * slot_idx) / num_robots
            radius = 1.2  # Formation radius
            
            robot.formation_x = victim.x + radius * np.cos(angle)
            robot.formation_y = victim.y + radius * np.sin(angle)


def update_rescue_progress(robots: List[Robot], victims: List[Site]):
    """Update rescue progress for each victim"""
    for victim in victims:
        # Count robots near this victim
        robots_working = 0
        for robot in robots:
            dist = np.sqrt((robot.x - victim.x)**2 + (robot.y - victim.y)**2)
            if dist < 2.0:  # Within working range
                robots_working += 1
        
        victim.num_robots_working = robots_working
        
        # Update rescue progress
        if robots_working > 0:
            victim.rescue_progress += 0.5 * robots_working
            victim.rescue_progress = min(100.0, victim.rescue_progress)
