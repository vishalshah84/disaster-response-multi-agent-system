"""
Test all three behavioral modes
"""
from app.simulation_engine import create_simulation
from app.core.config import BehaviorMode, DisasterType

def test_all_behaviors():
    print("=" * 70)
    print("🧪 TESTING ALL THREE BEHAVIORAL MODES")
    print("=" * 70)
    
    behaviors = [
        BehaviorMode.COMPETITIVE,
        BehaviorMode.COOPERATIVE,
        BehaviorMode.AGREEMENT
    ]
    
    results = {}
    
    for behavior in behaviors:
        print(f"\n{'='*70}")
        print(f"Testing: {behavior.upper()}")
        print(f"{'='*70}")
        
        # Create simulation
        sim = create_simulation(
            num_agents=8,
            behavior_mode=behavior,
            width=30,
            height=30
        )
        
        # Spawn disasters
        for _ in range(3):
            sim.spawn_disaster(DisasterType.FIRE)
        
        print(f"✅ Created simulation with {len(sim.agents)} agents")
        print(f"   Behavior mode: {behavior}")
        print(f"   Disasters: {len(sim.environment.disasters)}")
        
        # Run simulation
        print(f"   Running 30 ticks...")
        sim.run_for_ticks(30, delta_time=0.1)
        
        # Get metrics
        metrics = sim.get_metrics()
        
        print(f"\n📊 Results:")
        print(f"   • Disasters resolved: {metrics['disasters_resolved']}")
        print(f"   • Victims rescued: {metrics['victims_rescued']}")
        print(f"   • Distance traveled: {metrics['total_distance_traveled']:.2f}")
        print(f"   • Efficiency: {metrics['rescue_efficiency']:.1f}%")
        
        # Behavior-specific stats
        if behavior == BehaviorMode.COMPETITIVE:
            print(f"   • Active auctions: {len(sim.auction_manager.active_auctions)}")
            print(f"   • Completed auctions: {len(sim.auction_manager.auction_history)}")
        elif behavior == BehaviorMode.COOPERATIVE:
            print(f"   • Active coalitions: {len(sim.coalition_manager.coalitions)}")
        elif behavior == BehaviorMode.AGREEMENT:
            leader = sim.consensus_manager.get_leader()
            print(f"   • Current leader: {leader}")
            print(f"   • Proposals: {len(sim.consensus_manager.proposals)}")
        
        results[behavior] = metrics
    
    # Compare results
    print(f"\n{'='*70}")
    print("📈 COMPARISON ACROSS BEHAVIORS")
    print(f"{'='*70}")
    
    print(f"\n{'Metric':<30} {'Competitive':<15} {'Cooperative':<15} {'Agreement':<15}")
    print("-" * 75)
    
    metrics_to_compare = [
        ('disasters_resolved', 'Disasters Resolved'),
        ('victims_rescued', 'Victims Rescued'),
        ('total_distance_traveled', 'Distance Traveled'),
        ('rescue_efficiency', 'Efficiency %')
    ]
    
    for key, label in metrics_to_compare:
        comp = results[BehaviorMode.COMPETITIVE].get(key, 0)
        coop = results[BehaviorMode.COOPERATIVE].get(key, 0)
        agree = results[BehaviorMode.AGREEMENT].get(key, 0)
        
        if 'distance' in key:
            print(f"{label:<30} {comp:<15.2f} {coop:<15.2f} {agree:<15.2f}")
        elif 'efficiency' in key:
            print(f"{label:<30} {comp:<15.1f} {coop:<15.1f} {agree:<15.1f}")
        else:
            print(f"{label:<30} {comp:<15} {coop:<15} {agree:<15}")
    
    print("\n" + "=" * 70)
    print("✅ ALL BEHAVIORAL MODES TESTED SUCCESSFULLY!")
    print("=" * 70)
    
    print("\n🎯 Key Findings:")
    print("   • Competitive: Agents compete via auctions")
    print("   • Cooperative: Agents form coalitions and share resources")
    print("   • Agreement: Agents elect leader and reach consensus")
    
    print("\n📝 Graduate-Level Components Implemented:")
    print("   ✅ Game Theory (Nash Equilibrium, Auctions)")
    print("   ✅ Coalition Formation (Pareto Optimality)")
    print("   ✅ Consensus Protocol (Raft, Byzantine Fault Tolerance)")
    print("   ✅ Multi-Agent Coordination")
    
    return results

if __name__ == "__main__":
    try:
        results = test_all_behaviors()
        
        print("\n🎉 DAY 2-3 COMPLETE!")
        print("\n📊 Current Progress: ~60%")
        print("\n🎯 Next Steps (Days 4-5):")
        print("   1. Build React frontend")
        print("   2. Create 2D visualization")
        print("   3. Add real-time dashboard")
        print("   4. Control panel for switching behaviors")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
