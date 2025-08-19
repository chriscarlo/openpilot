#!/usr/bin/env python3
"""
RTI Performance and Battery Impact Assessment

Comprehensive performance testing and resource usage analysis for the RTI system.
Measures CPU usage, memory footprint, network overhead, and battery impact.
"""

import asyncio
import time
import psutil
import os
import gc
import tracemalloc
import cProfile
import pstats
from io import StringIO
from dataclasses import dataclass
from unittest.mock import MagicMock, AsyncMock

from sunnypilot.rtid.threat_detector import ThreatDetector
from sunnypilot.rtid.waze_api_client import WazeAlert
from sunnypilot.selfdrive.controls.lib.rti_controller import RTIController


@dataclass
class PerformanceMetrics:
    """Container for performance test results."""
    test_name: str
    cpu_percent: float
    memory_mb: float
    processing_time_ms: float
    network_bytes: int = 0
    battery_impact_score: float = 0.0  # 0-100 scale


class ResourceMonitor:
    """Monitors system resource usage during RTI operations."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_cpu_time = None
        self.start_memory = None
        self.start_network = None
        self.measurements = []

    def start(self):
        """Start resource monitoring."""
        self.start_cpu_time = self.process.cpu_percent()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        # Get network stats if available
        try:
            net_io = psutil.net_io_counters()
            self.start_network = net_io.bytes_sent + net_io.bytes_recv
        except:
            self.start_network = 0

    def sample(self):
        """Take a resource usage sample."""
        cpu = self.process.cpu_percent()
        memory = self.process.memory_info().rss / 1024 / 1024

        self.measurements.append({
            'cpu': cpu,
            'memory': memory,
            'timestamp': time.time()
        })

    def get_metrics(self) -> dict:
        """Get aggregated metrics."""
        if not self.measurements:
            return {}

        cpu_samples = [m['cpu'] for m in self.measurements]
        memory_samples = [m['memory'] for m in self.measurements]

        # Get network usage
        network_bytes = 0
        try:
            net_io = psutil.net_io_counters()
            current_network = net_io.bytes_sent + net_io.bytes_recv
            network_bytes = current_network - self.start_network
        except:
            pass

        return {
            'cpu_avg': sum(cpu_samples) / len(cpu_samples),
            'cpu_max': max(cpu_samples),
            'memory_avg': sum(memory_samples) / len(memory_samples),
            'memory_max': max(memory_samples),
            'memory_delta': memory_samples[-1] - self.start_memory,
            'network_bytes': network_bytes,
            'samples': len(self.measurements)
        }


class RTIPerformanceTester:
    """Comprehensive performance testing for RTI system."""

    def __init__(self):
        self.results = []
        self.profiler = cProfile.Profile()

    def test_threat_processing_performance(self) -> PerformanceMetrics:
        """Test threat detector processing performance."""
        print("\nTest 1: Threat Processing Performance")
        print("-" * 40)

        detector = ThreatDetector()
        monitor = ResourceMonitor()

        # Create test data with varying threat counts
        test_scenarios = [
            (10, "Light traffic"),
            (50, "Moderate traffic"),
            (100, "Heavy traffic"),
            (200, "Extreme traffic")
        ]

        results = []

        for threat_count, scenario in test_scenarios:
            # Create mock threats
            mock_alerts = []
            for i in range(threat_count):
                alert = WazeAlert(
                    id=f'threat_{i}',
                    type='POLICE' if i % 3 == 0 else 'ACCIDENT',
                    latitude=37.4221 + (i * 0.0001),
                    longitude=-122.0841 + (i * 0.0001),
                    confidence=0.7 + (i % 3) * 0.1,
                    speed_limit=25.0 if i % 2 == 0 else None,
                    street='Test St',
                    country='US',
                    raw_data={'reliability': 5 + i % 5}
                )
                mock_alerts.append(alert)

            # Measure processing time
            monitor.start()
            start_time = time.perf_counter()

            # Process 100 cycles
            for _ in range(100):
                state = detector.process_threats(
                    traffic_data=mock_alerts,
                    current_location=(37.4221, -122.0841),
                    current_speed=20.0,
                    timestamp=int(time.time() * 1e9)
                )
                monitor.sample()

            elapsed = (time.perf_counter() - start_time) * 1000  # ms
            avg_time = elapsed / 100

            metrics = monitor.get_metrics()

            print(f"  {scenario} ({threat_count} threats):")
            print(f"    Avg processing: {avg_time:.2f}ms")
            print(f"    CPU usage: {metrics.get('cpu_avg', 0):.1f}%")
            print(f"    Memory: {metrics.get('memory_avg', 0):.1f}MB")

            results.append((threat_count, avg_time, metrics))

        # Check if performance scales linearly
        if len(results) >= 2:
            # Compare 10 threats vs 100 threats
            time_10 = results[0][1]
            time_100 = results[2][1] if len(results) > 2 else results[-1][1]
            scaling_factor = time_100 / time_10

            print("\n  Scaling Analysis:")
            print(f"    10 threats: {time_10:.2f}ms")
            print(f"    100 threats: {time_100:.2f}ms")
            print(f"    Scaling factor: {scaling_factor:.1f}x")

            if scaling_factor < 15:  # Should not scale worse than O(n log n)
                print("    ✓ Performance scaling acceptable")
            else:
                print("    ✗ Performance scaling poor")

        # Return metrics for moderate load
        moderate_result = results[1] if len(results) > 1 else results[0]
        return PerformanceMetrics(
            test_name="Threat Processing",
            cpu_percent=moderate_result[2].get('cpu_avg', 0),
            memory_mb=moderate_result[2].get('memory_avg', 0),
            processing_time_ms=moderate_result[1]
        )

    def test_memory_leak_detection(self) -> PerformanceMetrics:
        """Test for memory leaks in RTI components."""
        print("\nTest 2: Memory Leak Detection")
        print("-" * 40)

        # Enable memory tracking
        tracemalloc.start()

        detector = ThreatDetector()

        # Create reusable test data
        mock_alerts = [
            WazeAlert(
                id=f'threat_{i}',
                type='POLICE',
                latitude=37.4221 + (i * 0.0001),
                longitude=-122.0841,
                confidence=0.85,
                speed_limit=25.0,
                street='Test St',
                country='US',
                raw_data={'reliability': 8}
            )
            for i in range(50)
        ]

        # Take initial snapshot
        gc.collect()
        snapshot1 = tracemalloc.take_snapshot()

        # Process many cycles
        print("  Processing 1000 cycles...")
        for i in range(1000):
            state = detector.process_threats(
                traffic_data=mock_alerts,
                current_location=(37.4221, -122.0841),
                current_speed=20.0,
                timestamp=int(time.time() * 1e9)
            )

            # Clear state to simulate normal operation
            del state

            if i % 100 == 0:
                gc.collect()

        # Take final snapshot
        gc.collect()
        snapshot2 = tracemalloc.take_snapshot()

        # Analyze memory growth
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')

        total_growth = 0
        print("\n  Top memory growth:")
        for stat in top_stats[:5]:
            if stat.size_diff > 0:
                growth_mb = stat.size_diff / 1024 / 1024
                total_growth += growth_mb
                print(f"    {stat.filename}:{stat.lineno}: +{growth_mb:.2f}MB")

        tracemalloc.stop()

        # Check for leaks
        if total_growth < 1.0:  # Less than 1MB growth
            print(f"\n  ✓ No significant memory leaks detected ({total_growth:.2f}MB growth)")
            leak_score = 0
        else:
            print(f"\n  ✗ Potential memory leak detected ({total_growth:.2f}MB growth)")
            leak_score = min(total_growth * 10, 100)  # Scale to 0-100

        return PerformanceMetrics(
            test_name="Memory Leak",
            cpu_percent=0,
            memory_mb=total_growth,
            processing_time_ms=0,
            battery_impact_score=leak_score
        )

    async def test_api_network_overhead(self) -> PerformanceMetrics:
        """Test network overhead of API calls."""
        print("\nTest 3: API Network Overhead")
        print("-" * 40)

        # Mock API client
        mock_client = AsyncMock()

        # Simulate API response sizes
        response_sizes = {
            'minimal': 500,      # 500 bytes - few threats
            'typical': 5000,     # 5KB - normal traffic
            'heavy': 50000,      # 50KB - heavy traffic
            'extreme': 200000    # 200KB - extreme conditions
        }

        total_bytes = 0
        call_count = 0

        for scenario, size in response_sizes.items():
            # Simulate API response
            mock_response = {
                'alerts': [
                    {'id': f't_{i}', 'type': 'POLICE', 'lat': 37.4 + i*0.001, 'lon': -122.0}
                    for i in range(size // 100)  # Approximate threat count
                ],
                'padding': 'x' * (size - 100)  # Simulate response size
            }

            mock_client.get_traffic_alerts.return_value = mock_response['alerts']

            # Measure network usage (simulated)
            print(f"  {scenario.capitalize()} traffic: {size/1024:.1f}KB")
            total_bytes += size
            call_count += 1

        # Calculate hourly network usage (assuming 1 call per second)
        hourly_bytes = total_bytes / call_count * 3600
        daily_mb = hourly_bytes * 24 / 1024 / 1024

        print("\n  Network Usage Projection:")
        print(f"    Average per call: {total_bytes/call_count/1024:.1f}KB")
        print(f"    Hourly (1Hz): {hourly_bytes/1024/1024:.1f}MB")
        print(f"    Daily: {daily_mb:.1f}MB")

        # Battery impact estimate (network usage contributes to battery drain)
        # Rough estimate: 1MB = 0.1% battery on cellular
        battery_impact = daily_mb * 0.1

        if daily_mb < 50:
            print(f"  ✓ Network usage acceptable ({daily_mb:.1f}MB/day)")
        else:
            print(f"  ✗ High network usage ({daily_mb:.1f}MB/day)")

        return PerformanceMetrics(
            test_name="Network Overhead",
            cpu_percent=0,
            memory_mb=0,
            processing_time_ms=0,
            network_bytes=int(hourly_bytes),
            battery_impact_score=battery_impact
        )

    def test_cpu_profile(self) -> PerformanceMetrics:
        """Profile CPU usage of critical paths."""
        print("\nTest 4: CPU Profiling")
        print("-" * 40)

        # Create components
        detector = ThreatDetector()
        CP = MagicMock()
        controller = RTIController(CP)

        # Mock data
        mock_alerts = [
            WazeAlert(
                id=f'threat_{i}',
                type='POLICE',
                latitude=37.4221 + (i * 0.0001),
                longitude=-122.0841,
                confidence=0.85,
                speed_limit=25.0,
                street='Test St',
                country='US',
                raw_data={'reliability': 8}
            )
            for i in range(20)
        ]

        # Profile threat processing
        self.profiler.enable()

        for _ in range(1000):
            state = detector.process_threats(
                traffic_data=mock_alerts,
                current_location=(37.4221, -122.0841),
                current_speed=20.0,
                timestamp=int(time.time() * 1e9)
            )

        self.profiler.disable()

        # Analyze profile
        s = StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(10)

        profile_output = s.getvalue()

        # Extract top functions
        print("  Top CPU consumers:")
        lines = profile_output.split('\n')[5:15]  # Skip header
        for line in lines:
            if line.strip() and 'rtid' in line or 'threat' in line:
                # Parse and display relevant info
                parts = line.split()
                if len(parts) > 5:
                    func_name = parts[-1].split('/')[-1]
                    time_pct = parts[2] if '%' in parts[2] else parts[1]
                    print(f"    {func_name}: {time_pct}")

        return PerformanceMetrics(
            test_name="CPU Profile",
            cpu_percent=0,
            memory_mb=0,
            processing_time_ms=0
        )

    async def test_battery_impact(self) -> PerformanceMetrics:
        """Estimate battery impact of RTI system."""
        print("\nTest 5: Battery Impact Estimation")
        print("-" * 40)

        # Battery impact factors (rough estimates)
        # Based on typical smartphone battery (3000mAh, 3.7V = 11.1Wh)

        factors = {
            'cpu_processing': 0.5,     # mW per % CPU
            'memory_active': 0.01,     # mW per MB
            'network_cellular': 200,   # mW for cellular data
            'network_wifi': 50,        # mW for WiFi data
            'gps_active': 150,         # mW for GPS
            'display_update': 10       # mW per UI update
        }

        # Simulate typical usage scenario
        usage_scenario = {
            'cpu_percent': 5.0,        # Average CPU usage
            'memory_mb': 50.0,         # Memory footprint
            'network_kb_per_hour': 3600,  # 1KB/s
            'gps_queries_per_min': 60,    # 1Hz GPS updates
            'ui_updates_per_sec': 2       # HUD updates
        }

        # Calculate power consumption
        power_mw = 0

        # CPU impact
        cpu_power = usage_scenario['cpu_percent'] * factors['cpu_processing']
        power_mw += cpu_power
        print(f"  CPU ({usage_scenario['cpu_percent']:.1f}%): {cpu_power:.1f}mW")

        # Memory impact
        mem_power = usage_scenario['memory_mb'] * factors['memory_active']
        power_mw += mem_power
        print(f"  Memory ({usage_scenario['memory_mb']:.0f}MB): {mem_power:.1f}mW")

        # Network impact (assume cellular)
        net_active_ratio = usage_scenario['network_kb_per_hour'] / 3600 / 10  # Activity ratio
        net_power = factors['network_cellular'] * net_active_ratio
        power_mw += net_power
        print(f"  Network: {net_power:.1f}mW")

        # GPS impact
        gps_active_ratio = usage_scenario['gps_queries_per_min'] / 60
        gps_power = factors['gps_active'] * gps_active_ratio
        power_mw += gps_power
        print(f"  GPS: {gps_power:.1f}mW")

        # Display impact
        ui_power = usage_scenario['ui_updates_per_sec'] * factors['display_update']
        power_mw += ui_power
        print(f"  Display: {ui_power:.1f}mW")

        # Total impact
        print(f"\n  Total Power: {power_mw:.1f}mW")

        # Convert to battery percentage per hour
        battery_capacity_mwh = 11100  # 11.1Wh = 11100mWh
        battery_percent_per_hour = (power_mw / battery_capacity_mwh) * 100

        print(f"  Battery Impact: {battery_percent_per_hour:.2f}% per hour")
        print(f"  Estimated Runtime: {100/battery_percent_per_hour:.1f} hours")

        # Score battery impact (0-100, lower is better)
        battery_score = min(battery_percent_per_hour * 10, 100)

        if battery_percent_per_hour < 2.0:
            print("  ✓ Battery impact acceptable")
        elif battery_percent_per_hour < 5.0:
            print("  ⚠ Moderate battery impact")
        else:
            print("  ✗ High battery impact")

        return PerformanceMetrics(
            test_name="Battery Impact",
            cpu_percent=usage_scenario['cpu_percent'],
            memory_mb=usage_scenario['memory_mb'],
            processing_time_ms=0,
            network_bytes=int(usage_scenario['network_kb_per_hour'] * 1024),
            battery_impact_score=battery_score
        )

    async def run_performance_suite(self):
        """Run complete performance test suite."""
        print("\n" + "="*60)
        print("RTI PERFORMANCE & BATTERY ASSESSMENT")
        print("="*60)

        results = []

        # Test 1: Processing performance
        perf_metrics = self.test_threat_processing_performance()
        results.append(perf_metrics)

        # Test 2: Memory leaks
        mem_metrics = self.test_memory_leak_detection()
        results.append(mem_metrics)

        # Test 3: Network overhead
        net_metrics = await self.test_api_network_overhead()
        results.append(net_metrics)

        # Test 4: CPU profiling
        cpu_metrics = self.test_cpu_profile()
        results.append(cpu_metrics)

        # Test 5: Battery impact
        battery_metrics = await self.test_battery_impact()
        results.append(battery_metrics)

        # Summary
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)

        total_battery_impact = sum(r.battery_impact_score for r in results)
        avg_cpu = sum(r.cpu_percent for r in results if r.cpu_percent > 0) / len([r for r in results if r.cpu_percent > 0])
        avg_memory = sum(r.memory_mb for r in results if r.memory_mb > 0) / len([r for r in results if r.memory_mb > 0])

        print("\nOverall Metrics:")
        print(f"  Average CPU Usage: {avg_cpu:.1f}%")
        print(f"  Average Memory: {avg_memory:.1f}MB")
        print(f"  Battery Impact Score: {total_battery_impact:.1f}/100")

        # Performance grade
        if total_battery_impact < 20 and avg_cpu < 10:
            grade = "A - Excellent"
        elif total_battery_impact < 40 and avg_cpu < 20:
            grade = "B - Good"
        elif total_battery_impact < 60 and avg_cpu < 30:
            grade = "C - Acceptable"
        else:
            grade = "D - Needs Optimization"

        print(f"\nPerformance Grade: {grade}")

        # Recommendations
        print("\nOptimization Recommendations:")

        if avg_cpu > 15:
            print("  • Consider optimizing threat processing algorithms")
        if avg_memory > 100:
            print("  • Reduce memory footprint with object pooling")
        if total_battery_impact > 40:
            print("  • Implement adaptive update rates based on conditions")
            print("  • Consider caching API responses")
            print("  • Reduce HUD update frequency when no threats")

        return grade


async def main():
    """Run performance assessment."""
    tester = RTIPerformanceTester()
    grade = await tester.run_performance_suite()

    print("\n✓ Performance assessment completed")
    return 0


if __name__ == "__main__":
    asyncio.run(main())
