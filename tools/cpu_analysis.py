#!/usr/bin/env python3
"""
Advanced CPU and Database Performance Analysis Tool

This tool provides comprehensive analysis of CPU utilization bottlenecks,
focusing on identifying whether issues are MySQL-related, system-related,
or application-related.
"""

import time
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pymysql
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import subprocess
import sys
import os

def get_mysql_connection(host='127.0.0.1', port=3307, user='root', password='', database='investment_data_new'):
    """Get MySQL connection with error handling"""
    try:
        return pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            connect_timeout=5,
            read_timeout=30,
            write_timeout=30,
            autocommit=True
        )
    except Exception as e:
        print(f"Failed to connect to MySQL: {e}")
        return None

def analyze_mysql_configuration(conn):
    """Analyze MySQL configuration for performance bottlenecks"""
    print("=== MySQL Configuration Analysis ===")

    config_queries = [
        ("InnoDB Buffer Pool Size", "SHOW VARIABLES LIKE 'innodb_buffer_pool_size'"),
        ("InnoDB Buffer Pool Instances", "SHOW VARIABLES LIKE 'innodb_buffer_pool_instances'"),
        ("InnoDB Flush Method", "SHOW VARIABLES LIKE 'innodb_flush_method'"),
        ("InnoDB Log File Size", "SHOW VARIABLES LIKE 'innodb_log_file_size'"),
        ("Max Connections", "SHOW VARIABLES LIKE 'max_connections'"),
        ("Query Cache Size", "SHOW VARIABLES LIKE 'query_cache_size'"),
        ("Table Open Cache", "SHOW VARIABLES LIKE 'table_open_cache'"),
        ("Thread Cache Size", "SHOW VARIABLES LIKE 'thread_cache_size'"),
        ("InnoDB Read IO Threads", "SHOW VARIABLES LIKE 'innodb_read_io_threads'"),
        ("InnoDB Write IO Threads", "SHOW VARIABLES LIKE 'innodb_write_io_threads'"),
    ]

    with conn.cursor() as cursor:
        for desc, query in config_queries:
            try:
                cursor.execute(query)
                result = cursor.fetchone()
                if result:
                    value = result[1]
                    # Convert bytes to human readable
                    if 'size' in desc.lower() and isinstance(value, (int, str)):
                        try:
                            if isinstance(value, str):
                                value = int(value)
                            if value > 1024**3:
                                print(".1f")
                            elif value > 1024**2:
                                print(".1f")
                            elif value > 1024:
                                print(".1f")
                            else:
                                print(f"{desc}: {value}")
                        except:
                            print(f"{desc}: {value}")
                    else:
                        print(f"{desc}: {value}")
            except Exception as e:
                print(f"{desc}: Error - {e}")

def analyze_mysql_performance():
    """Analyze MySQL performance bottlenecks comprehensively"""
    print("=== MySQL Performance Analysis ===")

    conn = get_mysql_connection()
    if not conn:
        return

    try:
        with conn.cursor() as cursor:
            # Check connection count and status
            cursor.execute("SHOW PROCESSLIST")
            processes = cursor.fetchall()
            active_connections = len([p for p in processes if p[6] not in ['Sleep', '']])
            sleeping_connections = len([p for p in processes if p[6] == 'Sleep'])

            print(f"Total MySQL connections: {len(processes)}")
            print(f"Active connections: {active_connections}")
            print(f"Sleeping connections: {sleeping_connections}")

            # Analyze connection states
            state_counts = {}
            for proc in processes:
                state = proc[6] if len(proc) > 6 else 'Unknown'
                state_counts[state] = state_counts.get(state, 0) + 1

            print("\nConnection states:")
            for state, count in sorted(state_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {state}: {count}")

            # Check for long-running queries
            cursor.execute("""
                SELECT ID, USER, HOST, DB, COMMAND, TIME, STATE, INFO
                FROM information_schema.processlist
                WHERE COMMAND != 'Sleep' AND TIME > 10
                ORDER BY TIME DESC
                LIMIT 10
            """)

            long_queries = cursor.fetchall()
            if long_queries:
                print(f"\n⚠️  Long-running queries (>10s): {len(long_queries)}")
                for query in long_queries[:5]:
                    print(f"  ID {query[0]}: {query[5]}s - {query[6] or 'No state'}")
                    if query[7]:  # INFO column
                        print(f"    Query: {query[7][:100]}...")
            else:
                print("\n✅ No long-running queries found")

        # Analyze MySQL configuration
        analyze_mysql_configuration(conn)

        # Analyze query performance schema (if available)
        analyze_query_performance_schema(conn)

        # Analyze InnoDB status
        analyze_innodb_status(conn)

        # Analyze table locks
        analyze_table_locks(conn)

    finally:
        conn.close()

def analyze_query_performance_schema(conn):
    """Analyze MySQL performance schema for query bottlenecks"""
    print("\n=== Query Performance Schema Analysis ===")

    try:
        with conn.cursor() as cursor:
            # Check if performance schema is enabled
            cursor.execute("SHOW VARIABLES LIKE 'performance_schema'")
            result = cursor.fetchone()
            if not result or result[1] != 'ON':
                print("❌ Performance Schema is disabled - enable it for detailed query analysis")
                return

            # Analyze slowest queries
            cursor.execute("""
                SELECT
                    schema_name,
                    digest_text,
                    count_star,
                    avg_timer_wait/1000000000 as avg_time_sec,
                    max_timer_wait/1000000000 as max_time_sec,
                    sum_rows_examined,
                    sum_rows_sent
                FROM performance_schema.events_statements_summary_by_digest
                WHERE schema_name IS NOT NULL
                AND avg_timer_wait > 100000000  -- > 0.1 second
                ORDER BY avg_timer_wait DESC
                LIMIT 10
            """)

            slow_queries = cursor.fetchall()
            if slow_queries:
                print(f"🐌 Top {len(slow_queries)} slowest queries:")
                for i, query in enumerate(slow_queries, 1):
                    print(f"{i}. Schema: {query[0]}")
                    print(".3f")
                    print(f"   Max: {query[4]:.3f}s, Count: {query[2]}")
                    print(f"   Rows examined: {query[5]}, Rows sent: {query[6]}")
                    if query[1]:  # digest_text
                        print(f"   Query: {query[1][:80]}..."[:100])
                    print()
            else:
                print("✅ No slow queries found in performance schema")

    except Exception as e:
        print(f"Query performance schema analysis failed: {e}")

def analyze_innodb_status(conn):
    """Analyze InnoDB status for detailed performance insights"""
    print("\n=== InnoDB Status Analysis ===")

    try:
        with conn.cursor() as cursor:
            cursor.execute("SHOW ENGINE INNODB STATUS")
            result = cursor.fetchone()

            if result and len(result) > 2:
                status_text = result[2]

                # Analyze semaphore waits
                if 'OS WAIT' in status_text:
                    print("⚠️  InnoDB OS waits detected - possible I/O bottleneck")
                    # Count OS waits
                    os_waits = status_text.count('OS WAIT')
                    print(f"   OS waits: {os_waits}")

                # Analyze buffer pool
                if 'Buffer pool hit rate' in status_text:
                    lines = status_text.split('\n')
                    for line in lines:
                        if 'Buffer pool hit rate' in line:
                            print(f"   {line.strip()}")

                # Analyze log performance
                if 'Log sequence number' in status_text:
                    lines = status_text.split('\n')
                    for line in lines:
                        if 'Log flushed up to' in line:
                            print(f"   {line.strip()}")

                # Analyze deadlock information
                if 'TRANSACTIONS' in status_text:
                    deadlock_section = status_text.split('TRANSACTIONS')[1].split('FILE I/O')[0]
                    if 'ROLL BACK' in deadlock_section:
                        print("⚠️  Deadlocks detected in recent history")

            # Analyze InnoDB metrics
            innodb_metrics = [
                "SHOW STATUS LIKE 'Innodb_buffer_pool_pages_total'",
                "SHOW STATUS LIKE 'Innodb_buffer_pool_pages_free'",
                "SHOW STATUS LIKE 'Innodb_buffer_pool_pages_dirty'",
                "SHOW STATUS LIKE 'Innodb_rows_read'",
                "SHOW STATUS LIKE 'Innodb_rows_inserted'",
                "SHOW STATUS LIKE 'Innodb_rows_updated'",
                "SHOW STATUS LIKE 'Innodb_rows_deleted'",
            ]

            print("\nInnoDB Metrics:")
            for query in innodb_metrics:
                try:
                    cursor.execute(query)
                    result = cursor.fetchone()
                    if result:
                        print(f"  {result[0]}: {result[1]}")
                except Exception as e:
                    print(f"  Error getting {query}: {e}")

    except Exception as e:
        print(f"InnoDB status analysis failed: {e}")

def analyze_table_locks(conn):
    """Analyze table locks and waiting queries"""
    print("\n=== Table Locks Analysis ===")

    try:
        with conn.cursor() as cursor:
            # Check current locks
            cursor.execute("""
                SELECT
                    r.trx_id waiting_trx_id,
                    r.trx_mysql_thread_id waiting_thread,
                    r.trx_query waiting_query,
                    b.trx_id blocking_trx_id,
                    b.trx_mysql_thread_id blocking_thread,
                    b.trx_query blocking_query
                FROM information_schema.innodb_lock_waits w
                JOIN information_schema.innodb_trx b ON b.trx_id = w.blocking_trx_id
                JOIN information_schema.innodb_trx r ON r.trx_id = w.requesting_trx_id
                LIMIT 10
            """)

            locks = cursor.fetchall()
            if locks:
                print(f"🔒 Active lock waits: {len(locks)}")
                for lock in locks[:5]:
                    print(f"  Waiting: Thread {lock[1]} - {lock[2][:50] if lock[2] else 'No query'}")
                    print(f"  Blocking: Thread {lock[4]} - {lock[5][:50] if lock[5] else 'No query'}")
                    print()
            else:
                print("✅ No active lock waits")

    except Exception as e:
        print(f"Table locks analysis failed: {e}")

def analyze_thread_activity():
    """Analyze thread activity and bottlenecks"""
    print("\n=== Thread Activity Analysis ===")

    # Get main process
    current_pid = psutil.Process().pid
    main_process = psutil.Process(current_pid)

    print(f"Main process PID: {current_pid}")
    print(f"Main process threads: {main_process.num_threads()}")
    print(f"Main process CPU%: {main_process.cpu_percent()}%")

    # Analyze thread states
    thread_states = {}
    for thread in threading.enumerate():
        state = getattr(thread, '_tstate_lock', None)
        if state:
            thread_states['locked'] = thread_states.get('locked', 0) + 1
        else:
            thread_states['active'] = thread_states.get('active', 0) + 1

    # Analyze all threads in the system
    all_threads = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
        try:
            if proc.info['cpu_percent'] is not None:
                all_threads.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Sort by CPU usage
    all_threads.sort(key=lambda x: x['cpu_percent'] or 0, reverse=True)

    print(f"\nTop 10 CPU-consuming processes:")
    for i, proc in enumerate(all_threads[:10], 1):
        cpu = proc['cpu_percent'] or 0
        mem = proc['memory_percent'] or 0
        print(".1f")

    # Analyze Python threads specifically
    python_threads = [t for t in threading.enumerate()]
    print(f"\nPython threads: {len(python_threads)}")
    for i, thread in enumerate(python_threads[:10], 1):
        print(f"  {i}. {thread.name} (ID: {thread.ident})")

    # Check for thread contention
    if len(python_threads) > psutil.cpu_count() * 2:
        print("⚠️  High thread count detected - possible thread contention")
    else:
        print("✅ Thread count looks reasonable")

def simulate_workload():
    """Simulate workload to test CPU utilization"""
    print("\n=== CPU Workload Simulation ===")

    def cpu_intensive_task(task_id):
        """CPU intensive task"""
        start_time = time.time()
        result = 0
        # CPU intensive calculation
        for i in range(1000000):
            result += i ** 2
        end_time = time.time()
        return f"Task {task_id}: {end_time - start_time:.3f}s"

    def io_intensive_task(task_id):
        """I/O intensive task"""
        start_time = time.time()
        time.sleep(0.1)  # Simulate I/O wait
        end_time = time.time()
        return f"Task {task_id}: {end_time - start_time:.3f}s"

    # Test CPU intensive
    print("Testing CPU intensive workload...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(cpu_intensive_task, i) for i in range(4)]
        for future in futures:
            print(f"  {future.result()}")

    # Test I/O intensive
    print("\nTesting I/O intensive workload...")
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(io_intensive_task, i) for i in range(16)]
        for future in futures:
            print(f"  {future.result()}")

def check_system_limits():
    """Check system limits that might affect performance"""
    print("\n=== System Limits Check ===")

    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"File descriptors limit: {soft}/{hard}")
    except:
        print("Could not check file descriptors limit")

    # Check ulimits
    try:
        import subprocess
        result = subprocess.run(['ulimit', '-n'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"ulimit -n: {result.stdout.strip()}")
    except:
        print("Could not check ulimit")

def analyze_bottlenecks():
    """Analyze potential performance bottlenecks"""
    print("\n=== Performance Bottleneck Analysis ===")

    # CPU analysis
    cpu_count = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"CPU cores: {cpu_count}")
    print(f"CPU usage: {cpu_percent}%")

    # Memory analysis
    mem = psutil.virtual_memory()
    print(f"Memory usage: {mem.percent}% ({mem.used/1024/1024/1024:.1f}GB/{mem.total/1024/1024/1024:.1f}GB)")

    # Disk I/O
    disk_io = psutil.disk_io_counters()
    if disk_io:
        print(f"Disk read: {disk_io.read_bytes/1024/1024:.1f}MB")
        print(f"Disk write: {disk_io.write_bytes/1024/1024:.1f}MB")

    # Network I/O
    net_io = psutil.net_io_counters()
    if net_io:
        print(f"Network sent: {net_io.bytes_sent/1024/1024:.1f}MB")
        print(f"Network recv: {net_io.bytes_recv/1024/1024:.1f}MB")

def monitor_system_resources(duration: int = 30, interval: int = 1):
    """Monitor system resources over time to identify bottlenecks"""
    print(f"\n=== System Resource Monitoring ({duration}s) ===")

    cpu_history = []
    mem_history = []
    disk_history = []
    net_history = []

    print("Monitoring system resources... (Ctrl+C to stop early)")
    print("Time | CPU% | Mem% | Disk Read | Disk Write | Net Sent | Net Recv")
    print("-" * 70)

    start_time = time.time()
    try:
        for i in range(duration):
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=None)
            mem_percent = psutil.virtual_memory().percent

            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_read = disk_io.read_bytes if disk_io else 0
            disk_write = disk_io.write_bytes if disk_io else 0

            # Network I/O
            net_io = psutil.net_io_counters()
            net_sent = net_io.bytes_sent if net_io else 0
            net_recv = net_io.bytes_recv if net_io else 0

            # Store history
            cpu_history.append(cpu_percent)
            mem_history.append(mem_percent)
            disk_history.append((disk_read, disk_write))
            net_history.append((net_sent, net_recv))

            # Print current values
            if i % 5 == 0:  # Print every 5 seconds
                print("5d")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")

    # Analyze collected data
    if cpu_history:
        avg_cpu = sum(cpu_history) / len(cpu_history)
        max_cpu = max(cpu_history)
        min_cpu = min(cpu_history)

        avg_mem = sum(mem_history) / len(mem_history)
        max_mem = max(mem_history)

        print("
=== Monitoring Summary ==="        print(".1f")
        print(".1f")
        print(".1f")

        # CPU utilization analysis
        if avg_cpu < 30:
            print("⚠️  LOW CPU UTILIZATION - Possible bottleneck elsewhere (I/O, locks, etc.)")
        elif avg_cpu > 80:
            print("⚠️  HIGH CPU UTILIZATION - CPU bottleneck detected")
        else:
            print("✅ CPU utilization looks balanced")

        # Memory analysis
        if max_mem > 90:
            print("⚠️  HIGH MEMORY USAGE - Possible memory bottleneck")
        elif max_mem < 50:
            print("✅ Memory usage looks good")

        # I/O analysis
        if len(disk_history) > 1:
            disk_read_rate = (disk_history[-1][0] - disk_history[0][0]) / duration / 1024 / 1024
            disk_write_rate = (disk_history[-1][1] - disk_history[0][1]) / duration / 1024 / 1024
            print(".1f")

            if disk_read_rate > 50 or disk_write_rate > 50:
                print("⚠️  HIGH DISK I/O - Possible I/O bottleneck")
            else:
                print("✅ Disk I/O looks reasonable")

def analyze_query_execution_plan(conn, query: str):
    """Analyze query execution plan for performance bottlenecks"""
    print("
=== Query Execution Plan Analysis ==="    print(f"Analyzing query: {query[:100]}...")

    try:
        with conn.cursor() as cursor:
            # Get execution plan
            explain_query = f"EXPLAIN FORMAT=JSON {query}"
            cursor.execute(explain_query)
            plan = cursor.fetchone()

            if plan and len(plan) > 0:
                import json
                plan_json = json.loads(plan[0])

                print("\nQuery execution plan:")
                print(f"Query cost: {plan_json.get('query_block', {}).get('cost_info', {}).get('query_cost', 'Unknown')}")

                # Analyze table access
                if 'query_block' in plan_json:
                    query_block = plan_json['query_block']
                    if 'table' in query_block:
                        table_info = query_block['table']
                        print(f"Table: {table_info.get('table_name', 'Unknown')}")
                        print(f"Access type: {table_info.get('access_type', 'Unknown')}")
                        print(f"Key used: {table_info.get('key', 'None')}")
                        print(f"Rows examined: {table_info.get('rows_examined_per_scan', 'Unknown')}")

                        # Check for full table scans
                        if table_info.get('access_type') == 'ALL':
                            print("⚠️  FULL TABLE SCAN detected - consider adding index")
                        elif 'Using index' in str(table_info):
                            print("✅ Using index - good!")
                        else:
                            print("ℹ️  Table access method: mixed")

    except Exception as e:
        print(f"Query execution plan analysis failed: {e}")

def analyze_index_usage(conn):
    """Analyze index usage and suggest improvements"""
    print("\n=== Index Usage Analysis ===")

    try:
        with conn.cursor() as cursor:
            # Get index usage statistics
            cursor.execute("""
                SELECT
                    object_schema,
                    object_name,
                    index_name,
                    count_read,
                    count_write,
                    count_fetch,
                    pages,
                    size
                FROM performance_schema.table_io_waits_summary_by_index_usage
                WHERE object_schema = 'investment_data_new'
                AND index_name IS NOT NULL
                ORDER BY count_read DESC
                LIMIT 20
            """)

            indexes = cursor.fetchall()
            if indexes:
                print("Top 20 indexes by read usage:")
                print("Schema | Table | Index | Reads | Writes | Fetches | Size (MB)")
                print("-" * 80)

                for idx in indexes:
                    size_mb = (idx[7] * 16384) / 1024 / 1024 if idx[7] else 0  # Approximate page size
                    print("10")

                # Check for unused indexes
                unused_indexes = [idx for idx in indexes if idx[3] == 0 and idx[4] == 0]  # No reads or writes
                if unused_indexes:
                    print(f"\n⚠️  {len(unused_indexes)} potentially unused indexes found")
                    for idx in unused_indexes[:5]:
                        print(f"  {idx[0]}.{idx[1]}.{idx[2]}")

            else:
                print("No index usage data available")

    except Exception as e:
        print(f"Index usage analysis failed: {e}")

def generate_recommendations():
    """Generate comprehensive recommendations based on analysis"""
    print("🎯 PERFORMANCE OPTIMIZATION RECOMMENDATIONS")
    print("=" * 60)

    recommendations = []

    # CPU-related recommendations
    recommendations.append(("CPU Optimization:", [
        "• If CPU < 30%: Check for I/O bottlenecks, locks, or inefficient queries",
        "• If CPU > 80%: Consider increasing thread count or using multiprocessing",
        "• Monitor thread contention - avoid excessive thread creation",
        "• Profile Python code for CPU-intensive functions"
    ]))

    # MySQL-related recommendations
    recommendations.append(("MySQL Optimization:", [
        "• Increase innodb_buffer_pool_size if memory allows",
        "• Enable query cache if read-heavy workload",
        "• Check for missing indexes on frequently queried columns",
        "• Monitor slow queries and optimize them",
        "• Consider connection pooling configuration"
    ]))

    # I/O related recommendations
    recommendations.append(("I/O Optimization:", [
        "• Use SSD storage for better I/O performance",
        "• Optimize queries to reduce data access",
        "• Implement proper indexing strategy",
        "• Consider data partitioning for large tables",
        "• Monitor disk I/O patterns and bottlenecks"
    ]))

    # Memory-related recommendations
    recommendations.append(("Memory Optimization:", [
        "• Increase system memory if frequently swapping",
        "• Optimize application memory usage",
        "• Configure appropriate MySQL buffer sizes",
        "• Monitor memory leaks in long-running processes",
        "• Consider memory-efficient data structures"
    ]))

    # Network-related recommendations
    recommendations.append(("Network Optimization:", [
        "• Optimize connection pooling to reduce connection overhead",
        "• Minimize network round trips with batch operations",
        "• Compress data transfer when appropriate",
        "• Monitor network latency and throughput",
        "• Consider connection keep-alive settings"
    ]))

    for category, items in recommendations:
        print(f"\n{category}")
        for item in items:
            print(f"  {item}")

    print("\n" + "=" * 60)
    print("🔧 IMMEDIATE ACTION ITEMS:")
    print("1. Enable MySQL Performance Schema for detailed monitoring")
    print("2. Run EXPLAIN on slow queries to identify optimization opportunities")
    print("3. Monitor system resources during peak load")
    print("4. Review and optimize index usage")
    print("5. Check MySQL configuration parameters")

def run_comprehensive_analysis(duration: int = 30):
    """Run comprehensive performance analysis"""
    print("🔬 COMPREHENSIVE CPU & DATABASE PERFORMANCE ANALYSIS")
    print("=" * 80)
    print(f"Analysis Duration: {duration} seconds")
    print("=" * 80)

    # System resource monitoring
    monitor_system_resources(duration=duration)

    # MySQL analysis
    analyze_mysql_performance()

    # Thread analysis
    analyze_thread_activity()

    # System limits
    check_system_limits()

    # Additional analysis
    analyze_bottlenecks()

    # Workload simulation
    simulate_workload()

    # Generate recommendations
    generate_recommendations()

def main():
    """Main analysis function"""
    print("🔍 ADVANCED CPU & DATABASE PERFORMANCE ANALYSIS TOOL")
    print("=" * 60)

    # Quick analysis mode
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        print("Running quick analysis (30 seconds)...")
        run_comprehensive_analysis(duration=30)
    elif len(sys.argv) > 1 and sys.argv[1] == '--full':
        print("Running full analysis (120 seconds)...")
        run_comprehensive_analysis(duration=120)
    else:
        print("Running standard analysis (60 seconds)...")
        print("Use --quick for 30s analysis or --full for 120s analysis")
        run_comprehensive_analysis(duration=60)

if __name__ == "__main__":
    main()
