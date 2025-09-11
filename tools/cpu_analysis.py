#!/usr/bin/env python3
"""
CPU Performance Analysis Tool
"""

import time
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor
import pymysql
import pandas as pd

def analyze_mysql_performance():
    """Analyze MySQL performance bottlenecks"""
    print("=== MySQL Performance Analysis ===")

    try:
        conn = pymysql.connect(
            host='127.0.0.1',
            port=3307,
            user='root',
            password='',
            database='investment_data',
            connect_timeout=5
        )

        with conn.cursor() as cursor:
            # Check connection count
            cursor.execute("SHOW PROCESSLIST")
            processes = cursor.fetchall()
            active_connections = len([p for p in processes if p[6] not in ['Sleep', '']])

            print(f"Total MySQL connections: {len(processes)}")
            print(f"Active connections: {active_connections}")

            # Check slow queries
            cursor.execute("""
                SELECT sql_text, exec_count, avg_timer_wait/1000000000 as avg_time_sec,
                       rows_examined, rows_sent
                FROM performance_schema.events_statements_summary_by_digest
                WHERE avg_timer_wait > 1000000000  -- > 1 second
                ORDER BY avg_timer_wait DESC
                LIMIT 10
            """)

            slow_queries = cursor.fetchall()
            if slow_queries:
                print(f"\nSlow queries (>1s): {len(slow_queries)} found")
                for i, query in enumerate(slow_queries[:5]):
                    print(".3f")
            else:
                print("\nNo slow queries found")

            # Check InnoDB status
            cursor.execute("SHOW ENGINE INNODB STATUS")
            innodb_status = cursor.fetchone()

            if innodb_status and len(innodb_status) > 2:
                status_text = innodb_status[2]
                # Look for semaphore waits
                if 'semaphores' in status_text.lower():
                    print("\nInnoDB semaphore waits detected - possible lock contention")

            # Check table locks
            cursor.execute("""
                SELECT table_schema, table_name, wait_timeout,
                       COUNT(*) as lock_count
                FROM information_schema.innodb_lock_waits
                GROUP BY table_schema, table_name, wait_timeout
                ORDER BY lock_count DESC
            """)

            locks = cursor.fetchall()
            if locks:
                print(f"\nTable locks detected: {len(locks)}")
                for lock in locks[:3]:
                    print(f"  {lock[0]}.{lock[1]}: {lock[3]} locks")

        conn.close()

    except Exception as e:
        print(f"MySQL analysis failed: {e}")

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

    print(f"Thread states: {thread_states}")

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

def main():
    """Main analysis function"""
    print("🔍 CPU Performance Analysis Tool")
    print("=" * 50)

    analyze_mysql_performance()
    analyze_thread_activity()
    check_system_limits()
    analyze_bottlenecks()
    simulate_workload()

    print("\n" + "=" * 50)
    print("📋 Recommendations:")
    print("1. If I/O bound: Consider using SSD storage, optimize queries")
    print("2. If CPU bound: Consider increasing thread count or using multiprocessing")
    print("3. If memory bound: Implement streaming processing, reduce batch sizes")
    print("4. If network bound: Optimize connection pooling, reduce round trips")
    print("5. Check MySQL configuration: innodb_buffer_pool_size, query_cache_size")

if __name__ == "__main__":
    main()
