# src/monitoring.py
import psutil
import time
import logging
import json
from datetime import datetime

logging.basicConfig(
    filename='logs/monitoring.log',
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    encoding='utf-8'
)

def log_system_metrics(interval=5, duration=60):
    """
    Monitor system performance: CPU, Memory, Disk usage.
    interval: seconds between samples
    duration: total monitoring duration in seconds
    """
    start_time = time.time()
    logging.info("Started system monitoring...")

    while (time.time() - start_time) < duration:
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage('/').percent

        metrics = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "disk_usage": disk_usage
        }

        logging.info(json.dumps(metrics))
        time.sleep(interval)

    logging.info("Monitoring finished.")


if __name__ == "__main__":
    log_system_metrics(interval=10, duration=120)
