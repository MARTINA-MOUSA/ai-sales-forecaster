import psutil
import time
import logging
import json
from datetime import datetime

# === إعداد التسجيل ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    encoding='utf-8'
)

def get_system_metrics():
    
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_usage = psutil.virtual_memory().percent
    disk_usage = psutil.disk_usage('/').percent

    metrics = {
        "cpu_usage": cpu_usage,
        "memory_usage": memory_usage,
        "disk_usage": disk_usage
    }

    logging.info(json.dumps(metrics, ensure_ascii=False))
    return metrics


def log_system_metrics(interval=5, duration=60, log_to_file=False):
    
    start_time = time.time()
    logging.info("Started system monitoring...")

    if log_to_file:
        filename = "monitoring_internal.log"
        file_handler = logging.FileHandler(filename, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)

    while (time.time() - start_time) < duration:
        get_system_metrics()
        time.sleep(interval)

    logging.info("Monitoring finished.")


if __name__ == "__main__":
    log_system_metrics(interval=10, duration=60, log_to_file=True)
