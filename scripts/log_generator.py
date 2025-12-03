"""
WebBotShield - Nginx Log Generator

Generates realistic Nginx access logs with both normal user traffic and bot patterns
for testing the bot detection system. Includes automatic log rotation.
"""

import random
import time
from datetime import datetime, timedelta
import argparse
from pathlib import Path
import os
import gzip
import shutil


class NginxLogGenerator:
    """Generate realistic Nginx access logs"""

    # Realistic User-Agents
    HUMAN_USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Mobile/15E148 Safari/604.1",
        "Mozilla/5.0 (iPad; CPU OS 17_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Mobile/15E148 Safari/604.1",
    ]

    BOT_USER_AGENTS = [
        "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
        "curl/7.68.0",
        "python-requests/2.31.0",
        "Scrapy/2.11.0",
        "Go-http-client/1.1",
        "Java/11.0.2",
        "Apache-HttpClient/4.5.13 (Java/11.0.2)",
        "PostmanRuntime/7.32.3",
        "wget/1.21.3",
        "",  # Empty user agent
    ]

    HTTP_METHODS = ["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"]

    # Realistic paths
    NORMAL_PATHS = [
        "/", "/index.html", "/about", "/contact", "/products", "/services",
        "/blog", "/blog/post-1", "/blog/post-2", "/api/users", "/api/products",
        "/static/css/style.css", "/static/js/app.js", "/images/logo.png",
        "/login", "/register", "/dashboard", "/profile", "/settings"
    ]

    # Bot-like paths (suspicious)
    BOT_PATHS = [
        "/admin", "/wp-admin", "/phpmyadmin", "/.env", "/config.php",
        "/backup.sql", "/admin/login", "/.git/config", "/api/internal",
        "/xmlrpc.php", "/wp-login.php", "/administrator"
    ]

    HTTP_STATUS_CODES = {
        'success': [200, 201, 204, 304],
        'client_error': [400, 401, 403, 404, 405, 429],
        'server_error': [500, 502, 503, 504]
    }

    REFERERS = [
        "https://www.google.com/",
        "https://www.bing.com/",
        "https://www.facebook.com/",
        "https://twitter.com/",
        "-"  # Direct access
    ]

    def __init__(self, output_file="logs/nginx/access.log", max_file_size_mb=100, max_backups=5):
        """Initialize log generator with log rotation

        Args:
            output_file: Path to the log file
            max_file_size_mb: Maximum size of log file before rotation (MB)
            max_backups: Maximum number of backup files to keep
        """
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.max_file_size = max_file_size_mb * 1024 * 1024  # Convert to bytes
        self.max_backups = max_backups
        self.last_rotation_check = time.time()

    def generate_ip(self, bot=False):
        """Generate random IP address"""
        if bot and random.random() < 0.7:  # Bots often reuse IPs
            return random.choice([
                f"10.0.{random.randint(0, 10)}.{random.randint(1, 50)}",
                f"192.168.1.{random.randint(1, 20)}"
            ])
        return f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 255)}"

    def generate_timestamp(self, base_time=None):
        """Generate timestamp in Nginx format"""
        if base_time is None:
            base_time = datetime.now()
        return base_time.strftime("%d/%b/%Y:%H:%M:%S +0000")

    def generate_normal_log(self, timestamp=None):
        """Generate normal human user log entry"""
        ip = self.generate_ip(bot=False)
        ts = self.generate_timestamp(timestamp)
        method = random.choice(["GET", "POST"])
        path = random.choice(self.NORMAL_PATHS)
        status_codes = self.HTTP_STATUS_CODES['success'] + self.HTTP_STATUS_CODES['client_error'][:2]
        # Weights: 4 success codes (85%), 2 client errors (15%)
        weights = [0.21, 0.21, 0.21, 0.22, 0.10, 0.05]  # Total = 1.0
        status = random.choices(status_codes, weights=weights)[0]
        bytes_sent = random.randint(500, 50000)
        referer = random.choice(self.REFERERS)
        user_agent = random.choice(self.HUMAN_USER_AGENTS)
        
        # Protocol (always TCP for HTTP), Port (80 or 443), Request_Type (HTTP or HTTPS)
        protocol = "TCP"
        port = random.choice([80, 443])
        request_type = "HTTPS" if port == 443 else "HTTP"

        return f'{ip} - - [{ts}] "{method} {path} HTTP/1.1" {status} {bytes_sent} "{referer}" "{user_agent}" protocol={protocol} port={port} request_type={request_type}\n'

    def generate_bot_log(self, timestamp=None, burst=False):
        """Generate bot-like log entry"""
        ip = self.generate_ip(bot=True)
        ts = self.generate_timestamp(timestamp)
        method = random.choice(["GET", "POST", "HEAD"])

        # Bots often access suspicious paths
        if random.random() < 0.6:
            path = random.choice(self.BOT_PATHS)
        else:
            path = random.choice(self.NORMAL_PATHS)

        # Bots tend to get more errors - avoid weights mismatch by using probabilities
        if random.random() < 0.15:  # 15% success
            status = random.choice(self.HTTP_STATUS_CODES['success'])
        elif random.random() < 0.80:  # 80% client errors (60% of remaining)
            status = random.choice(self.HTTP_STATUS_CODES['client_error'])
        else:  # 25% server errors
            status = random.choice(self.HTTP_STATUS_CODES['server_error'])

        bytes_sent = random.randint(200, 5000)
        referer = "-"
        user_agent = random.choice(self.BOT_USER_AGENTS)
        
        # Protocol (always TCP for HTTP), Port (80 or 443), Request_Type (HTTP or HTTPS)
        protocol = "TCP"
        port = random.choice([80, 443])
        request_type = "HTTPS" if port == 443 else "HTTP"

        return f'{ip} - - [{ts}] "{method} {path} HTTP/1.1" {status} {bytes_sent} "{referer}" "{user_agent}" protocol={protocol} port={port} request_type={request_type}\n'

    def generate_burst_attack(self, num_requests=150, timestamp=None):
        """Generate burst of requests from single IP (bot attack)"""
        logs = []
        bot_ip = self.generate_ip(bot=True)
        base_time = timestamp or datetime.now()

        for i in range(num_requests):
            ts_offset = base_time + timedelta(seconds=i)
            ts = self.generate_timestamp(ts_offset)
            method = "GET"
            path = random.choice(self.NORMAL_PATHS + self.BOT_PATHS)
            status = random.choice([200, 404, 403, 429])
            bytes_sent = random.randint(100, 1000)
            user_agent = random.choice(self.BOT_USER_AGENTS)
            
            # Protocol (always TCP for HTTP), Port (80 or 443), Request_Type (HTTP or HTTPS)
            protocol = "TCP"
            port = random.choice([80, 443])
            request_type = "HTTPS" if port == 443 else "HTTP"

            log = f'{bot_ip} - - [{ts}] "{method} {path} HTTP/1.1" {status} {bytes_sent} "-" "{user_agent}" protocol={protocol} port={port} request_type={request_type}\n'
            logs.append(log)

        return logs

    def rotate_logs(self):
        """Rotate log file if it exceeds max_file_size"""
        if not self.output_file.exists():
            return

        file_size = self.output_file.stat().st_size

        if file_size >= self.max_file_size:
            print(f"\n[{datetime.now()}] Log file size ({file_size / 1024 / 1024:.1f}MB) exceeded limit, rotating...")

            # Move existing backup files
            for i in range(self.max_backups - 1, 0, -1):
                old_backup = Path(f"{self.output_file}.{i}.gz")
                new_backup = Path(f"{self.output_file}.{i + 1}.gz")
                if old_backup.exists():
                    old_backup.rename(new_backup)

            # Compress current log file
            backup_file = Path(f"{self.output_file}.1.gz")
            with open(self.output_file, 'rb') as f_in:
                with gzip.open(backup_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Clear original log file
            self.output_file.write_text('')

            # Delete oldest backup if we exceed max_backups
            oldest_backup = Path(f"{self.output_file}.{self.max_backups + 1}.gz")
            if oldest_backup.exists():
                oldest_backup.unlink()

            print(f"[{datetime.now()}] Log rotated. Active backups: {len([f for f in self.output_file.parent.glob(f'{self.output_file.name}.*')])}")

    def generate_continuous(self, duration_seconds=300, normal_rate=5, bot_rate=2, burst_probability=0.1):
        """
        Generate continuous stream of logs

        Args:
            duration_seconds: How long to generate logs (seconds). 0 = infinite
            normal_rate: Normal user requests per second
            bot_rate: Bot requests per second
            burst_probability: Probability of burst attack per minute
        """
        duration_text = "indefinitely" if duration_seconds == 0 else f"for {duration_seconds} seconds"
        print(f"Starting log generation {duration_text}...")
        print(f"Normal rate: {normal_rate} req/s, Bot rate: {bot_rate} req/s")
        print(f"Output file: {self.output_file}")

        start_time = time.time()
        log_count = 0

        with open(self.output_file, 'a') as f:
            while duration_seconds == 0 or (time.time() - start_time < duration_seconds):
                current_time = datetime.now()

                # Generate normal traffic
                for _ in range(normal_rate):
                    log = self.generate_normal_log(current_time)
                    f.write(log)
                    log_count += 1

                # Generate bot traffic
                for _ in range(bot_rate):
                    log = self.generate_bot_log(current_time)
                    f.write(log)
                    log_count += 1

                # Random burst attack
                if random.random() < burst_probability:
                    print(f"[{datetime.now()}] Generating burst attack...")
                    burst_logs = self.generate_burst_attack(
                        num_requests=random.randint(100, 200),
                        timestamp=current_time
                    )
                    for log in burst_logs:
                        f.write(log)
                        log_count += len(burst_logs)

                f.flush()  # Ensure logs are written immediately

                # Check for log rotation every 60 seconds
                if time.time() - self.last_rotation_check >= 60:
                    self.last_rotation_check = time.time()
                    self.rotate_logs()

                time.sleep(1)  # 1 second interval

        elapsed = time.time() - start_time
        print(f"\nLog generation complete!")
        print(f"Total logs generated: {log_count}")
        print(f"Duration: {elapsed:.2f} seconds")
        print(f"Average rate: {log_count/elapsed:.2f} logs/sec")

    def generate_batch(self, num_normal=100, num_bot=20, num_bursts=2):
        """Generate batch of logs for testing"""
        print(f"Generating batch: {num_normal} normal + {num_bot} bot + {num_bursts} bursts")

        with open(self.output_file, 'w') as f:
            # Generate normal logs
            for _ in range(num_normal):
                log = self.generate_normal_log()
                f.write(log)

            # Generate bot logs
            for _ in range(num_bot):
                log = self.generate_bot_log()
                f.write(log)

            # Generate bursts
            for _ in range(num_bursts):
                burst_logs = self.generate_burst_attack()
                for log in burst_logs:
                    f.write(log)

        print(f"Batch generation complete! Logs written to {self.output_file}")


def main():
    parser = argparse.ArgumentParser(description="WebBotShield - Nginx Log Generator")
    parser.add_argument(
        '--mode',
        choices=['batch', 'continuous'],
        default='continuous',
        help='Generation mode: batch or continuous'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=300,
        help='Duration in seconds for continuous mode (default: 300)'
    )
    parser.add_argument(
        '--normal-rate',
        type=int,
        default=5,
        help='Normal requests per second (default: 5)'
    )
    parser.add_argument(
        '--bot-rate',
        type=int,
        default=2,
        help='Bot requests per second (default: 2)'
    )
    parser.add_argument(
        '--burst-probability',
        type=float,
        default=0.05,
        help='Burst attack probability per second (default: 0.05)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='logs/nginx/access.log',
        help='Output log file path'
    )
    parser.add_argument(
        '--max-file-size-mb',
        type=int,
        default=100,
        help='Maximum log file size before rotation in MB (default: 100)'
    )
    parser.add_argument(
        '--max-backups',
        type=int,
        default=5,
        help='Maximum number of compressed backup files to keep (default: 5)'
    )

    args = parser.parse_args()

    generator = NginxLogGenerator(
        output_file=args.output,
        max_file_size_mb=args.max_file_size_mb,
        max_backups=args.max_backups
    )

    if args.mode == 'continuous':
        generator.generate_continuous(
            duration_seconds=args.duration,
            normal_rate=args.normal_rate,
            bot_rate=args.bot_rate,
            burst_probability=args.burst_probability
        )
    else:
        generator.generate_batch()


if __name__ == "__main__":
    main()
