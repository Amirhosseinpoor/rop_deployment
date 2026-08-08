#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys


def ensure_waha_container():
    """Ensure the WAHA docker container is running when starting dev server."""
    if 'runserver' not in sys.argv or os.environ.get('RUN_MAIN') == 'true':
        return

    import shutil
    import subprocess

    if not shutil.which('docker'):
        print("[WAHA] Warning: Docker CLI not found. Skipping WAHA container check.")
        return

    try:
        res = subprocess.run(
            ['docker', 'inspect', '-f', '{{.State.Running}}', 'waha'],
            capture_output=True, text=True
        )

        if res.returncode == 0:
            if res.stdout.strip() == 'true':
                print("[WAHA] Docker container 'waha' is already running.")
                return
            else:
                print("[WAHA] Docker container 'waha' exists but is stopped. Starting container...")
                subprocess.run(['docker', 'start', 'waha'], check=True)
                print("[WAHA] Docker container 'waha' started successfully.")
                return
    except Exception:
        pass

    base_dir = os.path.dirname(os.path.abspath(__file__))
    env_file = os.path.join(base_dir, '.env')
    sessions_dir = os.path.join(base_dir, 'sessions')

    os.makedirs(sessions_dir, exist_ok=True)

    cmd = ['docker', 'run', '-d']
    if os.path.exists(env_file):
        cmd.extend(['--env-file', env_file])
    cmd.extend([
        '-v', f'{sessions_dir}:/app/.sessions',
        '--rm',
        '-p', '3009:3000',
        '--name', 'waha',
        'devlikeapro/waha'
    ])

    print("[WAHA] Starting WAHA docker container (devlikeapro/waha on port 3009)...")
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            print("[WAHA] Docker container 'waha' started successfully.")
        else:
            print(f"[WAHA] Failed to start container: {res.stderr.strip()}")
    except Exception as exc:
        print(f"[WAHA] Error executing docker command: {exc}")


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
    ensure_waha_container()
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
