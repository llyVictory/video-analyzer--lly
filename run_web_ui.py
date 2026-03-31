import os
import sys
import subprocess
import socket

def get_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

def run():
    print("=== Video Analyzer Web UI Starter ===")
    
    # Check for dependencies
    try:
        import flask
        import dotenv
    except ImportError:
        print("Installing missing dependencies (flask, python-dotenv)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "flask", "python-dotenv"])

    # Set project root to PYTHONPATH to ensure video_analyzer can be imported
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.environ['PYTHONPATH'] = project_root + os.pathsep + os.environ.get('PYTHONPATH', '')
    
    # Path to the server script
    server_script = os.path.join(project_root, "video-analyzer-ui", "video_analyzer_ui", "server.py")
    
    if not os.path.exists(server_script):
        print(f"Error: Could not find server script at {server_script}")
        return

    local_ip = get_ip()
    port = "8080" # 5000 is often used by system services, using 8080 for better compatibility
    
    print("\n" + "="*50)
    print(f" Web UI is ready!")
    print(f" Local Access:   http://localhost:{port}")
    print(f" Network Access: http://{local_ip}:{port} (From Windows Browser)")
    print("="*50 + "\n")
    print("Press Ctrl+C to stop.")
    
    try:
        # Use 0.0.0.0 to allow Windows host to access WSL2 service
        subprocess.run([sys.executable, server_script, "--dev", "--host", "0.0.0.0", "--port", port])
    except KeyboardInterrupt:
        print("\nShutting down server...")

if __name__ == "__main__":
    run()
