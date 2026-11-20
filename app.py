import os
import json
import subprocess
import signal
import threading
import time


from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from pathlib import Path

user_home_path = Path.home()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
server_process = None
server_port = None  # port the running llama-server was launched with
rpc_worker_process = None
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')  # next to app.py, regardless of CWD

# Defaults — overridden by load_config settings
DEFAULT_LLAMA_SERVER_HOST = "0.0.0.0"
DEFAULT_LLAMA_SERVER_PORT = 10000
DEFAULT_LLAMA_SERVER_PATH = f'{user_home_path}/llama.cpp/build/bin/llama-server'
DEFAULT_RPC_SERVER_PATH = f'{user_home_path}/llama.cpp/build/bin/rpc-server'
DEFAULT_RPC_WORKER_HOST = "0.0.0.0"
DEFAULT_RPC_WORKER_PORT = 50052

# Runtime values (set from config)
LLAMA_SERVER_HOST = DEFAULT_LLAMA_SERVER_HOST
LLAMA_SERVER_PORT = DEFAULT_LLAMA_SERVER_PORT
LLAMA_SERVER_PATH = DEFAULT_LLAMA_SERVER_PATH

def get_default_settings():
    """Return default settings for first-run"""
    return {
        "llama_server_path": DEFAULT_LLAMA_SERVER_PATH,
        "llama_server_host": DEFAULT_LLAMA_SERVER_HOST,
        "llama_server_port": DEFAULT_LLAMA_SERVER_PORT,
        "rpc_server_path": DEFAULT_RPC_SERVER_PATH,
        "rpc_worker": {
            "host": DEFAULT_RPC_WORKER_HOST,
            "port": DEFAULT_RPC_WORKER_PORT,
            "cache_dir": ""
        },
        "defaults": {
            "n_gpu_layers": -1,
            "n_ctx": 8192,
            "n_generate_tokens": 4096,
            "mmproj_path": "",
            "custom_flags": ""
        }
    }


def load_config():
    """Load configuration from JSON file"""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        # Coordinator RPC worker endpoints live at top level so a full-replace
        # of `settings` (via the Settings page) can't wipe them.
        config.setdefault("rpc_endpoints", [])
        return config
    return {"models": [], "settings": get_default_settings(), "rpc_endpoints": []}

def save_config(config):
    """Save configuration to JSON file"""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

def read_process_output(process, socketio):
    """Read and emit process output in real-time"""
    server_started = False
    try:
        for line in iter(process.stdout.readline, ''):
            if line:
                line = line.strip()
                socketio.emit('console_output', {'data': line})
                
                # Check for server start message
                if "main: server is listening on" in line:
                    server_started = True
                    socketio.emit('loading_complete', {'success': True, 'message': 'Model loaded successfully!'})
    except Exception as e:
        socketio.emit('console_output', {'data': f'Error reading output: {str(e)}'})

    # Output ended -> process exited. If it never became ready, loading failed.
    if not server_started:
        process.wait()
        socketio.emit('loading_complete', {'success': False, 'message': f'llama-server exited before it was ready (exit code {process.returncode})'})

def read_rpc_worker_output(process, socketio):
    """Read and emit rpc-server worker output in real-time"""
    worker_started = False
    try:
        for line in iter(process.stdout.readline, ''):
            if line:
                line = line.strip()
                socketio.emit('console_output', {'data': line})

                # rpc-server prints these on a successful bind/listen
                if not worker_started and ("Starting RPC server" in line or "Accepting" in line or "listening" in line.lower()):
                    worker_started = True
                    socketio.emit('rpc_worker_started', {'success': True, 'message': 'RPC worker is running'})
    except Exception as e:
        socketio.emit('console_output', {'data': f'Error reading RPC worker output: {str(e)}'})
    finally:
        # Stream ended -> process exited
        socketio.emit('rpc_worker_stopped', {'message': 'RPC worker process ended'})

@app.route('/')
def index():
    settings = load_config().get('settings', get_default_settings())
    return render_template('index.html', defaults=settings.get('defaults', get_default_settings()['defaults']))

@app.route('/settings')
def settings_page():
    settings = load_config().get('settings', get_default_settings())
    return render_template('settings.html', settings=settings)

@app.route('/api/models', methods=['GET'])
def get_models():
    config = load_config()
    return jsonify(config)

@app.route('/api/models', methods=['POST'])
def add_model():
    config = load_config()
    new_model = request.json
    
    # Validate required fields
    required_fields = ['model_name', 'path_to_gguf_file', 'n_gpu_layers', 'n_ctx', 'n_generate_tokens']
    for field in required_fields:
        if field not in new_model:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    # Check if model already exists
    for model in config['models']:
        if model['model_name'] == new_model['model_name']:
            return jsonify({'error': 'Model with this name already exists'}), 400
    
    # Check if file exists
    if not os.path.exists(new_model['path_to_gguf_file']):
        return jsonify({'error': 'GGUF file not found at specified path'}), 400
    
    config['models'].append(new_model)
    save_config(config)
    return jsonify({'success': True, 'message': 'Model added successfully'})

@app.route('/api/models/<path:model_name>', methods=['PUT'])
def update_model(model_name):
    config = load_config()
    updated_model = request.json

    # Renaming must not collide with another existing model
    new_name = updated_model.get('model_name', model_name)
    if new_name != model_name and any(m['model_name'] == new_name for m in config['models']):
        return jsonify({'error': 'Model with this name already exists'}), 400

    for i, model in enumerate(config['models']):
        if model['model_name'] == model_name:
            # Check if file exists for updated path
            if 'path_to_gguf_file' in updated_model:
                if not os.path.exists(updated_model['path_to_gguf_file']):
                    return jsonify({'error': 'GGUF file not found at specified path'}), 400
            
            config['models'][i] = updated_model
            save_config(config)
            return jsonify({'success': True, 'message': 'Model updated successfully'})
    
    return jsonify({'error': 'Model not found'}), 404

@app.route('/api/models/<path:model_name>', methods=['DELETE'])
def delete_model(model_name):
    config = load_config()
    config['models'] = [m for m in config['models'] if m['model_name'] != model_name]
    save_config(config)
    return jsonify({'success': True, 'message': 'Model deleted successfully'})

@app.route('/api/load', methods=['POST'])
def load_model():
    global server_process, server_port

    # Check if server is already running
    if server_process and server_process.poll() is None:
        return jsonify({'error': 'Model is already loaded'}), 400
    
    data = request.json
    #print(data)
    #breakpoint()
    model_path = data.get('path_to_gguf_file')
    n_gpu_layers = data.get('n_gpu_layers', 0)
    n_ctx = data.get('n_ctx', 0)
    n_predict = data.get('n_generate_tokens', -1)
    # Use per-model llama-server path if set, otherwise global
    server_path = data.get('llama_server_path', '').strip() or LLAMA_SERVER_PATH
    print(f"Manager: Loading model with parameters: n_gpu_layers {str(n_gpu_layers)}, n_ctx: {str(n_ctx)}, n_predict: {str(n_predict)}")
    
    if not model_path:
        return jsonify({'error': 'Model path is required'}), 400
    
    if not os.path.exists(model_path):
        return jsonify({'error': 'Model file not found'}), 400
    
    # Build the command
    cmd = [
        server_path,
        '--host', LLAMA_SERVER_HOST,
        '--model', model_path,
        '--port', str(LLAMA_SERVER_PORT),
        '-ngl', str(n_gpu_layers),
        '--ctx-size', str(n_ctx),
        '--predict', str(n_predict), 
	    #'--jinja',
	    #'--no-webui',
	    #'--ignore-eos',
	    #'--override-tensor', ".*ffn_.*_exps.*=CPU",
    ]
    
    # Add mmproj if provided
    mmproj_path = data.get('mmproj_path', '')
    if mmproj_path and mmproj_path.strip():
        cmd.extend(['--mmproj', mmproj_path])

    # Add RPC worker endpoints if provided (distributed inference).
    # Accepts a list of "host:port" or a comma-separated string.
    rpc_servers = data.get('rpc_servers', '')
    if isinstance(rpc_servers, list):
        rpc_servers = ','.join(s.strip() for s in rpc_servers if s and s.strip())
    if rpc_servers and rpc_servers.strip():
        cmd.extend(['--rpc', rpc_servers.strip()])

    # Add custom flags if provided
    custom_flags = data.get('custom_flags', '')
    if custom_flags and custom_flags.strip():
        # Split custom flags by spaces, respecting quoted strings
        import shlex
        try:
            custom_flags_list = shlex.split(custom_flags)
            cmd.extend(custom_flags_list)
        except Exception as e:
            return jsonify({'error': f'Invalid custom flags: {str(e)}'}), 400
    
    try:
        # Start the server process
        server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        server_port = LLAMA_SERVER_PORT

        # Start a thread to read output
        output_thread = threading.Thread(
            target=read_process_output,
            args=(server_process, socketio)
        )
        output_thread.daemon = True
        output_thread.start()
	
        #return jsonify({'success': True, 'message': 'Loading model...'})
        return jsonify({'success': True, 'message': f"Manager: Loading model with parameters: n_gpu_layers {str(n_gpu_layers)}, n_ctx: {str(n_ctx)}, n_predict: {str(n_predict)} ..."})
        
    except Exception as e:
        return jsonify({'error': f'Failed to start server: {str(e)}'}), 500

@app.route('/api/unload', methods=['POST'])
def unload_model():
    global server_process
    
    if server_process and server_process.poll() is None:
        try:
            server_process.terminate()
            server_process.wait(timeout=5)
            server_process = None
            socketio.emit('model_unloaded', {'success': True, 'message': 'Model unloaded successfully'})
            return jsonify({'success': True, 'message': 'Model unloaded successfully'})
        except subprocess.TimeoutExpired:
            server_process.kill()
            server_process = None
            socketio.emit('model_unloaded', {'success': True, 'message': 'Model force killed'})
            return jsonify({'success': True, 'message': 'Model force killed'})
        except Exception as e:
            return jsonify({'error': f'Failed to unload model: {str(e)}'}), 500
    else:
        return jsonify({'error': 'No model is currently loaded'}), 400

@app.route('/api/status', methods=['GET'])
def get_status():
    global server_process, LLAMA_SERVER_HOST, LLAMA_SERVER_PORT, LLAMA_SERVER_PATH
    is_running = server_process and server_process.poll() is None
    config = load_config()
    settings = config.get('settings', get_default_settings())
    return jsonify({
        'is_loaded': is_running,
        'llama_server_path': settings.get('llama_server_path', LLAMA_SERVER_PATH),
        'llama_server_host': settings.get('llama_server_host', LLAMA_SERVER_HOST),
        # While running, report the port llama-server was actually launched with
        'llama_server_port': server_port if is_running else settings.get('llama_server_port', LLAMA_SERVER_PORT)
    })

@app.route('/api/settings', methods=['GET'])
def get_settings():
    config = load_config()
    settings = config.get('settings', get_default_settings())
    return jsonify(settings)

@app.route('/api/settings', methods=['PUT'])
def update_settings():
    global LLAMA_SERVER_HOST, LLAMA_SERVER_PORT, LLAMA_SERVER_PATH
    new_settings = request.json
    
    config = load_config()
    
    # Validate port
    port = new_settings.get('llama_server_port', DEFAULT_LLAMA_SERVER_PORT)
    try:
        port = int(port)
        if port < 1 or port > 65535:
            return jsonify({'error': 'Port must be between 1 and 65535'}), 400
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid port number'}), 400
    
    # Merge settings
    config['settings'] = new_settings
    save_config(config)
    
    # Update runtime values
    LLAMA_SERVER_PATH = new_settings.get('llama_server_path', LLAMA_SERVER_PATH)
    LLAMA_SERVER_HOST = new_settings.get('llama_server_host', LLAMA_SERVER_HOST)
    LLAMA_SERVER_PORT = port
    
    print(f"Settings updated: path={LLAMA_SERVER_PATH}, host={LLAMA_SERVER_HOST}, port={LLAMA_SERVER_PORT}")
    return jsonify({'success': True, 'message': 'Settings saved successfully'})

@app.route('/rpc')
def rpc_page():
    config = load_config()
    settings = config.get('settings', get_default_settings())
    return render_template('rpc.html',
                           settings=settings,
                           rpc_endpoints=config.get('rpc_endpoints', []))

@app.route('/api/rpc/start_worker', methods=['POST'])
def start_rpc_worker():
    global rpc_worker_process

    if rpc_worker_process and rpc_worker_process.poll() is None:
        return jsonify({'error': 'RPC worker is already running'}), 400

    data = request.json or {}
    settings = load_config().get('settings', get_default_settings())
    rpc_worker_defaults = settings.get('rpc_worker', {})

    server_path = (data.get('rpc_server_path', '') or '').strip() \
        or settings.get('rpc_server_path', DEFAULT_RPC_SERVER_PATH)
    host = (data.get('host', '') or '').strip() \
        or rpc_worker_defaults.get('host', DEFAULT_RPC_WORKER_HOST)
    port = data.get('port', rpc_worker_defaults.get('port', DEFAULT_RPC_WORKER_PORT))
    cache_dir = (data.get('cache_dir', '') or '').strip() \
        or rpc_worker_defaults.get('cache_dir', '')

    try:
        port = int(port)
        if port < 1 or port > 65535:
            return jsonify({'error': 'Port must be between 1 and 65535'}), 400
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid port number'}), 400

    # Build the command: rpc-server -H <host> -p <port>
    cmd = [server_path, '-H', host, '-p', str(port)]

    # Add custom flags if provided
    custom_flags = data.get('custom_flags', '')
    custom_flags_list = []
    if custom_flags and custom_flags.strip():
        import shlex
        try:
            custom_flags_list = shlex.split(custom_flags)
            cmd.extend(custom_flags_list)
        except Exception as e:
            return jsonify({'error': f'Invalid custom flags: {str(e)}'}), 400

    # Custom cache directory for shared model weights: enable rpc-server's local
    # file cache (-c) and point LLAMA_CACHE at the chosen directory (rpc-server
    # stores received weights under <cache_dir>/rpc/).
    worker_env = os.environ.copy()
    if cache_dir:
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except Exception as e:
            return jsonify({'error': f'Could not create cache directory: {str(e)}'}), 400
        if '-c' not in custom_flags_list and '--cache' not in custom_flags_list:
            cmd.append('-c')
        worker_env['LLAMA_CACHE'] = cache_dir

    print(f"Manager: Starting RPC worker on {host}:{port} ({server_path})"
          + (f", cache_dir={cache_dir}" if cache_dir else ""))

    try:
        rpc_worker_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env=worker_env
        )

        output_thread = threading.Thread(
            target=read_rpc_worker_output,
            args=(rpc_worker_process, socketio)
        )
        output_thread.daemon = True
        output_thread.start()

        return jsonify({'success': True, 'message': f'Starting RPC worker on {host}:{port} ...'})
    except Exception as e:
        return jsonify({'error': f'Failed to start RPC worker: {str(e)}'}), 500

@app.route('/api/rpc/stop_worker', methods=['POST'])
def stop_rpc_worker():
    global rpc_worker_process

    if rpc_worker_process and rpc_worker_process.poll() is None:
        try:
            rpc_worker_process.terminate()
            rpc_worker_process.wait(timeout=5)
            rpc_worker_process = None
            socketio.emit('rpc_worker_stopped', {'message': 'RPC worker stopped'})
            return jsonify({'success': True, 'message': 'RPC worker stopped'})
        except subprocess.TimeoutExpired:
            rpc_worker_process.kill()
            rpc_worker_process = None
            socketio.emit('rpc_worker_stopped', {'message': 'RPC worker force killed'})
            return jsonify({'success': True, 'message': 'RPC worker force killed'})
        except Exception as e:
            return jsonify({'error': f'Failed to stop RPC worker: {str(e)}'}), 500
    else:
        return jsonify({'error': 'No RPC worker is currently running'}), 400

@app.route('/api/rpc/status', methods=['GET'])
def rpc_status():
    worker_running = rpc_worker_process and rpc_worker_process.poll() is None
    server_running = server_process and server_process.poll() is None
    return jsonify({
        'rpc_worker_running': bool(worker_running),
        'server_running': bool(server_running)
    })

@app.route('/api/rpc/endpoints', methods=['GET'])
def get_rpc_endpoints():
    config = load_config()
    return jsonify({'rpc_endpoints': config.get('rpc_endpoints', [])})

@app.route('/api/rpc/endpoints', methods=['POST'])
def save_rpc_endpoints():
    data = request.json or {}
    endpoints = data.get('rpc_endpoints', [])
    if not isinstance(endpoints, list):
        return jsonify({'error': 'rpc_endpoints must be a list'}), 400

    # Normalize: trim, drop blanks, dedupe while keeping order
    cleaned = []
    for ep in endpoints:
        ep = str(ep).strip()
        if ep and ep not in cleaned:
            cleaned.append(ep)

    config = load_config()
    config['rpc_endpoints'] = cleaned
    save_config(config)
    return jsonify({'success': True, 'message': 'RPC endpoints saved', 'rpc_endpoints': cleaned})

if __name__ == '__main__':
    # Create default config if it doesn't exist
    if not os.path.exists(CONFIG_FILE):
        default_config = {
            "models": [
                {
                    "model_name": "Example Model",
                    "path_to_gguf_file": "/path/to/your/model.gguf",
                    "n_gpu_layers": -1,     #Auto calculate GPU layers by default
                    "n_ctx": 8192,
                    "n_generate_tokens": 4096,
                    "mmproj_path": "",
                    "custom_flags": ""
                }
            ],
            "settings": get_default_settings()
        }
        save_config(default_config)
    
    # Apply settings from config
    config = load_config()
    settings = config.get('settings', get_default_settings())
    LLAMA_SERVER_PATH = settings.get('llama_server_path', LLAMA_SERVER_PATH)
    LLAMA_SERVER_HOST = settings.get('llama_server_host', LLAMA_SERVER_HOST)
    LLAMA_SERVER_PORT = settings.get('llama_server_port', LLAMA_SERVER_PORT)
    print(f"Settings loaded: path={LLAMA_SERVER_PATH}, host={LLAMA_SERVER_HOST}, port={LLAMA_SERVER_PORT}")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
