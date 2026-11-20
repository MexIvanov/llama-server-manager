import os
import json
import shlex
import shutil
import stat
import subprocess
import signal
import threading
import time
from collections import deque


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
script_runs = {}  # custom script name -> {'proc', 'ready', 'stopped', 'ready_pattern', 'log'}
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
        config.setdefault("custom_scripts", [])
        return config
    return {"models": [], "settings": get_default_settings(), "rpc_endpoints": [], "custom_scripts": []}

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

# ---------------------------------------------------------------------------
# Custom scripts: launch any model server (vLLM, SGLang, ...) via bash,
# optionally inside a conda env or venv. Several can run at once.
# ---------------------------------------------------------------------------

SCRIPT_ENV_TYPES = ('none', 'conda', 'venv')

def validate_script(data):
    """Return an error message for an invalid script profile, or None"""
    if not (data.get('name') or '').strip():
        return 'Name is required'
    if not (data.get('commands') or '').strip():
        return 'Run commands are required'
    env_type = data.get('env_type') or 'none'
    if env_type not in SCRIPT_ENV_TYPES:
        return f'Invalid environment type: {env_type}'
    if env_type != 'none' and not (data.get('env') or '').strip():
        return 'Environment name/path is required'
    for field, label in (('script_path', 'Script path'), ('workdir', 'Working directory')):
        value = (data.get(field) or '').strip()
        if value and not os.path.isabs(os.path.expanduser(value)):
            return f'{label} must be an absolute path'
    if (data.get('script_content') or '').strip() and not (data.get('script_path') or '').strip():
        return 'Script path is required to save script contents'
    return None

def clean_script(data):
    """The fields stored in config. Script contents live on disk, not in config."""
    return {
        'name': data['name'].strip(),
        'script_path': (data.get('script_path') or '').strip(),
        'env_type': data.get('env_type') or 'none',
        'env': (data.get('env') or '').strip(),
        'commands': data['commands'],
        'workdir': (data.get('workdir') or '').strip(),
        'ready_pattern': (data.get('ready_pattern') or '').strip(),
    }

def read_script_file(path):
    """Return a script file's text, or None if it's missing or not text"""
    try:
        with open(os.path.expanduser(path), 'r', encoding='utf-8') as f:
            return f.read()
    except (OSError, UnicodeDecodeError):
        return None

def save_script_content(data, script):
    """Write submitted script contents to script_path (if any). Returns an error message or None."""
    content = data.get('script_content') or ''
    if not content.strip():
        return None  # empty textarea: leave the file on disk alone
    path = os.path.expanduser(script['script_path'])
    try:
        if read_script_file(path) != content:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
    except OSError as e:
        return f'Could not save script file: {e}'
    return None

def find_conda():
    """Locate the conda executable (a systemd service usually doesn't have it on PATH)"""
    candidates = [os.environ.get('CONDA_EXE'), shutil.which('conda')]
    candidates += [os.path.join(user_home_path, d, 'bin', 'conda')
                   for d in ('miniconda3', 'anaconda3', 'miniforge3', 'mambaforge')]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None

def build_shell_command(script, conda=None):
    """Bash code: environment activation (if any), then the user's commands"""
    env_type = script.get('env_type', 'none')
    env = os.path.expanduser(script.get('env', ''))
    lines = []
    if env_type == 'conda':
        lines.append(f'eval "$({shlex.quote(conda)} shell.bash hook)" || '
                     f'{{ echo {shlex.quote("Failed to initialize conda")}; exit 1; }}')
        lines.append(f'conda activate {shlex.quote(env)} || '
                     f'{{ echo {shlex.quote("Failed to activate conda env: " + env)}; exit 1; }}')
    elif env_type == 'venv':
        activate = os.path.join(env, 'bin', 'activate')
        lines.append(f'source {shlex.quote(activate)} || '
                     f'{{ echo {shlex.quote("Failed to activate venv: " + env)}; exit 1; }}')
    lines.append(script['commands'])
    return '\n'.join(lines)

def script_running(name):
    run = script_runs.get(name)
    return run is not None and run['proc'].poll() is None

def script_state(run):
    """One of: starting (waiting for ready pattern), ready, running, stopped, failed"""
    code = run['proc'].poll()
    if code is None:
        if run['ready']:
            return {'state': 'ready', 'exit_code': None}
        return {'state': 'starting' if run['ready_pattern'] else 'running', 'exit_code': None}
    return {'state': 'stopped' if code == 0 or run['stopped'] else 'failed', 'exit_code': code}

def script_log(name, run, line):
    run['log'].append(line)
    socketio.emit('script_output', {'name': name, 'data': line})

def read_script_output(name, run):
    """Stream a custom script's output to its log buffer and the Custom Scripts page"""
    proc, pattern = run['proc'], run['ready_pattern']
    try:
        for line in iter(proc.stdout.readline, ''):
            line = line.rstrip()
            script_log(name, run, line)
            if pattern and not run['ready'] and pattern in line:
                run['ready'] = True
                socketio.emit('script_status', {'name': name, 'state': 'ready', 'message': f'{name} is ready'})
    except Exception as e:
        script_log(name, run, f'Error reading output: {str(e)}')

    proc.wait()
    message = f'{name} stopped' if run['stopped'] else f'{name} exited with code {proc.returncode}'
    run['log'].append(message)
    socketio.emit('script_status', {'name': name, **script_state(run), 'message': message})

def stop_process_group(proc, timeout=10):
    """SIGTERM the script's whole process group (bash plus e.g. vLLM workers), SIGKILL after timeout"""
    pgid = proc.pid  # start_new_session=True makes the bash process the group leader
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        proc.poll()  # reap bash so it doesn't linger as a zombie in the group
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            break  # every process in the group is gone
        time.sleep(0.2)
    else:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.wait()

@app.route('/scripts')
def scripts_page():
    return render_template('scripts.html')

@app.route('/api/scripts', methods=['GET'])
def get_scripts():
    scripts = load_config()['custom_scripts']
    for s in scripts:
        s['script_content'] = read_script_file(s['script_path']) if s.get('script_path') else None
    return jsonify({'custom_scripts': scripts})

@app.route('/api/scripts/status', methods=['GET'])
def get_scripts_status():
    return jsonify({name: script_state(run) for name, run in list(script_runs.items())})

@app.route('/api/scripts', methods=['POST'])
def add_script():
    data = request.json or {}
    error = validate_script(data)
    if error:
        return jsonify({'error': error}), 400

    config = load_config()
    script = clean_script(data)
    if any(s['name'] == script['name'] for s in config['custom_scripts']):
        return jsonify({'error': 'A script with this name already exists'}), 400

    error = save_script_content(data, script)
    if error:
        return jsonify({'error': error}), 400

    config['custom_scripts'].append(script)
    save_config(config)
    return jsonify({'success': True, 'message': 'Script profile added', 'name': script['name']})

@app.route('/api/scripts/<path:name>', methods=['PUT'])
def update_script(name):
    data = request.json or {}
    error = validate_script(data)
    if error:
        return jsonify({'error': error}), 400

    config = load_config()
    script = clean_script(data)
    index = next((i for i, s in enumerate(config['custom_scripts']) if s['name'] == name), None)
    if index is None:
        return jsonify({'error': 'Script not found'}), 404
    renamed = script['name'] != name
    if renamed and any(s['name'] == script['name'] for s in config['custom_scripts']):
        return jsonify({'error': 'A script with this name already exists'}), 400
    if renamed and script_running(name):
        return jsonify({'error': 'Stop the script before renaming it'}), 400

    error = save_script_content(data, script)
    if error:
        return jsonify({'error': error}), 400

    config['custom_scripts'][index] = script
    save_config(config)
    if renamed and name in script_runs:
        script_runs[script['name']] = script_runs.pop(name)  # keep its last log under the new name
    return jsonify({'success': True, 'message': 'Script profile updated', 'name': script['name']})

@app.route('/api/scripts/<path:name>', methods=['DELETE'])
def delete_script(name):
    if script_running(name):
        return jsonify({'error': 'Stop the script before deleting it'}), 400
    config = load_config()
    config['custom_scripts'] = [s for s in config['custom_scripts'] if s['name'] != name]
    save_config(config)
    script_runs.pop(name, None)
    return jsonify({'success': True, 'message': 'Script profile deleted'})

@app.route('/api/scripts/<path:name>/start', methods=['POST'])
def start_script(name):
    if script_running(name):
        return jsonify({'error': f'{name} is already running'}), 400
    script = next((s for s in load_config()['custom_scripts'] if s['name'] == name), None)
    if script is None:
        return jsonify({'error': 'Script not found'}), 404

    script_path = os.path.expanduser(script.get('script_path', ''))
    if script_path and not os.path.isfile(script_path):
        return jsonify({'error': f'Script not found: {script_path}'}), 400
    workdir = os.path.expanduser(script.get('workdir', '')) \
        or (os.path.dirname(script_path) if script_path else str(user_home_path))
    if not os.path.isdir(workdir):
        return jsonify({'error': f'Working directory not found: {workdir}'}), 400

    env_type = script.get('env_type', 'none')
    env_name = os.path.expanduser(script.get('env', ''))
    conda = None
    if env_type == 'conda':
        conda = find_conda()
        if not conda:
            return jsonify({'error': 'conda executable not found (set CONDA_EXE or put conda on PATH)'}), 400
    elif env_type == 'venv' and not os.path.isfile(os.path.join(env_name, 'bin', 'activate')):
        return jsonify({'error': f'venv not found (no bin/activate in {env_name})'}), 400

    proc_env = os.environ.copy()
    proc_env['PYTHONUNBUFFERED'] = '1'  # stream Python servers' logs instead of block-buffering them
    if script_path:
        proc_env['SCRIPT'] = script_path

    try:
        proc = subprocess.Popen(
            ['bash', '-c', build_shell_command(script, conda)],
            cwd=workdir,
            env=proc_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            start_new_session=True  # own process group, so stop can kill its children too
        )
    except Exception as e:
        return jsonify({'error': f'Failed to start script: {str(e)}'}), 500

    run = {'proc': proc, 'ready': False, 'stopped': False,
           'ready_pattern': script.get('ready_pattern', ''), 'log': deque(maxlen=1000)}
    script_runs[name] = run
    env_label = f'{env_type}: {env_name}' if env_type != 'none' else 'system'
    script_log(name, run, f'▶ Starting {name} in {workdir} (env: {env_label})')
    print(f"Manager: Starting custom script '{name}' in {workdir} (env: {env_label})")

    output_thread = threading.Thread(target=read_script_output, args=(name, run))
    output_thread.daemon = True
    output_thread.start()
    return jsonify({'success': True, 'message': f'Starting {name}...'})

@app.route('/api/scripts/<path:name>/stop', methods=['POST'])
def stop_script(name):
    if not script_running(name):
        return jsonify({'error': f'{name} is not running'}), 400
    run = script_runs[name]
    run['stopped'] = True
    try:
        stop_process_group(run['proc'])
    except Exception as e:
        return jsonify({'error': f'Failed to stop script: {str(e)}'}), 500
    return jsonify({'success': True, 'message': f'{name} stopped'})

@app.route('/api/scripts/<path:name>/log', methods=['GET'])
def get_script_log(name):
    run = script_runs.get(name)
    return jsonify({'lines': list(run['log']) if run else []})

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
