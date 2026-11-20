import os
import json
import functools
import re
import shlex
import shutil
import stat
import subprocess
import signal
import sys
import threading
import time
import uuid


from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from pathlib import Path

user_home_path = Path.home()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
# Run records (see launch_run), persisted in RUNS_DIR so they survive a manager restart
server_runs = {}  # 'llama-server' / 'rpc-worker' -> run record
script_runs = {}  # custom script name -> run record
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')  # next to app.py, regardless of CWD
RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'script_runs')  # process logs + run state

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
        },
        # Stop llama-server, the RPC worker and custom scripts when the manager stops or is killed
        # (see reaper_main). Off: they keep running and are re-adopted on the next start.
        "stop_processes_with_manager": True
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
    # Check if server is already running
    if server_running('llama-server'):
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
    
    # The binary runs under a bash wrapper (see launch_run), so a missing one wouldn't raise here
    if not shutil.which(server_path):
        return jsonify({'error': f'Failed to start server: llama-server not found at {server_path}'}), 500

    try:
        # Start the server process; it keeps running if the manager restarts
        launch_run('llama-server', 'llama-server', cmd, description='llama-server', port=LLAMA_SERVER_PORT)

        #return jsonify({'success': True, 'message': 'Loading model...'})
        return jsonify({'success': True, 'message': f"Manager: Loading model with parameters: n_gpu_layers {str(n_gpu_layers)}, n_ctx: {str(n_ctx)}, n_predict: {str(n_predict)} ..."})

    except Exception as e:
        return jsonify({'error': f'Failed to start server: {str(e)}'}), 500

@app.route('/api/unload', methods=['POST'])
def unload_model():
    if server_running('llama-server'):
        run = server_runs['llama-server']
        run['stopped'] = True
        save_runs()
        try:
            message = 'Model force killed' if stop_process_group(run, timeout=5) else 'Model unloaded successfully'
            socketio.emit('model_unloaded', {'success': True, 'message': message})
            return jsonify({'success': True, 'message': message})
        except Exception as e:
            return jsonify({'error': f'Failed to unload model: {str(e)}'}), 500
    else:
        return jsonify({'error': 'No model is currently loaded'}), 400

@app.route('/api/status', methods=['GET'])
def get_status():
    is_running = server_running('llama-server')
    run = server_runs.get('llama-server')
    config = load_config()
    settings = config.get('settings', get_default_settings())
    return jsonify({
        'is_loaded': is_running,
        'is_ready': bool(is_running and run['ready']),  # past "server is listening" (False while loading)
        'llama_server_path': settings.get('llama_server_path', LLAMA_SERVER_PATH),
        'llama_server_host': settings.get('llama_server_host', LLAMA_SERVER_HOST),
        # While running, report the port llama-server was actually launched with
        'llama_server_port': run['port'] if is_running else settings.get('llama_server_port', LLAMA_SERVER_PORT)
    })

@app.route('/api/log', methods=['GET'])
def get_log():
    """Recent llama-server output (current run, or the last one) for the dashboard console"""
    run = server_runs.get('llama-server')
    return jsonify(log_history(run) if run else {'lines': [], 'run': None, 'end': 0})

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
    if server_running('rpc-worker'):
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

    # The binary runs under a bash wrapper (see launch_run), so a missing one wouldn't raise here
    if not shutil.which(server_path):
        return jsonify({'error': f'Failed to start RPC worker: rpc-server not found at {server_path}'}), 500

    try:
        # Keeps running if the manager restarts
        launch_run('rpc-worker', 'rpc-worker', cmd, env=worker_env, description=f'RPC worker on {host}:{port}')
        return jsonify({'success': True, 'message': f'Starting RPC worker on {host}:{port} ...'})
    except Exception as e:
        return jsonify({'error': f'Failed to start RPC worker: {str(e)}'}), 500

@app.route('/api/rpc/stop_worker', methods=['POST'])
def stop_rpc_worker():
    if server_running('rpc-worker'):
        run = server_runs['rpc-worker']
        run['stopped'] = True
        save_runs()
        try:
            message = 'RPC worker force killed' if stop_process_group(run, timeout=5) else 'RPC worker stopped'
            socketio.emit('rpc_worker_stopped', {'message': message})
            return jsonify({'success': True, 'message': message})
        except Exception as e:
            return jsonify({'error': f'Failed to stop RPC worker: {str(e)}'}), 500
    else:
        return jsonify({'error': 'No RPC worker is currently running'}), 400

@app.route('/api/rpc/status', methods=['GET'])
def rpc_status():
    return jsonify({
        'rpc_worker_running': server_running('rpc-worker'),
        'server_running': server_running('llama-server')
    })

@app.route('/api/rpc/log', methods=['GET'])
def get_rpc_log():
    """Recent RPC worker and llama-server output (current or last runs), oldest run first,
    for the RPC page console"""
    runs = sorted((r for r in (server_runs.get('rpc-worker'), server_runs.get('llama-server')) if r),
                  key=lambda r: r.get('started_at', 0))
    return jsonify({'logs': [{'kind': r['kind'], **log_history(r)} for r in runs]})

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

# ---------------------------------------------------------------------------
# Process runs (llama-server, the RPC worker and custom scripts) outlive the manager: output goes
# to a log file (not a pipe that dies with the manager), RUN_WRAPPER writes the exit code to a
# file, and run records are saved in RUNS_DIR so a restarted manager adopts what's still running.
# ---------------------------------------------------------------------------

RUN_WRAPPER = 'exit_file=$1; shift; "$@"; code=$?; echo "$code" > "$exit_file"; exit "$code"'
LINE_BREAK = re.compile(rb'\r\n|\r|\n')  # \r too, so progress bars show up as they update
RUN_FILES = (('servers.json', server_runs), ('runs.json', script_runs))
runs_lock = threading.Lock()

def save_runs():
    """Persist run records (everything but the Popen handle)"""
    with runs_lock:
        os.makedirs(RUNS_DIR, exist_ok=True)
        for filename, runs in RUN_FILES:
            data = {key: {k: v for k, v in run.items() if k != 'proc'} for key, run in list(runs.items())}
            path = os.path.join(RUNS_DIR, filename)
            with open(path + '.tmp', 'w') as f:
                json.dump(data, f, indent=2)
            os.replace(path + '.tmp', path)

def proc_identity(pid):
    """[start time, boot id] of a live process, so a recycled PID is never mistaken for ours"""
    try:
        with open(f'/proc/{pid}/stat') as f:
            fields = f.read().rsplit(')', 1)[1].split()  # fields after "pid (comm)"
        with open('/proc/sys/kernel/random/boot_id') as f:
            boot_id = f.read().strip()
    except (OSError, IndexError):
        return None
    if fields[0] in ('Z', 'X'):  # exited (zombie / dead)
        return None
    return [fields[19], boot_id]

def run_alive(run):
    if run.get('proc') is not None:  # started by this manager process: it's our child
        return run['proc'].poll() is None
    return run['identity'] is not None and proc_identity(run['pid']) == run['identity']

def run_exit_code(run):
    try:
        with open(run['exit_file']) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        pass  # killed by a signal, so the wrapper never wrote it
    proc = run.get('proc')
    return proc.returncode if proc is not None else None

def runs_for(run):
    return script_runs if run['kind'] == 'script' else server_runs

def server_running(kind):
    """Whether the 'llama-server' or 'rpc-worker' run is alive"""
    run = server_runs.get(kind)
    return run is not None and run_alive(run)

def script_running(name):
    run = script_runs.get(name)
    return run is not None and run_alive(run)

def script_state(run):
    """One of: starting (waiting for ready pattern), ready, running, stopped, failed"""
    if run_alive(run):
        if run['ready']:
            return {'state': 'ready', 'exit_code': None}
        return {'state': 'starting' if run['ready_pattern'] else 'running', 'exit_code': None}
    code = run_exit_code(run)
    return {'state': 'stopped' if code == 0 or run_stopped(run) else 'failed', 'exit_code': code}

def run_id(run):
    return os.path.splitext(os.path.basename(run['log']))[0]

def last_line_end(data):
    """Offset just past the last line break in data (0 if none)"""
    return max(data.rfind(b'\n'), data.rfind(b'\r')) + 1

def log_history(run, max_lines=1000, max_bytes=256 * 1024):
    """The last complete lines of a run's log (reading only its end), plus the run id and the byte
    offset just past those lines. Live lines carry the same offsets, so a page can skip repeats."""
    try:
        with open(run['log'], 'rb') as f:
            size = f.seek(0, os.SEEK_END)
            start = max(0, size - max_bytes)
            f.seek(start)
            data = f.read()
    except OSError:
        return {'lines': [], 'run': run_id(run), 'end': 0}
    cut = last_line_end(data)  # a trailing partial line comes later as a live line
    lines = LINE_BREAK.split(data[:cut])[:-1]
    if start > 0:
        lines = lines[1:]  # first line is probably cut off
    return {'lines': [line.decode('utf-8', 'replace') for line in lines[-max_lines:]],
            'run': run_id(run), 'end': start + cut}

def is_ready_line(run, line):
    """Whether a log line shows the process is up"""
    if run['kind'] == 'llama-server':
        return 'main: server is listening on' in line
    if run['kind'] == 'rpc-worker':  # rpc-server prints one of these on a successful bind/listen
        return 'Starting RPC server' in line or 'Accepting' in line or 'listening' in line.lower()
    return bool(run['ready_pattern']) and run['ready_pattern'] in line

def emit_line(run, line, pos=None):
    """pos: byte offset in the log just past this line (None for messages not from the log)"""
    data = {'data': line, 'run': run_id(run), 'pos': pos}
    if run['kind'] == 'script':
        socketio.emit('script_output', {'name': run['name'], **data})
    else:
        socketio.emit('console_output', data)

def mark_ready(run):
    run['ready'] = True
    save_runs()
    if run['kind'] == 'llama-server':
        socketio.emit('loading_complete', {'success': True, 'message': 'Model loaded successfully!'})
    elif run['kind'] == 'rpc-worker':
        socketio.emit('rpc_worker_started', {'success': True, 'message': 'RPC worker is running'})
    else:
        socketio.emit('script_status', {'name': run['name'], 'state': 'ready', 'message': f"{run['name']} is ready"})

def finish_run(run):
    """Record that a run ended: reap it and tell the pages (once per run)"""
    if run.get('proc') is not None:
        run['proc'].wait()
    if runs_for(run).get(run['name']) is not run:
        return  # replaced by a newer run, or the profile was deleted, while this one wound down
    code = run_exit_code(run)
    name = run['name']
    if run['kind'] == 'script':
        if run_stopped(run):
            message = f'{name} stopped'
        elif code is None:
            message = f'{name} exited (exit code unknown)'
        else:
            message = f'{name} exited with code {code}'
        try:
            with open(run['log'], 'a') as f:
                f.write(message + '\n')
        except OSError:
            pass
    run['finished'] = True
    save_runs()

    if run['kind'] == 'llama-server':
        if not run['ready']:  # output ended before "server is listening": loading failed
            socketio.emit('loading_complete', {'success': False, 'message': f'llama-server exited before it was ready (exit code {code})'})
    elif run['kind'] == 'rpc-worker':
        socketio.emit('rpc_worker_stopped', {'message': 'RPC worker process ended'})
    else:
        socketio.emit('script_status', {'name': name, **script_state(run), 'message': message})

def handle_log_line(run, raw, pos):
    line = raw.decode('utf-8', 'replace').rstrip()
    emit_line(run, line, pos)
    if not run['ready'] and is_ready_line(run, line):
        mark_ready(run)

def follow_log(run, offset=0):
    """Stream new log lines to the pages and watch for readiness until the process ends"""
    partial = b''
    partial_start = offset  # file offset where `partial` begins
    process_gone = False
    try:
        with open(run['log'], 'rb') as f:
            f.seek(offset)
            while True:
                data = f.read(65536)
                if data:
                    buf, line_start = partial + data, 0
                    for brk in LINE_BREAK.finditer(buf):
                        handle_log_line(run, buf[line_start:brk.start()], partial_start + brk.end())
                        line_start = brk.end()
                    partial, partial_start = buf[line_start:], partial_start + line_start
                elif process_gone:
                    break
                elif not run_alive(run):
                    process_gone = True  # one more read for anything it wrote last
                else:
                    time.sleep(0.25)
        if partial:
            handle_log_line(run, partial, partial_start + len(partial))
    except Exception as e:
        emit_line(run, f'Error reading output: {str(e)}')
        while run_alive(run):
            time.sleep(1)
    finish_run(run)

def start_follower(run, offset=0):
    thread = threading.Thread(target=follow_log, args=(run, offset))
    thread.daemon = True
    thread.start()

def stop_marker(run):
    """File the reaper creates for runs it stops, so they show as stopped rather than failed"""
    return os.path.splitext(run['log'])[0] + '.stopped'

def run_stopped(run):
    """Stopped on purpose: from the UI, or by the reaper when the manager stopped"""
    return run['stopped'] or os.path.exists(stop_marker(run))

def discard_run_files(run):
    for path in (run.get('log'), run.get('exit_file'), run.get('log') and stop_marker(run)):
        if path:
            try:
                os.remove(path)
            except OSError:
                pass

@functools.lru_cache(maxsize=None)
def systemd_scope_available():
    """Whether processes can run in their own systemd scope. Without one, a process started by the
    systemd service lives in the service's cgroup, and restarting the service kills it."""
    if not shutil.which('systemd-run'):
        return False
    try:
        return subprocess.run(['systemd-run', '--user', '--scope', '--quiet', '--collect', 'true'],
                              capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False

def launch_run(kind, name, argv, env=None, cwd=None, first_line=None, description='', **fields):
    """Start argv so it outlives the manager, record it under server_runs[kind] or
    script_runs[name], and follow its log. Raises if it can't be started."""
    run_id = uuid.uuid4().hex[:12]
    log_path = os.path.join(RUNS_DIR, f'{run_id}.log')
    exit_path = os.path.join(RUNS_DIR, f'{run_id}.exit')
    cmd = ['bash', '-c', RUN_WRAPPER, 'llm-run', exit_path] + list(argv)
    if systemd_scope_available():
        # Own systemd scope, outside the manager service's cgroup, so restarting the service doesn't
        # kill it. The unit name lets uninstall-service.sh find these.
        cmd = ['systemd-run', '--user', '--scope', '--quiet', '--collect', f'--unit=llama-server-manager-{run_id}',
               f'--description=Llama-server Manager: {description}'] + cmd

    os.makedirs(RUNS_DIR, exist_ok=True)
    try:
        with open(log_path, 'w') as log:
            if first_line:
                log.write(first_line + '\n')
        with open(log_path, 'a') as log:
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,  # a file, not a pipe, so the process survives a manager restart
                stderr=subprocess.STDOUT,
                start_new_session=True  # own process group, so stopping kills its children too
            )
    except Exception:
        discard_run_files({'log': log_path})
        raise

    runs = script_runs if kind == 'script' else server_runs
    previous = runs.get(name)
    if previous:
        discard_run_files(previous)
    run = {'kind': kind, 'name': name, 'pid': proc.pid, 'identity': proc_identity(proc.pid),
           'started_at': time.time(), 'log': log_path, 'exit_file': exit_path, 'ready_pattern': '', 'ready': False,
           'stopped': False, 'finished': False, **fields, 'proc': proc}
    runs[name] = run
    save_runs()
    start_follower(run)
    return run

def adopt_runs():
    """On startup, pick up processes started before the manager restarted"""
    for filename, runs in RUN_FILES:  # load everything first: finish_run below saves all files
        try:
            with open(os.path.join(RUNS_DIR, filename)) as f:
                saved = json.load(f)
        except (OSError, ValueError):
            continue
        for key, run in saved.items():
            run['kind'] = 'script' if runs is script_runs else key
            run['name'] = key
            runs[key] = run

    for runs in (server_runs, script_runs):
        for run in list(runs.values()):
            if run.get('finished'):
                continue
            if not run_alive(run):
                finish_run(run)  # ended while the manager was down
                continue
            try:
                with open(run['log'], 'rb') as f:
                    data = f.read()
            except OSError:
                data = b''
            if not run['ready'] and any(is_ready_line(run, line.decode('utf-8', 'replace'))
                                        for line in LINE_BREAK.split(data)):
                run['ready'] = True  # became ready while the manager was down
            start_follower(run, offset=last_line_end(data))
            label = f"custom script '{run['name']}'" if run['kind'] == 'script' else run['kind']
            print(f"Manager: Re-attached to {label} (pid {run['pid']})")
    save_runs()

def stop_process_group(run, timeout=10):
    """SIGTERM the run's whole process group (e.g. bash plus vLLM workers), SIGKILL after timeout.
    Returns True if it had to SIGKILL."""
    pgid = run['pid']  # start_new_session=True makes the wrapper bash the group leader
    proc = run.get('proc')
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return False
    forced = False
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None:
            proc.poll()  # reap our child so it doesn't linger as a zombie in the group
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            break  # every process in the group is gone
        time.sleep(0.2)
    else:
        forced = True
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if proc is not None:
        proc.wait()
    return forced

# ---------------------------------------------------------------------------
# Stopping processes when the manager stops. The manager can't run cleanup code when it's killed,
# so a small helper process (the "reaper") does it: it holds the read end of a pipe whose write end
# only the outermost manager process has, and the pipe closes however that process ends.
# ---------------------------------------------------------------------------

reaper_pipe = None  # write end; stays open for the manager's lifetime

def reaper_main():
    """`app.py --reaper`: wait for the manager to end (stdin closes), then, if the setting is on,
    stop every process it started"""
    class ManagerGone(Exception):
        pass

    def on_sigterm(*_):
        raise ManagerGone  # being stopped ourselves (e.g. by uninstall-service.sh) counts too

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, on_sigterm)
    try:
        sys.stdin.buffer.read()  # returns at EOF: the manager is gone
    except ManagerGone:
        pass
    signal.signal(signal.SIGTERM, signal.SIG_IGN)  # don't get cut off halfway through

    if not load_config().get('settings', {}).get('stop_processes_with_manager', True):
        return
    runs = []
    for filename, _ in RUN_FILES:
        try:
            with open(os.path.join(RUNS_DIR, filename)) as f:
                runs.extend(json.load(f).values())
        except (OSError, ValueError):
            pass
    live = [run for run in runs if not run.get('finished') and run_alive(run)]  # identity-checked
    for run in live:
        open(stop_marker(run), 'w').close()
    threads = [threading.Thread(target=stop_process_group, args=(run,)) for run in live]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if live:
        print(f"Manager stopped: stopped {', '.join(run['name'] for run in live)}", flush=True)

def start_reaper():
    global reaper_pipe
    read_end, reaper_pipe = os.pipe()  # both non-inheritable: no child gets the write end
    cmd = [sys.executable, os.path.abspath(__file__), '--reaper']
    if systemd_scope_available():
        # Own scope, so tearing down the service's cgroup (even with SIGKILL) doesn't take it along
        cmd = ['systemd-run', '--user', '--scope', '--quiet', '--collect',
               f'--unit=llama-server-manager-reaper-{uuid.uuid4().hex[:12]}',
               '--description=Llama-server Manager: stop processes when the manager stops'] + cmd
    subprocess.Popen(cmd, stdin=read_end, start_new_session=True)
    os.close(read_end)

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
        run = script_runs.pop(name)  # keep its last log under the new name
        run['name'] = script['name']
        script_runs[script['name']] = run
        save_runs()
    return jsonify({'success': True, 'message': 'Script profile updated', 'name': script['name']})

@app.route('/api/scripts/<path:name>', methods=['DELETE'])
def delete_script(name):
    if script_running(name):
        return jsonify({'error': 'Stop the script before deleting it'}), 400
    config = load_config()
    config['custom_scripts'] = [s for s in config['custom_scripts'] if s['name'] != name]
    save_config(config)
    run = script_runs.pop(name, None)
    if run:
        discard_run_files(run)
        save_runs()
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

    env_label = f'{env_type}: {env_name}' if env_type != 'none' else 'system'
    try:
        launch_run('script', name, ['bash', '-c', build_shell_command(script, conda)],
                   env=proc_env, cwd=workdir, first_line=f'▶ Starting {name} in {workdir} (env: {env_label})',
                   description=f'script {name}', ready_pattern=script.get('ready_pattern', ''))
    except Exception as e:
        return jsonify({'error': f'Failed to start script: {str(e)}'}), 500
    print(f"Manager: Starting custom script '{name}' in {workdir} (env: {env_label})")
    return jsonify({'success': True, 'message': f'Starting {name}...'})

@app.route('/api/scripts/<path:name>/stop', methods=['POST'])
def stop_script(name):
    if not script_running(name):
        return jsonify({'error': f'{name} is not running'}), 400
    run = script_runs[name]
    run['stopped'] = True
    save_runs()
    try:
        stop_process_group(run)
    except Exception as e:
        return jsonify({'error': f'Failed to stop script: {str(e)}'}), 500
    return jsonify({'success': True, 'message': f'{name} stopped'})

@app.route('/api/scripts/<path:name>/log', methods=['GET'])
def get_script_log(name):
    run = script_runs.get(name)
    return jsonify(log_history(run) if run else {'lines': [], 'run': None, 'end': 0})

if __name__ == '__main__':
    if '--reaper' in sys.argv[1:]:
        reaper_main()
        sys.exit(0)

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

    debug = True
    # Re-adopt llama-server / RPC worker / scripts still running from before a restart. With the debug
    # reloader this block also runs in the file-watcher process; only the serving one adopts.
    if not debug or os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        adopt_runs()
    # The reaper belongs to the outermost process (the reloader's file watcher, or the only process
    # without it): reloads after app.py changes don't end that process, so they don't stop anything.
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        start_reaper()

    socketio.run(app, host='0.0.0.0', port=5000, debug=debug, allow_unsafe_werkzeug=True)
