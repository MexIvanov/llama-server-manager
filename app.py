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
CONFIG_FILE = 'config.json'

# Defaults — overridden by load_config settings
DEFAULT_LLAMA_SERVER_HOST = "0.0.0.0"
DEFAULT_LLAMA_SERVER_PORT = 10000
DEFAULT_LLAMA_SERVER_PATH = f'{user_home_path}/llama.cpp/build/bin/llama-server'

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
            return json.load(f)
    return {"models": [], "settings": get_default_settings()}

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
                
                # Check for error messages
                if "error" in line.lower() or "failed" in line.lower():
                    if not server_started:
                        socketio.emit('loading_complete', {'success': False, 'message': f'Error: {line}'})
    except Exception as e:
        socketio.emit('console_output', {'data': f'Error reading output: {str(e)}'})

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

@app.route('/api/models/<model_name>', methods=['PUT'])
def update_model(model_name):
    config = load_config()
    updated_model = request.json
    
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

@app.route('/api/models/<model_name>', methods=['DELETE'])
def delete_model(model_name):
    config = load_config()
    config['models'] = [m for m in config['models'] if m['model_name'] != model_name]
    save_config(config)
    return jsonify({'success': True, 'message': 'Model deleted successfully'})

@app.route('/api/load', methods=['POST'])
def load_model():
    global server_process
    
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
        'llama_server_port': settings.get('llama_server_port', LLAMA_SERVER_PORT)
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
