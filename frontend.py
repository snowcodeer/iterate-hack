"""Simple web frontend for UniWrap."""

import os
import sys
import subprocess
import json
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class UniWrapHandler(BaseHTTPRequestHandler):
    """HTTP handler for UniWrap frontend."""
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(self._get_index_html().encode())
        elif self.path == '/api/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'ready'}).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        """Handle POST requests."""
        if self.path == '/api/generate':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            url = data.get('url', '')
            evaluate = data.get('evaluate', False)
            
            try:
                # Run UniWrap
                result = self._run_uniwrap(url, evaluate)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def _run_uniwrap(self, url: str, evaluate: bool) -> dict:
        """Run UniWrap CLI command."""
        cmd = [sys.executable, '-m', 'uniwrap', url, '--format', 'both']
        if evaluate:
            cmd.append('--evaluate')
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            raise Exception(f"UniWrap failed: {result.stderr}")
        
        # Find generated files
        output_dir = Path('environments')
        env_files = list(output_dir.rglob('*.py'))
        json_files = list(output_dir.rglob('spec.json'))
        
        return {
            'success': True,
            'output': result.stdout,
            'env_files': [str(f) for f in env_files],
            'spec_files': [str(f) for f in json_files]
        }
    
    def _get_index_html(self) -> str:
        """Get HTML for frontend."""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UniWrap - RL Environment Generator</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 600px;
            width: 100%;
            padding: 40px;
        }
        h1 {
            color: #333;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .subtitle {
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            color: #333;
            font-weight: 500;
        }
        input[type="text"] {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        input[type="text"]:focus {
            outline: none;
            border-color: #667eea;
        }
        .checkbox-group {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        input[type="checkbox"] {
            width: 18px;
            height: 18px;
        }
        button {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
        }
        button:active {
            transform: translateY(0);
        }
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            background: #f5f5f5;
            border-radius: 8px;
            display: none;
        }
        .result.show {
            display: block;
        }
        .result.success {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        .result.error {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
        .loading {
            display: none;
            text-align: center;
            margin-top: 20px;
        }
        .loading.show {
            display: block;
        }
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .help-text {
            font-size: 0.9em;
            color: #666;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎮 UniWrap</h1>
        <p class="subtitle">Generate RL environments from any game</p>
        
        <form id="generateForm">
            <div class="form-group">
                <label for="url">Repository URL or Path</label>
                <input 
                    type="text" 
                    id="url" 
                    name="url" 
                    placeholder="https://github.com/user/repo or http://localhost:3000"
                    required
                >
                <p class="help-text">Enter a GitHub repo URL, web game URL, or local path</p>
            </div>
            
            <div class="form-group">
                <div class="checkbox-group">
                    <input type="checkbox" id="evaluate" name="evaluate">
                    <label for="evaluate" style="margin: 0;">Run evaluation agent</label>
                </div>
            </div>
            
            <button type="submit" id="submitBtn">Generate Environment</button>
        </form>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p style="margin-top: 10px;">Generating environment wrapper...</p>
        </div>
        
        <div class="result" id="result"></div>
    </div>
    
    <script>
        document.getElementById('generateForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const url = document.getElementById('url').value;
            const evaluate = document.getElementById('evaluate').checked;
            const submitBtn = document.getElementById('submitBtn');
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            
            submitBtn.disabled = true;
            loading.classList.add('show');
            result.classList.remove('show', 'success', 'error');
            
            try {
                const response = await fetch('/api/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ url, evaluate })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    result.className = 'result show success';
                    result.innerHTML = `
                        <h3>✓ Success!</h3>
                        <p>Environment generated successfully.</p>
                        <p><strong>Files created:</strong></p>
                        <ul>
                            ${data.env_files.map(f => `<li>${f}</li>`).join('')}
                        </ul>
                    `;
                } else {
                    result.className = 'result show error';
                    result.innerHTML = `<h3>✗ Error</h3><p>${data.error || 'Unknown error'}</p>`;
                }
            } catch (error) {
                result.className = 'result show error';
                result.innerHTML = `<h3>✗ Error</h3><p>${error.message}</p>`;
            } finally {
                submitBtn.disabled = false;
                loading.classList.remove('show');
            }
        });
    </script>
</body>
</html>"""


def main():
    """Run the frontend server."""
    port = int(os.getenv('PORT', 8080))
    server = HTTPServer(('localhost', port), UniWrapHandler)
    print(f"UniWrap frontend running at http://localhost:{port}")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == '__main__':
    main()

