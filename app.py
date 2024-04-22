from flask import Flask, render_template
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)

# Global flag to control the execution of the script
running_flag = False

@socketio.on('connect')
def handle_connect():
    global running_flag
    if not running_flag:
        start_script()
        running_flag = True

@socketio.on('disconnect')
def handle_disconnect():
    global running_flag
    if running_flag:
        stop_script()
        running_flag = False

def start_script():
    # Implement your script start logic here
    print("Script started")

def stop_script():
    # Implement your script stop logic here
    print("Script stopped")

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
