import asyncio
import websockets
import json
import os
import base64
from asyncio import Queue

PORT = 8069
UPLOADS_DIR = "files"  # Directory to store uploaded files

clients_queue = Queue()
active_client = None

# --- Helper Functions ---

def safe_delete(file_name):
    """Safely delete a file from the uploads directory."""
    # Basic sanitization to prevent directory traversal
    base_name = os.path.basename(file_name)
    if not base_name or base_name in ('.', '..'):
        print(f"Attempted to delete an invalid file name: {file_name}")
        return False

    file_path = os.path.join(UPLOADS_DIR, base_name)
    
    try:
        if os.path.exists(file_path) and os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Successfully deleted file: {file_path}")
            return True
        else:
            print(f"File not found for deletion: {file_path}")
            return False
    except Exception as e:
        print(f"Error deleting file {file_path}: {e}")
        return False

def save_file(file_name, file_content_b64):
    """Decode a base64 string and save it as a file."""
    try:
        # The data is a data URL: "data:<mime_type>;base64,<encoded_data>"
        # We need to split it to get just the base64 part.
        header, encoded = file_content_b64.split(",", 1)
        decoded_data = base64.b64decode(encoded)

        # Save the file to the uploads directory
        file_path = os.path.join(UPLOADS_DIR, file_name)
        with open(file_path, "wb") as f:
            f.write(decoded_data)
        print(f"Successfully saved file: {file_path}")
        return True
    except Exception as e:
        print(f"Error processing or saving file {file_name}: {e}")
        return False

def delete_all_uploads():
    """Delete all files in the uploads directory."""
    try:
        for file_name in os.listdir(UPLOADS_DIR):
            file_path = os.path.join(UPLOADS_DIR, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print("All files in uploads directory deleted.")
    except Exception as e:
        print(f"Error deleting all uploads: {e}")

# --- Main WebSocket Handler ---

async def handle_client(websocket):
    """Handle a single websocket client, with queue and timeout logic."""
    global active_client
    global clients_queue  # <-- Add this line
    print(f"Client connected: {websocket.remote_address}")
    await clients_queue.put(websocket)
    
    if active_client is not None or clients_queue.qsize() > 1:
        await websocket.send("__WAITING__")
        print(f"Client {websocket.remote_address} is waiting in queue.")

    try:
        while True:
            # Wait until this client is at the front of the queue
            if clients_queue.empty() or clients_queue._queue[0] is not websocket:
                await asyncio.sleep(1)
                continue

            active_client = websocket
            await websocket.send("__YOUR_TURN__")
            print(f"Client {websocket.remote_address} is now active.")

            while True:
                try:
                    # Wait for a message with a 10-minute timeout
                    raw_message = await asyncio.wait_for(websocket.recv(), timeout=600)
                    data = json.loads(raw_message)
                    msg_type = data.get("type")
                    msg_id = data.get("id") # For tracking requests

                    if msg_type == "file_upload":
                        file_info = data.get("file", {})
                        file_name = file_info.get("name")
                        file_data = file_info.get("data")
                        if file_name and file_data:
                            if save_file(file_name, file_data):
                                await websocket.send(f"__FILE_UPLOAD_SUCCESS__:{msg_id}")
                            else:
                                await websocket.send(f"__FILE_UPLOAD_FAIL__:{msg_id}")
                        continue # Don't process as a text message

                    elif msg_type == "file_delete":
                        file_name = data.get("fileName")
                        if file_name:
                            if safe_delete(file_name):
                                await websocket.send(f"__FILE_DELETE_SUCCESS__:{msg_id}")
                            else:
                                await websocket.send(f"__FILE_DELETE_FAIL__:{msg_id}")
                        continue # Don't process as a text message

                    elif msg_type == "text":
                        await websocket.send("__BUSY__")
                        message = data.get("message", "")
                        
                        # Simulate processing time
                        await asyncio.sleep(3)

                        # Echo back the text message as a response for demonstration
                        response = f"Received your message: '{message}'"
                        await websocket.send(response)

                        await websocket.send("lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.")
                        
                        # Send back some example markdown and math
                        await websocket.send("This is **bold**, _italic_, and a `code` snippet.")
                        await websocket.send("```python\n# Your text has been processed!\ndef hello():\n    print('Hello from the server!')\n``` \n\nAnd some math: $E = mc^2$")
                        
                        # Signal that the turn is over and the client can send again
                        await websocket.send("__YOUR_TURN__")

                except asyncio.TimeoutError:
                    print(f"Client {websocket.remote_address} timed out.")
                    await websocket.send("Disconnected due to inactivity (10 min timeout).")
                    await websocket.close()
                    break
                except websockets.ConnectionClosed:
                    print(f"Connection with {websocket.remote_address} closed.")
                    break
                except Exception as e:
                    print(f"Unexpected error: {e}")  # <-- Add this line
                    break
            break
    finally:
        # Clean up: remove client from the queue and reset active_client if it was this one
        if active_client is websocket:
            active_client = None
        
        if websocket in clients_queue._queue:
            new_queue = Queue()
            while not clients_queue.empty():
                item = await clients_queue.get()
                if item is not websocket:
                    await new_queue.put(item)
            clients_queue = new_queue

        # Delete all files in uploads when client disconnects
        delete_all_uploads()

        print(f"Client {websocket.remote_address} disconnected. Queue size: {clients_queue.qsize()}")


async def main():
    """Main entry point for the WebSocket server."""
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    print(f"Uploads directory '{UPLOADS_DIR}' is ready.")

    async with websockets.serve(handle_client, "0.0.0.0", PORT):
        print(f"WebSocket server running on wss://0.0.0.0:{PORT}")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
