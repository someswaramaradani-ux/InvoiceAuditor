import time
import os, sys
import uvicorn
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from work_flow import app

# Assume your LangGraph application is compiled and imported here
# from graph_flow import app, InvoiceProcessingState 

# Define the folder to watch
INCOMING_FOLDER = "data/incoming"
# Create the folder if it doesn't exist (important for Docker/testing)
os.makedirs(INCOMING_FOLDER, exist_ok=True) 

# --- LangGraph Execution Stub (Replace with your actual compiled app invocation) ---
def trigger_invoice_workflow(file_path: str):
    """
    Function to execute the compiled LangGraph workflow.
    
    In a real system, this would queue the task or run the graph.
    """
    file_name = os.path.basename(file_path)
    print(f"\n[TRIGGER] New file detected: {file_name}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting LangGraph workflow...")

    # 🚨 Actual LangGraph Invocation
    initial_state = app.invoke({
        "file_path": file_path, 
        "file_name": file_name, 
        "status": "RECEIVED", 
        "errors": []
    })
    
    # print(f"[COMPLETED] Workflow finished for {file_name}. Recommendation: {initial_state.get('recommendation')}")
    print(f"[COMPLETED] Workflow finished for {file_name}. (STUBBED)")


# --- Watchdog Event Handler ---
class InvoiceHandler(FileSystemEventHandler):
    """
    Custom handler to process file system events.
    """
    def on_created(self, event):
        """Called when a file or directory is created."""
        if not event.is_directory and not os.path.basename(event.src_path).startswith('.'):
            # Wait briefly to ensure the file is fully written before processing
            time.sleep(1) 
            trigger_invoice_workflow(event.src_path)

# --- Service Initialization ---
if __name__ == "__main__":
    event_handler = InvoiceHandler()
    observer = Observer()
    observer.schedule(event_handler, INCOMING_FOLDER, recursive=False)
    
    print(f"👀 AI Invoice Auditor Monitoring Service started.")
    print(f"   Watching folder: {INCOMING_FOLDER}")
    
    observer.start()
    
    # Keep the main thread alive, but allow the observer to exit on interrupt
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()