# Streamlit Web Portal for YOLO Tracking

Web-based interface for remote access to YOLO tracking with ephemeral filtering and local ID reconciliation.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_streamlit.txt
```

### 2. Run Locally
```bash
streamlit run streamlit_tracking_app.py
```
Access at: `http://localhost:8501`

### 3. Run on Remote Server (Accessible from Network)
```bash
streamlit run streamlit_tracking_app.py --server.address 0.0.0.0 --server.port 8501
```
Access from any device: `http://YOUR_SERVER_IP:8501`

## Features

✅ **Real-time Video Processing** - Live annotated video feed  
✅ **6 Interactive Plots** - All visualizations from original script  
✅ **Parameter Controls** - Adjust thresholds on-the-fly  
✅ **Live Statistics** - Real-time tracking metrics  
✅ **Remote Access** - Access from any browser  

## Usage

1. Enter model and video paths in sidebar
2. Adjust parameters (ephemeral threshold, IoU, etc.)
3. Click "▶️ Start Processing"
4. Watch real-time updates
5. Click "⏹️ Stop" to halt processing

## Configuration

Default paths (edit in sidebar):
- Model: `../yolo11l.pt`
- Video: `../videos/8th-grade-vid.mp4`

Parameters:
- **Ephemeral Threshold**: Min frames for persistent ID (default: 30)
- **Reevaluation Interval**: How often to re-filter (default: 10)
- **Temporal Gap**: Max frame gap for reconciliation (default: 35)
- **Min IoU**: Spatial overlap threshold (default: 0.3)

## Deployment Options

### Option 1: Local Network
```bash
streamlit run streamlit_tracking_app.py --server.address 0.0.0.0
```

### Option 2: Firewall (allow port)
```bash
sudo ufw allow 8501
```

### Option 3: SSH Tunnel (if firewall blocks)
On local machine:
```bash
ssh -L 8501:localhost:8501 user@remote_server
```
Then access: `http://localhost:8501`

## Notes

- Video processing happens on server (GPU/CPU)
- Browser only displays results (lightweight)
- Works on headless servers (no X11 needed)
- Press "Stop" to halt before video ends

