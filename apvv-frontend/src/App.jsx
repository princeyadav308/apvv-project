import React, { useState, useEffect, useRef } from 'react';
import { Camera, AlertTriangle, Flame, Shield, UserX, Eye, Box, Power, Play, Pause, RefreshCw, Clock } from 'lucide-react';

const CVMonitoringApp = () => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedCamera, setSelectedCamera] = useState(0);
  const [alerts, setAlerts] = useState([]);
  const [currentFrame, setCurrentFrame] = useState(null);
  const [detections, setDetections] = useState([]);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const wsRef = useRef(null);
  const canvasRef = useRef(null);
  const imgRef = useRef(null);
  const animationFrameRef = useRef(null);
  const lastUpdateRef = useRef(0);
  const throttleRef = useRef(0);

  const cameras = [
    { id: 0, name: 'Webcam', location: 'Local Camera' },
    { id: 1, name: 'External Camera', location: 'USB Camera' }
  ];

  // Connect to WebSocket
  const connectWebSocket = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    setConnectionStatus('connecting');
    const ws = new WebSocket('ws://localhost:8000/ws/video');

    ws.onopen = () => {
      console.log('WebSocket connected');
      setConnectionStatus('connected');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        // Update frame
        setCurrentFrame(`data:image/jpeg;base64,${data.frame}`);
        
        // Update detections
        if (data.detections && data.detections.length > 0) {
          setDetections(data.detections);
          
          // Create alerts for violations
          data.detections.forEach(det => {
            if (!det.compliant) {
              const newAlert = {
                id: Date.now() + Math.random(),
                type: 'ppe',
                name: 'Safety Gear Violation',
                severity: 'high',
                timestamp: new Date().toLocaleTimeString(),
                camera: selectedCamera,
                confidence: det.person_confidence,
                details: {
                  helmet: det.has_helmet,
                  vest: det.has_vest
                },
                bbox: det.bbox
              };
              
              setAlerts(prev => {
                // Avoid duplicates
                const exists = prev.some(a => 
                  Math.abs(a.bbox.x - det.bbox.x) < 20 && 
                  Date.now() - new Date(a.timestamp).getTime() < 5000
                );
                if (exists) return prev;
                return [newAlert, ...prev.slice(0, 19)];
              });
            }
          });
        }
      } catch (error) {
        console.error('Error parsing message:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setConnectionStatus('error');
    };
    
    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setConnectionStatus('disconnected');
      wsRef.current = null;
    };

    wsRef.current = ws;
  };

  // Start camera
  const startCamera = async () => {
    try {
      const response = await fetch('http://localhost:8000/camera/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camera_id: selectedCamera })
      });
      
      const data = await response.json();
      
      if (data.success) {
        setIsProcessing(true);
        connectWebSocket();
      } else {
        alert('Failed to start camera: ' + data.error);
      }
    } catch (error) {
      console.error('Error starting camera:', error);
      alert('Cannot connect to backend. Make sure Python server is running on port 8000');
    }
  };

  // Stop camera
  const stopCamera = async () => {
    try {
      await fetch('http://localhost:8000/camera/stop', {
        method: 'POST'
      });
      
      setIsProcessing(false);
      if (wsRef.current) {
        wsRef.current.close();
      }
    } catch (error) {
      console.error('Error stopping camera:', error);
    }
  };

  // Toggle processing
  const toggleProcessing = () => {
    if (isProcessing) {
      stopCamera();
    } else {
      startCamera();
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  // Draw detections on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;

    if (!canvas || !img || !currentFrame) return;

    const ctx = canvas.getContext('2d');

    // Cancel previous animation frame
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }

    const drawCanvas = () => {
      // Wait for image to load
      if (img.complete && img.naturalWidth > 0) {
        // Set canvas size to match image dimensions
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw detections
        detections.forEach((det) => {
          const { x, y, width, height } = det.bbox;

          // Choose color based on compliance
          const color = det.compliant ? '#22c55e' : '#ef4444';

          // Draw bounding box
          ctx.strokeStyle = color;
          ctx.lineWidth = 3;
          ctx.strokeRect(x, y, width, height);

          // Draw label background
          const label = det.compliant ? 'COMPLIANT' : 'VIOLATION';
          ctx.fillStyle = color;
          ctx.fillRect(x, y - 25, 120, 25);

          // Draw label text
          ctx.fillStyle = 'white';
          ctx.font = 'bold 12px Arial';
          ctx.fillText(label, x + 5, y - 8);

          // Draw PPE status
          ctx.font = '10px Arial';
          const helmetText = `Helmet: ${det.has_helmet ? 'YES' : 'NO'}`;
          const vestText = `Vest: ${det.has_vest ? 'YES' : 'NO'}`;

          ctx.fillStyle = det.has_helmet ? '#22c55e' : '#ef4444';
          ctx.fillText(helmetText, x, y + 15);

          ctx.fillStyle = det.has_vest ? '#22c55e' : '#ef4444';
          ctx.fillText(vestText, x, y + 30);
        });
      }

      // Schedule next frame
      animationFrameRef.current = requestAnimationFrame(drawCanvas);
    };

    // Start drawing loop
    animationFrameRef.current = requestAnimationFrame(drawCanvas);
  }, [currentFrame, detections]);

  const getSeverityColor = (severity) => {
    const colors = {
      low: 'bg-blue-500',
      medium: 'bg-yellow-500',
      high: 'bg-orange-500',
      critical: 'bg-red-500'
    };
    return colors[severity] || 'bg-gray-500';
  };

  const getConnectionStatusColor = () => {
    switch (connectionStatus) {
      case 'connected': return 'text-green-400';
      case 'connecting': return 'text-yellow-400';
      case 'error': return 'text-red-400';
      default: return 'text-slate-400';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white p-4">
      <div className="max-w-7xl mx-auto">
        <div className="mb-6">
          <h1 className="text-3xl font-bold mb-2">APVV - Live PPE Detection System</h1>
          <p className="text-blue-200">Real-time Safety Gear Monitoring with AI</p>
        </div>

        {/* Connection Status Banner */}
        {connectionStatus === 'error' && (
          <div className="mb-4 bg-red-900/30 border border-red-700 rounded-lg p-4">
            <p className="text-red-300">
              ⚠️ Cannot connect to backend. Make sure Python server is running:
              <code className="ml-2 bg-slate-800 px-2 py-1 rounded">python ppe_detection_backend.py</code>
            </p>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="lg:col-span-2 space-y-4">
            {/* Camera Controls */}
            <div className="bg-slate-800/50 backdrop-blur rounded-lg p-4 border border-slate-700">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <Camera className="w-5 h-5 text-blue-400" />
                  <select
                    value={selectedCamera}
                    onChange={(e) => setSelectedCamera(Number(e.target.value))}
                    disabled={isProcessing}
                    className="bg-slate-700 text-white px-4 py-2 rounded-lg border border-slate-600 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
                  >
                    {cameras.map(cam => (
                      <option key={cam.id} value={cam.id}>
                        {cam.name} - {cam.location}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="flex gap-2">
                  <button
                    onClick={toggleProcessing}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg font-semibold transition-all ${
                      isProcessing 
                        ? 'bg-red-500 hover:bg-red-600' 
                        : 'bg-green-500 hover:bg-green-600'
                    }`}
                  >
                    {isProcessing ? (
                      <>
                        <Pause className="w-4 h-4" />
                        Stop Detection
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4" />
                        Start Detection
                      </>
                    )}
                  </button>
                  
                  <button
                    onClick={() => {
                      setDetections([]);
                      setAlerts([]);
                    }}
                    className="flex items-center gap-2 px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg font-semibold transition-all"
                  >
                    <RefreshCw className="w-4 h-4" />
                  </button>
                </div>
              </div>

              <div className="flex items-center gap-4 text-sm">
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${isProcessing ? 'bg-green-400 animate-pulse' : 'bg-slate-600'}`}></div>
                  <span className="text-slate-400">
                    Camera: <span className="text-white font-semibold">{isProcessing ? 'Active' : 'Stopped'}</span>
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${connectionStatus === 'connected' ? 'bg-green-400' : 'bg-slate-600'}`}></div>
                  <span className="text-slate-400">
                    Backend: <span className={`font-semibold ${getConnectionStatusColor()}`}>
                      {connectionStatus.charAt(0).toUpperCase() + connectionStatus.slice(1)}
                    </span>
                  </span>
                </div>
              </div>
            </div>

            {/* Video Display */}
            <div className="bg-slate-800/50 backdrop-blur rounded-lg border border-slate-700 overflow-hidden">
              <div className="relative bg-slate-900" style={{ height: '480px' }}>
                {currentFrame ? (
                  <>
                    <img
                      ref={imgRef}
                      src={currentFrame}
                      alt="Camera feed"
                      className="w-full h-full object-contain"
                      style={{ maxWidth: '100%', maxHeight: '100%' }}
                    />
                    <canvas
                      ref={canvasRef}
                      className="absolute inset-0 w-full h-full pointer-events-none"
                      style={{ width: '100%', height: '100%' }}
                    />
                  </>
                ) : (
                  <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-slate-800 to-slate-900">
                    <div className="text-center">
                      <Camera className="w-16 h-16 text-slate-600 mx-auto mb-4" />
                      <p className="text-slate-400 font-semibold">
                        {isProcessing ? 'Loading camera feed...' : 'Click "Start Detection" to begin'}
                      </p>
                      <p className="text-sm text-slate-500 mt-2">
                        {cameras.find(c => c.id === selectedCamera)?.name}
                      </p>
                    </div>
                  </div>
                )}

                {isProcessing && (
                  <div className="absolute top-4 left-4 bg-red-500/90 px-3 py-1 rounded-full text-xs font-bold flex items-center gap-2">
                    <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
                    LIVE - PPE DETECTION
                  </div>
                )}

                <div className="absolute bottom-4 left-4 bg-slate-900/80 px-3 py-2 rounded-lg">
                  <div className="flex items-center gap-3 text-xs">
                    <div className="flex items-center gap-2">
                      <Shield className="w-3 h-3 text-green-400" />
                      <span>Helmet Detection</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Shield className="w-3 h-3 text-yellow-400" />
                      <span>Vest Detection</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-slate-900/50 p-3 border-t border-slate-700">
                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-4">
                    <span className="text-slate-400">
                      Detections: <span className="text-white font-semibold">{detections.length}</span>
                    </span>
                    <span className="text-slate-400">
                      Violations: <span className="text-red-400 font-semibold">
                        {detections.filter(d => !d.compliant).length}
                      </span>
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Detection Info */}
            <div className="bg-slate-800/50 backdrop-blur rounded-lg p-4 border border-slate-700">
              <h3 className="font-bold mb-3 flex items-center gap-2">
                <Shield className="w-5 h-5 text-blue-400" />
                PPE Detection Model Status
              </h3>
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-slate-900/50 p-3 rounded-lg">
                  <div className="flex items-center gap-2 mb-1">
                    <Shield className="w-4 h-4 text-yellow-500" />
                    <span className="text-sm font-semibold">Helmet Detection</span>
                  </div>
                  <div className="text-xs text-green-400">Active - YOLOv8</div>
                </div>
                <div className="bg-slate-900/50 p-3 rounded-lg">
                  <div className="flex items-center gap-2 mb-1">
                    <Shield className="w-4 h-4 text-orange-500" />
                    <span className="text-sm font-semibold">Vest Detection</span>
                  </div>
                  <div className="text-xs text-green-400">Active - Color Analysis</div>
                </div>
              </div>
            </div>
          </div>

          {/* Alerts Panel */}
          <div className="space-y-4">
            <div className="bg-slate-800/50 backdrop-blur rounded-lg p-4 border border-slate-700">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-bold flex items-center gap-2">
                  <AlertTriangle className="w-5 h-5 text-orange-400" />
                  PPE Violations
                </h2>
                <span className="text-xs bg-red-500 px-2 py-1 rounded-full font-bold">
                  {alerts.length}
                </span>
              </div>

              <div className="space-y-2 max-h-96 overflow-y-auto">
                {alerts.length === 0 ? (
                  <div className="text-center py-8 text-slate-400">
                    <Shield className="w-12 h-12 mx-auto mb-2 opacity-50" />
                    <p className="text-sm">No violations detected</p>
                    <p className="text-xs mt-1">All personnel compliant</p>
                  </div>
                ) : (
                  alerts.map(alert => (
                    <div
                      key={alert.id}
                      className="bg-slate-900/50 p-3 rounded-lg border border-red-700/50 hover:border-red-600 transition-all"
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <Shield className="w-4 h-4 text-red-400" />
                          <span className="font-semibold text-sm">{alert.name}</span>
                        </div>
                        <span className={`${getSeverityColor(alert.severity)} px-2 py-0.5 rounded text-xs font-bold uppercase`}>
                          {alert.severity}
                        </span>
                      </div>
                      
                      <div className="text-xs text-slate-400 space-y-1">
                        <div className="flex items-center gap-2">
                          <Clock className="w-3 h-3" />
                          <span>{alert.timestamp}</span>
                        </div>
                        <div className="flex items-center gap-4">
                          <span className={alert.details.helmet ? 'text-green-400' : 'text-red-400'}>
                            Helmet: {alert.details.helmet ? '✓' : '✗'}
                          </span>
                          <span className={alert.details.vest ? 'text-green-400' : 'text-red-400'}>
                            Vest: {alert.details.vest ? '✓' : '✗'}
                          </span>
                        </div>
                        <div>Confidence: <span className="text-green-400 font-semibold">{(alert.confidence * 100).toFixed(0)}%</span></div>
                      </div>
                      
                      <div className="mt-3 flex gap-2">
                        <button className="flex-1 bg-blue-500 hover:bg-blue-600 text-xs py-1.5 rounded font-semibold transition-all">
                          View Details
                        </button>
                        <button className="flex-1 bg-slate-700 hover:bg-slate-600 text-xs py-1.5 rounded font-semibold transition-all">
                          Acknowledge
                        </button>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>

            <div className="bg-slate-800/50 backdrop-blur rounded-lg p-4 border border-slate-700">
              <h3 className="font-bold mb-3">Detection Statistics</h3>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-slate-400">Total Detections</span>
                  <span className="font-bold text-blue-400">{detections.length}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-slate-400">Violations</span>
                  <span className="font-bold text-red-400">
                    {detections.filter(d => !d.compliant).length}
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-slate-400">Compliant</span>
                  <span className="font-bold text-green-400">
                    {detections.filter(d => d.compliant).length}
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-slate-400">Total Alerts</span>
                  <span className="font-bold text-orange-400">{alerts.length}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CVMonitoringApp;