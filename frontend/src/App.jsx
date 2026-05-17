import { useState, useRef, useCallback, useEffect } from 'react';

function App() {
  const [isDrawing, setIsDrawing] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(0);
  const [confidenceScores, setConfidenceScores] = useState({});
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const canvasRef = useRef(null);
  const hiddenCanvasRef = useRef(null);
  const ctxRef = useRef(null);
  const lastPosRef = useRef({ x: 0, y: 0 });

  // Canvas dimensions
  const CANVAS_WIDTH = 400;
  const CANVAS_HEIGHT = 400;
  const MNIST_SIZE = 28;

  // Initialize canvases
  useEffect(() => {
    const canvas = canvasRef.current;
    const hiddenCanvas = hiddenCanvasRef.current;
    
    if (canvas && hiddenCanvas) {
      // Main drawing canvas
      canvas.width = CANVAS_WIDTH;
      canvas.height = CANVAS_HEIGHT;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 14;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctxRef.current = ctx;
      
      // Hidden canvas for processing
      hiddenCanvas.width = CANVAS_WIDTH;
      hiddenCanvas.height = CANVAS_HEIGHT;
      const hiddenCtx = hiddenCanvas.getContext('2d');
      hiddenCtx.fillStyle = '#000000';
      hiddenCtx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
      hiddenCtx.strokeStyle = '#ffffff';
      hiddenCtx.lineWidth = 14;
      hiddenCtx.lineCap = 'round';
      hiddenCtx.lineJoin = 'round';
    }
  }, []);

  // Get position from mouse or touch events
  const getPosition = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    if (e.touches && e.touches.length > 0) {
      return {
        x: (e.touches[0].clientX - rect.left) * scaleX,
        y: (e.touches[0].clientY - rect.top) * scaleY
      };
    }
    
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY
    };
  };

  // Start drawing
  const startDrawing = (e) => {
    e.preventDefault();
    setIsDrawing(true);
    const pos = getPosition(e);
    lastPosRef.current = pos;
    
    // Draw a dot at the starting position
    const ctx = ctxRef.current;
    if (ctx) {
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, ctx.lineWidth / 2, 0, Math.PI * 2);
      ctx.fill();
    }
    
    const hiddenCanvas = hiddenCanvasRef.current;
    if (hiddenCanvas) {
      const hiddenCtx = hiddenCanvas.getContext('2d');
      hiddenCtx.beginPath();
      hiddenCtx.arc(pos.x, pos.y, 7, 0, Math.PI * 2);
      hiddenCtx.fill();
    }
  };

  // Draw
  const draw = (e) => {
    if (!isDrawing) return;
    e.preventDefault();
    
    const pos = getPosition(e);
    const ctx = ctxRef.current;
    
    if (ctx) {
      ctx.beginPath();
      ctx.moveTo(lastPosRef.current.x, lastPosRef.current.y);
      ctx.lineTo(pos.x, pos.y);
      ctx.stroke();
    }
    
    const hiddenCanvas = hiddenCanvasRef.current;
    if (hiddenCanvas) {
      const hiddenCtx = hiddenCanvas.getContext('2d');
      hiddenCtx.beginPath();
      hiddenCtx.moveTo(lastPosRef.current.x, lastPosRef.current.y);
      hiddenCtx.lineTo(pos.x, pos.y);
      hiddenCtx.stroke();
    }
    
    lastPosRef.current = pos;
  };

  // Stop drawing and predict
  const stopDrawing = useCallback(async (e) => {
    if (!isDrawing) return;
    e.preventDefault();
    setIsDrawing(false);
    
    await predictDigit();
  }, [isDrawing]);

  // Clear canvas
  const clearCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const hiddenCanvas = hiddenCanvasRef.current;
    
    if (canvas && ctxRef.current) {
      ctxRef.current.fillStyle = '#0f172a';
      ctxRef.current.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    }
    
    if (hiddenCanvas) {
      const hiddenCtx = hiddenCanvas.getContext('2d');
      hiddenCtx.fillStyle = '#000000';
      hiddenCtx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    }
    
    setPrediction(null);
    setConfidence(0);
    setConfidenceScores({});
    setError(null);
  }, []);

  // Downsample and preprocess image to match MNIST format
  const preprocessImage = () => {
    const hiddenCanvas = hiddenCanvasRef.current;
    if (!hiddenCanvas) return null;
    
    const CANVAS_SIZE = CANVAS_WIDTH;
    
    // Get the canvas image data
    const ctx = hiddenCanvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    const pixels = imageData.data;
    
    // Find the bounding box of the drawn digit
    let minX = CANVAS_SIZE, maxX = 0, minY = CANVAS_SIZE, maxY = 0;
    let hasPixel = false;
    
    for (let y = 0; y < CANVAS_SIZE; y++) {
      for (let x = 0; x < CANVAS_SIZE; x++) {
        const idx = (y * CANVAS_SIZE + x) * 4;
        const brightness = pixels[idx]; // Red channel (grayscale)
        
        // Check if this pixel is drawn (white/light)
        if (brightness > 128) {
          hasPixel = true;
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }
    
    // If nothing drawn, return empty
    if (!hasPixel) {
      return Array(784).fill(0);
    }
    
    // Calculate bounding box dimensions
    const boxWidth = maxX - minX + 1;
    const boxHeight = maxY - minY + 1;
    
    // Create a temporary canvas for the cropped digit
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = boxWidth;
    tempCanvas.height = boxHeight;
    const tempCtx = tempCanvas.getContext('2d');
    
    // Draw the cropped region
    tempCtx.drawImage(
      hiddenCanvas,
      minX, minY, boxWidth, boxHeight,
      0, 0, boxWidth, boxHeight
    );
    
    // MNIST digits are centered in a 20x20 area within 28x28 image
    const targetSize = 20;
    const padding = (MNIST_SIZE - targetSize) / 2; // 4 pixels padding on each side
    
    // Create the final 28x28 canvas with proper centering
    const mnistCanvas = document.createElement('canvas');
    mnistCanvas.width = MNIST_SIZE;
    mnistCanvas.height = MNIST_SIZE;
    const mnistCtx = mnistCanvas.getContext('2d');
    
    // Fill with black background
    mnistCtx.fillStyle = '#000000';
    mnistCtx.fillRect(0, 0, MNIST_SIZE, MNIST_SIZE);
    
    // Calculate scale to fit the digit in 20x20 area
    const scale = targetSize / Math.max(boxWidth, boxHeight);
    const scaledWidth = Math.min(boxWidth * scale, targetSize);
    const scaledHeight = Math.min(boxHeight * scale, targetSize);
    
    // Calculate centered position
    const offsetX = padding + (targetSize - scaledWidth) / 2;
    const offsetY = padding + (targetSize - scaledHeight) / 2;
    
    // Draw the digit centered and scaled
    mnistCtx.drawImage(
      tempCanvas,
      0, 0, boxWidth, boxHeight,
      offsetX, offsetY, scaledWidth, scaledHeight
    );
    
    // Get the final pixel data
    const finalImageData = mnistCtx.getImageData(0, 0, MNIST_SIZE, MNIST_SIZE);
    const finalPixels = finalImageData.data;
    
    // Convert to grayscale and normalize (0-1 range)
    const grayscale = [];
    for (let i = 0; i < finalPixels.length; i += 4) {
      // Use red channel (grayscale)
      const value = finalPixels[i] / 255;
      grayscale.push(value);
    }
    
    return grayscale;
  };

  // Send prediction request
  const predictDigit = async () => {
    const pixels = preprocessImage();
    if (!pixels) {
      setError('Failed to process image');
      return;
    }
    
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/predict-digit', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ pixels }),
      });
      
      if (!response.ok) {
        throw new Error('Prediction failed');
      }
      
      const data = await response.json();
      setPrediction(data.predicted_digit);
      setConfidence((data.confidence * 100).toFixed(1));
      setConfidenceScores(data.confidence_scores);
    } catch (err) {
      setError(err.message || 'Failed to get prediction');
    } finally {
      setIsLoading(false);
    }
  };

  // Get confidence bar color based on percentage
  const getConfidenceColor = (score) => {
    if (score >= 0.8) return 'bg-red-500';
    if (score >= 0.5) return 'bg-orange-500';
    return 'bg-slate-500';
  };

  return (
    <div className="min-h-screen bg-slate-950 flex items-center justify-center p-4">
      {/* Background decorative elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 left-0 w-96 h-96 bg-red-600/5 rounded-full blur-3xl"></div>
        <div className="absolute bottom-0 right-0 w-96 h-96 bg-red-600/5 rounded-full blur-3xl"></div>
      </div>
      
      <div className="relative z-10 w-full max-w-4xl">
        {/* Main Dashboard Container */}
        <div className="backdrop-blur-xl bg-slate-900/60 border border-slate-700/50 rounded-3xl p-8 shadow-2xl">
          {/* Header */}
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold text-white mb-2 tracking-tight">
              <span className="text-red-500">DIGIT</span> RECOGNIZER
            </h1>
            <p className="text-slate-400 text-sm uppercase tracking-widest">
              Hand-Drawn Neural Network Classifier
            </p>
          </div>
          
          <div className="grid lg:grid-cols-2 gap-8">
            {/* Left Column - Canvas */}
            <div className="flex flex-col items-center">
              {/* Drawing Canvas Container */}
              <div className="relative group">
                <div className="absolute -inset-1 bg-gradient-to-r from-red-600 to-red-400 rounded-2xl blur opacity-25 group-hover:opacity-50 transition duration-300"></div>
                <div className="relative border-2 border-red-600/50 rounded-2xl overflow-hidden">
                  <canvas
                    ref={canvasRef}
                    className={`canvas-cursor touch-none rounded-2xl ${isDrawing ? 'glow-effect' : ''}`}
                    onMouseDown={startDrawing}
                    onMouseMove={draw}
                    onMouseUp={stopDrawing}
                    onMouseLeave={stopDrawing}
                    onTouchStart={startDrawing}
                    onTouchMove={draw}
                    onTouchEnd={stopDrawing}
                  />
                </div>
              </div>
              
              {/* Action Buttons */}
              <div className="flex gap-4 mt-6 w-full max-w-xs">
                <button
                  onClick={clearCanvas}
                  className="flex-1 px-6 py-3 bg-slate-800 hover:bg-slate-700 text-white font-semibold rounded-xl border border-slate-600 transition-all duration-200 hover:border-red-500/50 hover:shadow-lg hover:shadow-red-500/20 active:scale-95"
                >
                  <span className="flex items-center justify-center gap-2">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                    Clear
                  </span>
                </button>
                <button
                  onClick={predictDigit}
                  disabled={isLoading}
                  className="flex-1 px-6 py-3 bg-gradient-to-r from-red-600 to-red-500 hover:from-red-500 hover:to-red-400 text-white font-semibold rounded-xl transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-lg hover:shadow-red-500/30 active:scale-95"
                >
                  <span className="flex items-center justify-center gap-2">
                    {isLoading ? (
                      <>
                        <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                        </svg>
                        Processing...
                      </>
                    ) : (
                      <>
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                        </svg>
                        Predict
                      </>
                    )}
                  </span>
                </button>
              </div>
            </div>
            
            {/* Right Column - Results */}
            <div className="flex flex-col">
              {/* Prediction Result Card */}
              <div className="backdrop-blur-lg bg-slate-800/50 border border-slate-600/50 rounded-2xl p-6 mb-6">
                <h2 className="text-slate-400 text-xs font-semibold uppercase tracking-widest mb-4">
                  Prediction Result
                </h2>
                
                {error ? (
                  <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 text-red-400 text-center">
                    <svg className="w-8 h-8 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <p className="text-sm">{error}</p>
                  </div>
                ) : prediction !== null ? (
                  <div className="text-center">
                    <div className="text-8xl font-bold text-white mb-2 glow-effect">
                      {prediction}
                    </div>
                    <div className="text-slate-400 text-sm">
                      Confidence: <span className="text-red-400 font-semibold">{confidence}%</span>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <svg className="w-16 h-16 mx-auto text-slate-600 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122" />
                    </svg>
                    <p className="text-slate-500 text-sm">Draw a digit (0-9)</p>
                    <p className="text-slate-600 text-xs mt-1">to see the prediction</p>
                  </div>
                )}
              </div>
              
              {/* Confidence Breakdown */}
              <div className="backdrop-blur-lg bg-slate-800/50 border border-slate-600/50 rounded-2xl p-6 flex-1">
                <h2 className="text-slate-400 text-xs font-semibold uppercase tracking-widest mb-4">
                  Confidence Breakdown
                </h2>
                
                {Object.keys(confidenceScores).length > 0 ? (
                  <div className="space-y-3">
                    {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((digit) => {
                      const score = confidenceScores[digit.toString()] || 0;
                      const isPredicted = digit === prediction;
                      
                      return (
                        <div key={digit} className="flex items-center gap-3">
                          <span className={`w-6 h-6 flex items-center justify-center text-sm font-semibold rounded ${
                            isPredicted 
                              ? 'bg-red-500 text-white' 
                              : 'bg-slate-700 text-slate-400'
                          }`}>
                            {digit}
                          </span>
                          <div className="flex-1 h-3 bg-slate-700/50 rounded-full overflow-hidden">
                            <div 
                              className={`h-full confidence-bar ${getConfidenceColor(score)} ${isPredicted ? 'glow-effect' : ''}`}
                              style={{ width: `${score * 100}%` }}
                            />
                          </div>
                          <span className="text-xs text-slate-500 w-12 text-right">
                            {(score * 100).toFixed(1)}%
                          </span>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-full min-h-[200px]">
                    <p className="text-slate-600 text-sm text-center">
                      Confidence scores will appear<br />after prediction
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
          
          {/* Footer */}
          <div className="mt-8 pt-6 border-t border-slate-700/50 text-center">
            <p className="text-slate-500 text-xs">
              Trained on MNIST Dataset • MLP Neural Network • 3 Hidden Layers
            </p>
          </div>
        </div>
      </div>
      
      {/* Hidden canvas for processing */}
      <canvas ref={hiddenCanvasRef} style={{ display: 'none' }} />
    </div>
  );
}

export default App;