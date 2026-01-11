import React, { useEffect, useRef } from 'react';

interface AudioVisualizerProps {
  isActive?: boolean;
  analyser?: AnalyserNode | null;
  color?: string;
  width?: number;
  height?: number;
}

const AudioVisualizer: React.FC<AudioVisualizerProps> = ({ 
  isActive = false, 
  analyser, 
  color = '#3b82f6',
  width = 100,
  height = 60
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Number of bars to display
    const barCount = 5;
    let bars: number[] = Array(barCount).fill(height * 0.2); // Start with small height
    
    // Setup for Analyser
    let dataArray: Uint8Array;
    if (analyser) {
        // frequencyBinCount is half the fftSize
        dataArray = new Uint8Array(analyser.frequencyBinCount);
    }

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;
      const maxBarHeight = canvas.height * 0.8;

      if (analyser) {
          analyser.getByteFrequencyData(dataArray);
          
          // Calculate bars from frequency data
          // Use lower frequencies which are usually more dominant in voice
          const effectiveLength = Math.floor(dataArray.length * 0.7); 
          const step = Math.max(1, Math.floor(effectiveLength / barCount));
          
          bars = bars.map((current, i) => {
              let sum = 0;
              for (let j = 0; j < step; j++) {
                  // Access data from the array
                  const val = dataArray[i * step + j] || 0;
                  sum += val;
              }
              const avg = sum / step;
              
              // Scale 0-255 to bar height
              const target = (avg / 255) * maxBarHeight + (height * 0.1);
              
              // Smooth interpolation
              return current + (target - current) * 0.3;
          });

      } else {
          // Simulation Mode (Legacy/Fallback)
          bars = bars.map((h) => {
            if (!isActive) return Math.max(height * 0.1, h * 0.9);
            const target = Math.random() * maxBarHeight + (height * 0.2);
            return h + (target - h) * 0.2;
          });
      }

      ctx.fillStyle = color;
      bars.forEach((h, i) => {
        // Calculate x position centered
        const barWidth = width / (barCount * 1.5); // Leave some gap
        const gap = barWidth / 2;
        const totalWidth = barCount * barWidth + (barCount - 1) * gap;
        const startX = (width - totalWidth) / 2;
        
        const x = startX + i * (barWidth + gap);
        const y = centerY - h / 2;
        
        // Draw rounded rect
        ctx.beginPath();
        if (ctx.roundRect) {
            ctx.roundRect(x, y, barWidth, h, 4);
        } else {
            ctx.rect(x, y, barWidth, h);
        }
        ctx.fill();
      });

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => cancelAnimationFrame(animationRef.current);
  }, [isActive, analyser, color, width, height]);

  return <canvas ref={canvasRef} width={width} height={height} className="block" />;
};

export default AudioVisualizer;