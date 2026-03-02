import React from 'react';
import { X, Maximize2, Minimize2 } from 'lucide-react';

export interface Shape {
  id: string;
  type: 'line' | 'circle' | 'rect' | 'text' | 'polygon' | 'arrow' | 'path' | 'triangle';
  x?: number;
  y?: number;
  x1?: number;
  y1?: number;
  x2?: number;
  y2?: number;
  r?: number;
  width?: number;
  height?: number;
  points?: string; // "x1,y1 x2,y2 ..."
  d?: string; // SVG path data
  content?: string;
  color?: string;
  strokeWidth?: number;
  fill?: string;
  label?: string;
  fontSize?: number;
}

export interface DiagramData {
  title: string;
  description?: string;
  shapes: Shape[];
  viewBox?: string; // "0 0 100 100"
}

interface DiagramBoardProps {
  data: DiagramData | null;
  onClose: () => void;
}

const DiagramBoard: React.FC<DiagramBoardProps> = ({ data, onClose }) => {
  const [isExpanded, setIsExpanded] = React.useState(false);

  if (!data) return null;

  const defaultViewBox = "0 0 400 300";
  const viewBox = data.viewBox || defaultViewBox;

  return (
    <div 
      className={`absolute transition-all duration-300 ease-in-out bg-white dark:bg-gray-900/95 backdrop-blur-md border border-indigo-500/30 shadow-2xl rounded-2xl overflow-hidden z-50 flex flex-col ${
        isExpanded 
          ? 'inset-4' 
          : 'bottom-32 right-4 w-80 h-64'
      }`}
    >
      {/* Header */}
      <div className="flex items-center justify-between p-3 bg-gray-100 dark:bg-gray-800/50 border-b border-gray-300 dark:border-gray-700">
        <h3 className="text-indigo-300 font-semibold text-sm truncate flex-1">
          {data.title || "知识点图示"}
        </h3>
        <div className="flex items-center gap-1">
          <button 
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-1.5 text-gray-400 dark:text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:text-white hover:bg-gray-200 dark:bg-gray-700 rounded-lg transition-colors"
          >
            {isExpanded ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
          </button>
          <button 
            onClick={onClose}
            className="p-1.5 text-gray-400 dark:text-gray-500 dark:text-gray-400 hover:text-red-400 hover:bg-red-900/20 rounded-lg transition-colors"
          >
            <X size={16} />
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-4 flex flex-col items-center justify-center bg-white dark:bg-gray-900 relative">
        {/* Grid Background */}
        <div className="absolute inset-0 opacity-10 pointer-events-none" 
             style={{ 
               backgroundImage: 'radial-gradient(#6366f1 1px, transparent 1px)', 
               backgroundSize: '20px 20px' 
             }} 
        />

        {(!data.shapes || !Array.isArray(data.shapes) || data.shapes.length === 0) ? (
          <div className="text-red-400 text-xs z-10 p-4 bg-black/50 rounded overflow-auto max-h-full max-w-full">
            <p>No valid shapes found. Raw data:</p>
            <pre>{JSON.stringify(data, null, 2)}</pre>
          </div>
        ) : (
          <>
            <svg 
              viewBox={viewBox} 
              className="w-full h-full max-w-full max-h-full z-10"
              preserveAspectRatio="xMidYMid meet"
              style={{ minHeight: '200px' }}
            >
              {/* Debug border to ensure SVG is rendering and taking up space */}
              {isExpanded && <rect x="0" y="0" width="100%" height="100%" fill="none" stroke="rgba(255,0,0,0.3)" strokeWidth="4" />}
              
              {data.shapes.map((shape, index) => {
                // Force a visible color if the model sends black or dark colors
                // Also check for 'stroke' property even though it's not in our type definition,
                // because the model might send it anyway based on standard SVG knowledge.
                let stroke = shape.color || (shape as any).stroke || "#e0e7ff";
                if (stroke === 'black' || stroke === '#000000' || stroke === '#000') {
                    stroke = '#e0e7ff';
                }
                const fill = shape.fill || "none";
                // Handle both strokeWidth and stroke-width (which the model might invent)
                let strokeWidth = shape.strokeWidth;
                if (strokeWidth === undefined) {
                  strokeWidth = (shape as any)['stroke-width'];
                }
                if (strokeWidth === undefined) {
                  strokeWidth = 4; // Default to 4 for better visibility
                }
                
                const key = shape.id || `shape-${index}`;
                
                const type = (shape.type || '').toLowerCase();

                switch (type) {
                  case 'line':
                    return (
                      <g key={key}>
                        <line 
                          x1={shape.x1 || 0} y1={shape.y1 || 0} 
                          x2={shape.x2 || 0} y2={shape.y2 || 0} 
                          stroke={stroke} 
                          strokeWidth={strokeWidth} 
                          strokeLinecap="round"
                        />
                        {shape.label && (
                          <text 
                            x={(shape.x1! + shape.x2!) / 2} 
                            y={(shape.y1! + shape.y2!) / 2 - 5} 
                            fill={stroke} 
                            fontSize={shape.fontSize || 14} 
                            textAnchor="middle"
                          >
                            {shape.label}
                          </text>
                        )}
                      </g>
                    );
                  case 'arrow':
                    return (
                      <g key={key}>
                        <defs>
                          <marker id={`arrowhead-${key}`} markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                            <polygon points="0 0, 10 3.5, 0 7" fill={stroke} />
                          </marker>
                        </defs>
                        <line 
                          x1={shape.x1 || 0} y1={shape.y1 || 0} 
                          x2={shape.x2 || 0} y2={shape.y2 || 0} 
                          stroke={stroke} 
                          strokeWidth={strokeWidth} 
                          markerEnd={`url(#arrowhead-${key})`}
                        />
                        {shape.label && (
                          <text 
                            x={(shape.x1! + shape.x2!) / 2} 
                            y={(shape.y1! + shape.y2!) / 2 - 5} 
                            fill={stroke} 
                            fontSize={shape.fontSize || 14} 
                            textAnchor="middle"
                          >
                            {shape.label}
                          </text>
                        )}
                      </g>
                    );
                  case 'circle':
                    return (
                      <g key={key}>
                        <circle 
                          cx={shape.x || 0} cy={shape.y || 0} r={shape.r || 10} 
                          stroke={stroke} 
                          strokeWidth={strokeWidth} 
                          fill={fill} 
                        />
                        {shape.label && (
                          <text 
                            x={shape.x || 0} y={(shape.y || 0) + (shape.r || 10) + 15} 
                            fill={stroke} 
                            fontSize={shape.fontSize || 14} 
                            textAnchor="middle"
                          >
                            {shape.label}
                          </text>
                        )}
                      </g>
                    );
                  case 'rect':
                    return (
                      <g key={key}>
                        <rect 
                          x={shape.x || 0} y={shape.y || 0} 
                          width={shape.width || 50} height={shape.height || 50} 
                          stroke={stroke} 
                          strokeWidth={strokeWidth} 
                          fill={fill} 
                          rx="4"
                        />
                        {shape.label && (
                          <text 
                            x={(shape.x || 0) + (shape.width || 50) / 2} 
                            y={(shape.y || 0) + (shape.height || 50) / 2} 
                            fill={stroke} 
                            fontSize={shape.fontSize || 14} 
                            textAnchor="middle" 
                            alignmentBaseline="middle"
                          >
                            {shape.label}
                          </text>
                        )}
                      </g>
                    );
                  case 'polygon':
                  case 'triangle': // Fallback for when the model invents a 'triangle' type
                    return (
                      <g key={key}>
                        <polygon 
                          points={shape.points || ""} 
                          stroke={stroke} 
                          strokeWidth={strokeWidth} 
                          fill={fill} 
                        />
                        {shape.label && (
                          <text 
                            x={10} y={10} 
                            fill={stroke} 
                            fontSize={shape.fontSize || 14} 
                          >
                            {shape.label}
                          </text>
                        )}
                      </g>
                    );
                  case 'path':
                    return (
                      <g key={key}>
                        <path 
                          d={shape.d || ""} 
                          stroke={stroke} 
                          strokeWidth={strokeWidth} 
                          fill={fill} 
                        />
                        {shape.label && (
                          <text 
                            x={10} y={10} 
                            fill={stroke} 
                            fontSize={shape.fontSize || 14} 
                          >
                            {shape.label}
                          </text>
                        )}
                      </g>
                    );
                  case 'text':
                    return (
                      <text 
                        key={key}
                        x={shape.x || 0} y={shape.y || 0} 
                        fill={stroke} 
                        fontSize={shape.fontSize || shape.height || 24} 
                        textAnchor="middle"
                      >
                        {shape.content || shape.label || (shape as any).text}
                      </text>
                    );
                  default:
                    return null;
                }
              })}
            </svg>
            
            {/* Debug View: Show raw data if expanded */}
            {isExpanded && (
              <div className="w-full mt-4 p-2 bg-black/80 rounded text-[10px] text-green-400 font-mono overflow-auto max-h-32 z-20">
                <pre>{JSON.stringify(data.shapes, null, 2)}</pre>
              </div>
            )}
          </>
        )}
      </div>
      
      {data.description && (
        <div className="p-3 bg-gray-100 dark:bg-gray-800/50 border-t border-gray-300 dark:border-gray-700 text-xs text-gray-600 dark:text-gray-300">
          {data.description}
        </div>
      )}
    </div>
  );
};

export default DiagramBoard;
